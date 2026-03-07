"""
GL Code Classification Engine  —  v3
======================================
Three-tier classification pipeline:

  LAYER 1 — Exact Match Cache
    Near-exact string match (≥95%) against historical transactions.
    Fastest path. Handles recurring vendors like Walmart, DoorDash, Google Ads.

  LAYER 2 — TF-IDF Embedding Similarity
    TF-IDF vector representations + cosine similarity.
    v3 fixes: location noise stripped, vendor prefix boosted.
    Better than fuzzy matching — finds 'DD DOORDASH LUNAGRI' → 6170
    even without an exact prior match.

  LAYER 3 — Claude API Fallback
    Novel transactions sent to Claude with full GL dictionary context.
    API key stored server-side only — never exposed to the client.

Public API:
    classify_from_text(classified_text, classify_text, gl_dict_text, api_key)
        → list[dict]  (one dict per transaction)

This module has NO file path dependencies. All I/O is handled by the caller
(main.py for the API, or the __main__ block for local CLI use).
"""

import csv
import io
import json
import re
import time
import warnings
from difflib import SequenceMatcher

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
MODEL          = "claude-sonnet-4-20250514"
L1_THRESHOLD   = 0.95   # fuzzy score  ≥ this → Layer 1
L2_THRESHOLD   = 0.40   # cosine sim   ≥ this → Layer 2
L2_CONF_HIGH   = 0.90   # cosine ≥ 0.80
L2_CONF_MED    = 0.75   # cosine 0.55–0.79
L2_CONF_LOW    = 0.62   # cosine 0.40–0.54
LLM_FLOOR_CONF = 0.50
BATCH_SIZE     = 15

# ════════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ════════════════════════════════════════════════════════════════════════════
def clean_amount(s: str) -> float:
    s = s.strip().replace(",", "").replace("$", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return 0.0


def normalise_desc(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ── Fix 1: Strip location noise ──────────────────────────────────────────────
# US state codes, phone numbers, and zip codes carry zero GL signal but
# heavily pollute TF-IDF cosine similarity (e.g. EDIBLE.COM → MAILCHIMP
# purely because both show "ATLANTA GA"). Strip them before vectorising.
_LOCATION_NOISE = re.compile(
    r'\b(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|'
    r'mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|'
    r'ut|vt|va|wa|wv|wi|wy|dc)\b'
    r'|\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b'
    r'|\b\d{5}(?:\-\d{4})?\b'
)

def strip_location_noise(desc_norm: str) -> str:
    cleaned = _LOCATION_NOISE.sub(" ", desc_norm)
    return re.sub(r"\s+", " ", cleaned).strip()


# ── Fix 2: Boost vendor prefix ───────────────────────────────────────────────
# The GL-relevant signal is always at the START of a bank string.
# Duplicating the first 2 tokens doubles their TF weight so the vendor name
# dominates cosine similarity over trailing location noise.
def boost_vendor_prefix(desc_norm: str) -> str:
    tokens = desc_norm.split()
    prefix = " ".join(tokens[:2])
    return f"{prefix} {prefix} {desc_norm}"


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (from CSV text strings, not file paths)
# ════════════════════════════════════════════════════════════════════════════
def _read_csv(text: str) -> list[dict]:
    """Parse CSV text (handles UTF-8 BOM, CRLF line endings)."""
    text = text.lstrip("\ufeff")
    return list(csv.DictReader(io.StringIO(text)))


def load_historical(text: str) -> list[dict]:
    records = []
    for row in _read_csv(text):
        desc = row.get("Description", "").strip()
        gl   = row.get("Assigned GL Code", "").strip()
        if desc and gl:
            records.append({
                "desc_raw":  desc,
                "desc_norm": normalise_desc(desc),
                "gl_code":   gl,
                "amount":    clean_amount(row.get(" Amount ", row.get("Amount", "0"))),
            })
    return records


def load_gl_dictionary(text: str) -> list[dict]:
    entries = []
    for row in _read_csv(text):
        vendor   = row.get("Billing Company (PPE)/Vendor", "").strip()
        gl_class = row.get("GL Class", "").strip()
        if not gl_class:
            continue
        gl_code = gl_class.split("-")[0].strip() if "-" in gl_class else gl_class.split()[0].strip()
        entries.append({
            "vendor_raw":  vendor,
            "vendor_norm": normalise_desc(vendor),
            "gl_code":     gl_code,
            "gl_class":    gl_class,
        })
    return entries


def load_to_classify(text: str) -> list[dict]:
    records = []
    for row in _read_csv(text):
        desc = row.get("Description", "").strip()
        if desc:
            records.append({
                "date":      row.get("Date", "").strip(),
                "desc_raw":  desc,
                "desc_norm": normalise_desc(desc),
                "amount":    clean_amount(row.get(" Amount ", row.get("Amount", "0"))),
            })
    return records


def build_code_to_class(gl_dict: list, historical: list) -> dict:
    mapping = {}
    for e in gl_dict:
        if e["gl_code"] not in mapping:
            mapping[e["gl_code"]] = e["gl_class"]
    for h in historical:
        if h["gl_code"] not in mapping:
            mapping[h["gl_code"]] = h["gl_code"]
    return mapping


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — EXACT MATCH CACHE
# ════════════════════════════════════════════════════════════════════════════
def layer1_exact_match(txn: dict, historical: list, code_to_class: dict) -> dict | None:
    best_score, best_gl = 0.0, None
    for h in historical:
        score = fuzzy_score(txn["desc_norm"], h["desc_norm"])
        if score > best_score:
            best_score, best_gl = score, h["gl_code"]

    if best_score >= L1_THRESHOLD:
        conf = round(min(0.97 + (best_score - L1_THRESHOLD) * 0.5, 0.99), 2)
        return {
            "gl_code":    best_gl,
            "gl_class":   code_to_class.get(best_gl, ""),
            "confidence": conf,
            "method":     "Layer 1 — Exact Match",
            "reasoning":  f"Near-exact string match to historical record ({best_score:.0%} similarity).",
        }
    return None


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — TF-IDF EMBEDDING INDEX
# ════════════════════════════════════════════════════════════════════════════
class EmbeddingIndex:
    """
    Sparse TF-IDF vector index over historical transaction descriptions.

    Improvements over plain fuzzy matching:
    - Tokenises into words: 'DOORDASH' becomes a shared signal across all
      DoorDash transactions regardless of which restaurant follows.
    - Fix 1 (strip_location_noise): removes state codes / phone numbers
      before vectorising → eliminates false location-driven matches.
    - Fix 2 (boost_vendor_prefix): duplicates first 2 tokens to amplify
      vendor name TF weight over trailing noise tokens.

    Production upgrade path: replace TF-IDF with dense neural embeddings
    (e.g. OpenAI text-embedding-3-small) for semantic matching without
    requiring shared tokens between query and corpus.
    """

    def __init__(self, historical: list):
        self.historical = historical
        corpus = [
            boost_vendor_prefix(strip_location_noise(h["desc_norm"]))
            for h in historical
        ]
        self.gl_codes = [h["gl_code"] for h in historical]

        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            min_df=1, sublinear_tf=True,
        )
        self.char_mat = self.char_vec.fit_transform(corpus)

        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=1, sublinear_tf=True,
        )
        self.word_mat = self.word_vec.fit_transform(corpus)

    def query(self, desc_norm: str) -> tuple[str, float, str]:
        """Return (gl_code, cosine_score, matched_desc_raw) for best match."""
        cleaned = boost_vendor_prefix(strip_location_noise(desc_norm))

        char_sims = cosine_similarity(
            self.char_vec.transform([cleaned]), self.char_mat
        ).flatten()
        word_sims = cosine_similarity(
            self.word_vec.transform([cleaned]), self.word_mat
        ).flatten()

        combined = 0.45 * char_sims + 0.55 * word_sims
        best_idx  = int(np.argmax(combined))
        return (
            self.gl_codes[best_idx],
            float(combined[best_idx]),
            self.historical[best_idx]["desc_raw"],
        )


def layer2_embedding(
    txn: dict,
    index: EmbeddingIndex,
    code_to_class: dict,
) -> dict | None:
    gl_code, cosine_sim, matched_raw = index.query(txn["desc_norm"])

    if cosine_sim < L2_THRESHOLD:
        return None

    conf = (
        L2_CONF_HIGH if cosine_sim >= 0.80 else
        L2_CONF_MED  if cosine_sim >= 0.55 else
        L2_CONF_LOW
    )
    return {
        "gl_code":    gl_code,
        "gl_class":   code_to_class.get(gl_code, ""),
        "confidence": conf,
        "method":     "Layer 2 — TF-IDF Embedding",
        "reasoning":  (
            f"TF-IDF cosine {cosine_sim:.2f} to: '{matched_raw[:50]}'. "
            f"Location noise stripped; vendor prefix boosted."
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — CLAUDE API FALLBACK
# ════════════════════════════════════════════════════════════════════════════
def _build_context(historical: list, gl_dict: list) -> str:
    code_examples: dict[str, list] = {}
    for h in historical:
        gl = h["gl_code"]
        if gl not in code_examples:
            code_examples[gl] = []
        if len(code_examples[gl]) < 3:
            code_examples[gl].append(h["desc_raw"][:50])

    seen: set = set()
    dict_lines = []
    for e in gl_dict:
        key = (e["gl_code"], e["gl_class"])
        if key not in seen:
            seen.add(key)
            dict_lines.append(f"  {e['gl_class']} | {e['vendor_raw']}")

    examples_lines = [
        f"  {gl}: {' | '.join(descs)}"
        for gl, descs in sorted(code_examples.items())
    ]
    return (
        "=== GL CODE DICTIONARY ===\n"
        + "\n".join(dict_lines[:80])
        + "\n\n=== HISTORICAL CLASSIFICATION EXAMPLES ===\n"
        + "\n".join(examples_lines)
    )


def _fallback() -> dict:
    return {
        "gl_code":    "REVIEW",
        "gl_class":   "Needs Manual Review",
        "confidence": 0.10,
        "method":     "No Match — Manual Review Required",
        "reasoning":  "Could not classify automatically. Please review manually.",
    }


def call_claude_batch(
    transactions: list,
    context: str,
    api_key: str,
) -> list[dict]:
    """Call Claude API for a batch of transactions. Returns one result per txn."""
    txn_lines = "\n".join(
        f"{i+1}. Date={t['date']} | Desc={t['desc_raw']} | Amount={t['amount']}"
        for i, t in enumerate(transactions)
    )

    prompt = f"""You are a financial GL code classifier for an insurance company.

{context}

Classify each transaction below. Return ONLY a JSON array — no preamble, no markdown fences.
Each element must have exactly these keys:
  "index"      : (int) 1-based
  "gl_code"    : (string) e.g. "6170"
  "gl_class"   : (string) e.g. "6170 Trvl Meals Entertain Admin"
  "confidence" : (float 0.0–1.0)
  "reasoning"  : (string) one sentence

Transactions:
{txn_lines}

Rules:
- AUTOPAY PAYMENT → gl_code "9999", gl_class "9999 Credit Card Payment", confidence 0.99
- Meals / restaurants / food → 6170
- Office supplies / retail / Amazon / Walmart → 6160
- Software / SaaS subscriptions → 6200
- Utilities → 6260 | Telephone → 6250 | Postage → 6140
- Outside services / contractors → 6115
- Licensing / permits / state fees → 6120
- Marketing / advertising → 5030
- Return exactly {len(transactions)} elements in the array.
"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      MODEL,
                "max_tokens": 2000,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$",        "", raw).strip()

        index_map = {r["index"]: r for r in json.loads(raw)}
        return [
            {
                "gl_code":    str(index_map[i+1].get("gl_code",    "UNKNOWN")),
                "gl_class":   str(index_map[i+1].get("gl_class",   "")),
                "confidence": float(index_map[i+1].get("confidence", LLM_FLOOR_CONF)),
                "method":     "Layer 3 — Claude API",
                "reasoning":  str(index_map[i+1].get("reasoning",  "Classified by LLM.")),
            }
            if i+1 in index_map else _fallback()
            for i in range(len(transactions))
        ]

    except Exception as e:
        print(f"  ⚠  Layer 3 batch error: {e}")
        return [_fallback() for _ in transactions]


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API  —  single entry point for FastAPI and CLI
# ════════════════════════════════════════════════════════════════════════════
def classify_from_text(
    classified_text: str,
    classify_text:   str,
    gl_dict_text:    str,
    api_key:         str = "",
) -> list[dict]:
    """
    Run the full three-tier classification pipeline on CSV text inputs.

    Parameters
    ----------
    classified_text : str   Contents of classified.csv
    classify_text   : str   Contents of classify.csv
    gl_dict_text    : str   Contents of gl_code_dictionary.csv
    api_key         : str   Anthropic API key (Layer 3). Empty → fallback only.

    Returns
    -------
    list[dict]  One dict per transaction with keys:
        date, desc_raw, amount, gl_code, gl_class,
        confidence, method, reasoning
    """
    # ── Load data ─────────────────────────────────────────────────────────
    historical    = load_historical(classified_text)
    gl_dict       = load_gl_dictionary(gl_dict_text)
    to_classify   = load_to_classify(classify_text)
    code_to_class = build_code_to_class(gl_dict, historical)

    # ── Build Layer 2 index (one-time cost per request) ───────────────────
    embed_index = EmbeddingIndex(historical)
    context     = _build_context(historical, gl_dict)

    results:   list         = [None] * len(to_classify)
    llm_queue: list[tuple]  = []   # (original_index, txn)
    counts = {"L1": 0, "L2": 0, "L3": 0}

    # ── Layer 1 + Layer 2 ─────────────────────────────────────────────────
    for i, txn in enumerate(to_classify):
        r = layer1_exact_match(txn, historical, code_to_class)
        if r:
            results[i] = {**txn, **r}
            counts["L1"] += 1
            continue

        r = layer2_embedding(txn, embed_index, code_to_class)
        if r:
            results[i] = {**txn, **r}
            counts["L2"] += 1
            continue

        llm_queue.append((i, txn))
        counts["L3"] += 1

    # ── Layer 3 ───────────────────────────────────────────────────────────
    for batch_start in range(0, len(llm_queue), BATCH_SIZE):
        batch = llm_queue[batch_start: batch_start + BATCH_SIZE]
        idxs  = [b[0] for b in batch]
        txns  = [b[1] for b in batch]

        if api_key:
            llm_results = call_claude_batch(txns, context, api_key)
        else:
            llm_results = [_fallback() for _ in txns]

        for orig_idx, txn, llm_res in zip(idxs, txns, llm_results):
            results[orig_idx] = {**txn, **llm_res}

        if api_key and batch_start + BATCH_SIZE < len(llm_queue):
            time.sleep(0.4)   # gentle rate-limit buffer

    print(
        f"Classification complete — "
        f"L1:{counts['L1']} L2:{counts['L2']} L3:{counts['L3']} "
        f"/ {len(to_classify)} total"
    )
    return results


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT  (python gl_classifier.py)
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    import sys

    CLASSIFIED_CSV = os.getenv("CLASSIFIED_CSV", "classified.csv")
    CLASSIFY_CSV   = os.getenv("CLASSIFY_CSV",   "classify.csv")
    GL_DICT_CSV    = os.getenv("GL_DICT_CSV",     "gl_code_dictionary.csv")
    OUTPUT_CSV     = os.getenv("OUTPUT_CSV",      "classified_output.csv")
    API_KEY        = os.getenv("ANTHROPIC_API_KEY", "")

    for path in [CLASSIFIED_CSV, CLASSIFY_CSV, GL_DICT_CSV]:
        if not os.path.exists(path):
            print(f"❌  File not found: {path}")
            sys.exit(1)

    print("🔄  Loading CSV files…")
    classified_text = open(CLASSIFIED_CSV, encoding="utf-8-sig").read()
    classify_text   = open(CLASSIFY_CSV,   encoding="utf-8-sig").read()
    gl_dict_text    = open(GL_DICT_CSV,    encoding="utf-8-sig").read()

    print("⚙️   Running classification pipeline…")
    results = classify_from_text(classified_text, classify_text, gl_dict_text, API_KEY)

    fieldnames = ["Date", "Description", "Amount", "Assigned_GL_Code",
                  "GL_Class", "Confidence_Score", "Match_Method", "Reasoning"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Date":             r["date"],
                "Description":      r["desc_raw"],
                "Amount":           r["amount"],
                "Assigned_GL_Code": r["gl_code"],
                "GL_Class":         r.get("gl_class", ""),
                "Confidence_Score": f"{r['confidence']:.0%}",
                "Match_Method":     r["method"],
                "Reasoning":        r["reasoning"],
            })

    high   = sum(1 for r in results if r["confidence"] >= 0.85)
    medium = sum(1 for r in results if 0.60 <= r["confidence"] < 0.85)
    low    = sum(1 for r in results if r["confidence"] < 0.60)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅  DONE  —  {len(results)} transactions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 High   (≥85%) : {high}
🟡 Medium (60-84%): {medium}
🔴 Low    (<60%) : {low}
📄  Output → {OUTPUT_CSV}
""")
