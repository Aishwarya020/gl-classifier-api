"""
GL Classifier — FastAPI Backend
================================
Exposes a single POST /classify endpoint that accepts the three CSV files,
runs the full three-tier classification pipeline, and returns JSON results.

The ANTHROPIC_API_KEY is stored as a server-side environment variable.
It is never sent to or accepted from the client.

Endpoints:
    GET  /         → health check
    POST /classify → run classification pipeline
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gl_classifier import classify_from_text

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GL Code Classifier",
    description="Three-tier GL code classification API for financial transactions.",
    version="3.0.0",
)

# ── CORS — allow the React frontend to call this API ─────────────────────────
# In production, replace "*" with your actual Vercel frontend URL, e.g.:
# allow_origins=["https://gl-classifier.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API key lives here only — never sent to the browser ──────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ════════════════════════════════════════════════════════════════════════════
class Transaction(BaseModel):
    date:       str
    description:         str
    amount:     float
    assigned_gl_code:    str
    gl_class:   str
    confidence_score: float
    match_method:        str
    reasoning:  str


class ClassifyResponse(BaseModel):
    total:       int
    l1_count:    int
    l2_count:    int
    l3_count:    int
    review_count: int
    transactions: list[Transaction]


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════
@app.get("/")
def health():
    """Health check — confirms the API is running."""
    return {
        "status":  "ok",
        "service": "GL Code Classifier API",
        "version": "3.0.0",
        "layer3_enabled": bool(ANTHROPIC_API_KEY),
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    classified: UploadFile = File(..., description="classified.csv — historical labelled transactions"),
    classify:   UploadFile = File(..., description="classify.csv   — new transactions to classify"),
    gl_dict:    UploadFile = File(..., description="gl_code_dictionary.csv — vendor→GL code mapping"),
):
    """
    Classify financial transactions using the three-tier pipeline.

    Accepts three CSV files as multipart/form-data.
    Returns classified transactions with GL codes, confidence scores, and reasoning.
    The Anthropic API key is sourced from the server environment — never from the request.
    """
    # ── Validate file types ───────────────────────────────────────────────
    for f in [classified, classify, gl_dict]:
        if not f.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail=f"'{f.filename}' is not a CSV file. All three uploads must be .csv"
            )

    # ── Read file contents ────────────────────────────────────────────────
    try:
        classified_text = (await classified.read()).decode("utf-8-sig")
        classify_text   = (await classify.read()).decode("utf-8-sig")
        gl_dict_text    = (await gl_dict.read()).decode("utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {e}")

    # ── Run classification pipeline ───────────────────────────────────────
    try:
        results = classify_from_text(
            classified_text=classified_text,
            classify_text=classify_text,
            gl_dict_text=gl_dict_text,
            api_key=ANTHROPIC_API_KEY,   # ← key comes from env, never from client
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification pipeline error: {e}")

    # ── Build summary counts ──────────────────────────────────────────────
    l1_count     = sum(1 for r in results if "Layer 1" in r.get("match_method", ""))
    l2_count     = sum(1 for r in results if "Layer 2" in r.get("match_method", ""))
    l3_count     = sum(1 for r in results if "Layer 3" in r.get("match_method", "") or "Claude" in r.get("match_method", ""))
    review_count = sum(1 for r in results if r.get("assigned_gl_code") == "REVIEW")

    return ClassifyResponse(
        total=len(results),
        l1_count=l1_count,
        l2_count=l2_count,
        l3_count=l3_count,
        review_count=review_count,
        transactions=[
            Transaction(
                date=r.get("date", ""),
                description=r.get("description", ""),
                amount=float(r.get("amount", 0)),
                assigned_gl_code=r.get("assigned_gl_code", "REVIEW"),
                gl_class=r.get("gl_class", ""),
                confidence_score=float(r.get("confidence_score", 0.1)),
                match_method=r.get("match_method", ""),
                reasoning=r.get("reasoning", ""),
            )
            for r in results
        ],
    )
