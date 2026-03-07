# GL Code Classifier — API Backend

Three-tier classification pipeline that assigns General Ledger codes to financial transactions. Built as a FastAPI backend for deployment on [Render.com](https://render.com).

## Architecture

```
React Frontend (Vercel)
        │
        │  POST /classify  (3 CSV files)
        ▼
FastAPI Backend (Render)
        │
        ├── Layer 1: Exact Match Cache       (fuzzy string ≥ 95%)
        ├── Layer 2: TF-IDF Embedding        (cosine similarity ≥ 0.40)
        └── Layer 3: Claude API Fallback     (novel vendors)
                          │
                          └── ANTHROPIC_API_KEY (env var, never in code)
```

**Security note:** The Anthropic API key is stored as a server-side environment variable on Render. It is never sent to or accepted from the client.

---

## Files

```
gl-classifier-api/
├── main.py            ← FastAPI server — endpoints, CORS, request handling
├── gl_classifier.py   ← Classification engine — all three layers
├── requirements.txt   ← Python dependencies
├── render.yaml        ← Render.com deployment config
├── .gitignore         ← Excludes .env, CSVs, __pycache__
└── README.md
```

---

## Local Development

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/gl-classifier-api.git
cd gl-classifier-api
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key (local only — never commit this)

```bash
# Create a .env file — already in .gitignore so it won't be committed
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

### 3. Run the server

```bash
# Load .env and start
export $(cat .env) && uvicorn main:app --reload
```

Server runs at `http://localhost:8000`

### 4. Test with curl

```bash
curl -X POST http://localhost:8000/classify \
  -F "classified=@classified.csv" \
  -F "classify=@classify.csv" \
  -F "gl_dict=@gl_code_dictionary.csv"
```

### 5. Interactive API docs

Open `http://localhost:8000/docs` — FastAPI auto-generates a Swagger UI where you can upload files and test the endpoint interactively.

---

## Deploy to Render.com (free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — GL Classifier API"
git remote add origin https://github.com/YOUR_USERNAME/gl-classifier-api.git
git push -u origin main
```

### Step 2 — Create a Render account

Go to [render.com](https://render.com) → sign up (free, no credit card needed).

### Step 3 — Connect your GitHub repo

1. Click **New +** → **Web Service**
2. Connect your GitHub account
3. Select the `gl-classifier-api` repository
4. Render will auto-detect `render.yaml` — all settings are pre-filled

### Step 4 — Add the API key as an environment variable

In the Render dashboard for your service:

1. Go to **Environment** tab
2. Click **Add Environment Variable**
3. Key: `ANTHROPIC_API_KEY`
4. Value: `sk-ant-your-key-here`
5. Click **Save Changes**

This is the only place the key lives. It never touches the frontend or the git repo.

### Step 5 — Deploy

Click **Deploy** — Render will:
1. Clone your repo
2. Run `pip install -r requirements.txt`
3. Start the server with `uvicorn main:app --host 0.0.0.0 --port $PORT`

Your API will be live at:
```
https://gl-classifier-api.onrender.com
```

---

## API Reference

### `GET /`
Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "GL Code Classifier API",
  "version": "3.0.0",
  "layer3_enabled": true
}
```

---

### `POST /classify`
Classify financial transactions from three CSV files.

**Request:** `multipart/form-data`

| Field        | Type | Description |
|---|---|---|
| `classified` | file | `classified.csv` — historical labelled transactions |
| `classify`   | file | `classify.csv` — new transactions to classify |
| `gl_dict`    | file | `gl_code_dictionary.csv` — vendor→GL code mapping |

**Response:**
```json
{
  "total": 290,
  "l1_count": 251,
  "l2_count": 5,
  "l3_count": 34,
  "review_count": 0,
  "transactions": [
    {
      "date": "11/4/2025",
      "desc_raw": "WALMART.COM WALMART.COM AR",
      "amount": 150.38,
      "gl_code": "6160",
      "gl_class": "6160 Office Supplies",
      "confidence": 0.99,
      "method": "Layer 1 — Exact Match",
      "reasoning": "Near-exact string match to historical record (100% similarity)."
    }
  ]
}
```

---

## Run as CLI (no server)

You can also run the classifier directly from the command line:

```bash
# Place your CSV files in the same directory, then:
python gl_classifier.py

# Or specify custom paths via env vars:
CLASSIFIED_CSV=path/to/classified.csv \
CLASSIFY_CSV=path/to/classify.csv \
GL_DICT_CSV=path/to/gl_code_dictionary.csv \
OUTPUT_CSV=output.csv \
ANTHROPIC_API_KEY=sk-ant-... \
python gl_classifier.py
```

Output is written to `classified_output.csv`.

---

## Classification Pipeline Detail

### Layer 1 — Exact Match Cache
- Uses `difflib.SequenceMatcher` for fuzzy string comparison
- Threshold: ≥ 95% similarity
- Typical result: 85–90% of transactions classified here
- Confidence reported: 97–99%

### Layer 2 — TF-IDF Embedding (v3)
- Blended char n-gram (45%) + word n-gram (55%) TF-IDF vectors
- Cosine similarity threshold: ≥ 0.40
- **Fix 1:** US state codes, phone numbers, zip codes stripped before vectorising
- **Fix 2:** First 2 tokens (vendor name) duplicated to boost TF weight
- Production upgrade: replace with dense neural embeddings (OpenAI `text-embedding-3-small`)

### Layer 3 — Claude API
- Batches of 15 transactions per API call
- Full GL dictionary + historical examples as context
- API key stored as server environment variable only

---

## Notes on the Free Render Tier

- **Cold starts:** Free tier services spin down after 15 minutes of inactivity. The first request after inactivity takes ~30 seconds. Subsequent requests are fast.
- **Memory:** 512MB RAM — sufficient for the TF-IDF index on 200–300 historical records.
- **Upgrade path:** Render's $7/month "Starter" tier eliminates cold starts.
