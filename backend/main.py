import os
import gc
import torch

# ── Memory optimisation (must happen before heavy imports) ───────────────────
torch.set_num_threads(1)
gc.collect()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import fitness, heart

app = FastAPI(
    title="AI Fitness Coach — Production Pro",
    description="BMI analysis, heart risk prediction, and personalised fitness coaching.",
    version="2.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
frontend_url = os.environ.get("FRONTEND_URL", "https://ai-coach-lomesh-pro.netlify.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_url,
        "https://ai-coach-lomesh-pro.netlify.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fitness.router, prefix="/api", tags=["Fitness"])
app.include_router(heart.router,   prefix="/api", tags=["Heart Risk"])

@app.get("/")
async def root():
    llm_key_set = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return {
        "status":       "✅ AI Fitness Coach API is running!",
        "llm_enabled":  llm_key_set,
        "message":      "LLM active" if llm_key_set else "⚠️ OPENAI_API_KEY not set — using fallback mode",
        "version":      "2.0.0",
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
