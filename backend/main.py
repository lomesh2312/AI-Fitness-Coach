import torch
import gc
import os

# DEPLOYMENT FIX: Minimize memory footprint to fit in Render's 512MB
torch.set_num_threads(1)
gc.collect()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import fitness, heart

app = FastAPI(title="AI Fitness Coach - Production Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fitness.router, prefix="/api", tags=["Fitness Stage 1"])
app.include_router(heart.router, prefix="/api", tags=["Heart Stage 2"])

@app.get("/")
async def root():
    return {"message": "AI Fitness Coach API is running!"}

if __name__ == "__main__":
    import uvicorn
    # DEPLOYMENT FIX: Must dynamically bind to $PORT or 8000, and use host 0.0.0.0
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
