from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import fitness, heart

app = FastAPI(title="AI Fitness Coach - Production Pro")

import os

frontend_url = os.environ.get("FRONTEND_URL", "https://ai-coach-lomesh-pro.netlify.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:8000", "http://127.0.0.1:8000"],
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
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
