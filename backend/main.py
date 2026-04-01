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
    uvicorn.run(app, host="127.0.0.1", port=8000)
