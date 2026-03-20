from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routes.health import router as health_router
from .routes.history import router as history_router
from .routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables automatically when the app starts
    Base.metadata.create_all(bind=engine)
    yield
    # Nothing special to clean up on shutdown for now


# Create FastAPI application
app = FastAPI(
    title="NetGuard API",
    version="1.0",
    description="Explainable ML-based Network Fault Risk Assessment API",
    lifespan=lifespan,
)

# Allow frontend requests from browser
# For development this is open to all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    # Simple root endpoint to confirm API is running
    return {
        "message": "NetGuard API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "history": "/predictions",
    }


# Register route groups
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)