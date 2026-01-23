"""
Visionary Model Serving API

FastAPI application for serving the CatBoost price prediction model.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import mlflow.catboost
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Environment configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "visionary_price_predictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Global model reference
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown."""
    global model
    
    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Load model from MLflow registry
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Loading model from: {model_uri}")
    
    try:
        model = mlflow.catboost.load_model(model_uri)
        logger.success(f"Model loaded successfully: {MODEL_NAME} ({MODEL_STAGE})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model from {model_uri}: {e}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Visionary API...")
    model = None


# Initialize FastAPI application
app = FastAPI(
    title="Visionary Price Predictor",
    description="ML-powered price prediction API using CatBoost",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Request/Response Models -----

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    
    features: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries for prediction",
        example=[
            {
                "feature_1": 100,
                "feature_2": "category_a",
                "feature_3": 25.5,
            }
        ],
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predictions: List[float] = Field(
        ...,
        description="List of predicted prices",
    )
    count: int = Field(
        ...,
        description="Number of predictions returned",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    model_stage: Optional[str] = Field(None, description="Model stage")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    detail: str = Field(..., description="Error message")


# ----- API Endpoints -----

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    summary="Get price predictions",
    description="Submit features and receive price predictions from the CatBoost model.",
)
async def predict(request: PredictionRequest):
    """Generate predictions for the provided features."""
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up.",
        )
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame(request.features)
        logger.info(f"Received prediction request with {len(df)} samples")
        
        # Generate predictions
        predictions = model.predict(df).tolist()
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return PredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )
        
    except KeyError as e:
        logger.warning(f"Missing feature in request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required feature: {e}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and the model is loaded.",
)
async def health():
    """Return health status of the service."""
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name=MODEL_NAME if model is not None else None,
        model_stage=MODEL_STAGE if model is not None else None,
    )


@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information.",
)
async def root():
    """Root endpoint with API information."""
    
    return {
        "service": "Visionary Price Predictor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get(
    "/ready",
    summary="Readiness check",
    description="Check if the service is ready to accept requests.",
)
async def ready():
    """Kubernetes readiness probe endpoint."""
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    
    return {"status": "ready"}


# ----- Main Entry Point -----

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "visionary-serving:app",
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", "8000")),
        workers=int(os.getenv("UVICORN_WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        reload=False,
    )
