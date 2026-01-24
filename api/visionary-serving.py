"""
Visionary Model Serving API

FastAPI application for serving the CatBoost price prediction model.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import mlflow.catboost
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Environment variables (configured via k8s/values.yaml -> ConfigMap)
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_STAGE = os.environ["MODEL_STAGE"]
MODEL_PATH = os.getenv("MODEL_PATH", "")

# Global model reference
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown."""
    global model
    
    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Determine model source
    if MODEL_PATH:
        model_uri = MODEL_PATH
        logger.info(f"Loading model from path: {model_uri}")
    else:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from MLflow registry: {model_uri}")
    
    try:
        model = mlflow.catboost.load_model(model_uri)
        logger.success(f"Model loaded successfully from: {model_uri}")
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


# ----- Required Features -----
# Must match the exact order used during model training

REQUIRED_FEATURES = [
    "origin",
    "destination",
    "days_before_departure",
    "airline",
    "stops",
    "flight_duration",
    "departure_day_of_week",
    "departure_is_weekend",
    "departure_time_hour",
    "departure_time_minute",
    "arrival_time_hour",
    "arrival_time_minute",
    "origin_country",
    "destination_country",
    "origin_departure_holidays",
    "destination_departure_holidays",
]


# ----- Request/Response Models -----

class FlightFeatures(BaseModel):
    """Feature set for a single flight prediction."""
    
    origin: str = Field(..., description="Origin airport code (e.g., 'CDG', 'MAD', 'LHR')")
    destination: str = Field(..., description="Destination airport code (e.g., 'CDG', 'MAD', 'LHR')")
    days_before_departure: int = Field(..., description="Number of days before departure (query_date to departure_date)")
    airline: str = Field(..., description="Airline code or name")
    stops: int = Field(..., description="Number of stops (0 = direct flight)")
    flight_duration: float = Field(..., description="Flight duration in hours")
    departure_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    departure_is_weekend: bool = Field(..., description="Whether departure is on a weekend (Saturday or Sunday)")
    departure_time_hour: int = Field(..., ge=0, le=23, description="Departure hour (0-23)")
    departure_time_minute: int = Field(..., ge=0, le=59, description="Departure minute (0-59)")
    arrival_time_hour: int = Field(..., ge=0, le=23, description="Arrival hour (0-23)")
    arrival_time_minute: int = Field(..., ge=0, le=59, description="Arrival minute (0-59)")
    origin_country: str = Field(..., description="ISO country code of origin airport (e.g., 'FR', 'ES', 'GB')")
    destination_country: str = Field(..., description="ISO country code of destination airport (e.g., 'FR', 'ES', 'GB')")
    origin_departure_holidays: str = Field(..., description="Holiday name at origin on departure date, or 'None' if not a holiday")
    destination_departure_holidays: str = Field(..., description="Holiday name at destination on departure date, or 'None' if not a holiday")


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    
    features: List[FlightFeatures] = Field(
        ...,
        description="List of flight features for prediction",
        example=[
            {
                "origin": "CDG",
                "destination": "LHR",
                "days_before_departure": 14,
                "airline": "British Airways",
                "stops": 0,
                "flight_duration": 1.25,
                "departure_day_of_week": 2,
                "departure_is_weekend": False,
                "departure_time_hour": 10,
                "departure_time_minute": 30,
                "arrival_time_hour": 11,
                "arrival_time_minute": 45,
                "origin_country": "FR",
                "destination_country": "GB",
                "origin_departure_holidays": "None",
                "destination_departure_holidays": "None",
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
    mlflow_tracking_uri: Optional[str] = Field(None, description="MLflow tracking server URL")


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
        # Convert Pydantic models to list of dicts and then to DataFrame
        features_data = [feature.model_dump() for feature in request.features]
        df = pd.DataFrame(features_data)
        
        # Ensure columns are in the expected order for the model
        df = df[REQUIRED_FEATURES]
        
        logger.info(f"Received prediction request with {len(df)} samples")
        logger.debug(f"Features: {df.columns.tolist()}")
        
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
            detail=f"Missing required feature: {e}. Required features: {REQUIRED_FEATURES}",
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
        mlflow_tracking_uri=MLFLOW_TRACKING_URI if model is not None else None,
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


@app.get(
    "/features",
    summary="Get required features",
    description="Returns the list of features required for prediction.",
)
async def get_features():
    """Return the list of required features for prediction."""
    
    return {
        "required_features": REQUIRED_FEATURES,
        "count": len(REQUIRED_FEATURES),
        "schema": {
            "origin": "string (categorical) - Origin airport code (e.g., 'CDG', 'MAD', 'LHR')",
            "destination": "string (categorical) - Destination airport code (e.g., 'CDG', 'MAD', 'LHR')",
            "days_before_departure": "integer (numerical) - Days between query and departure",
            "airline": "string (categorical) - Airline code or name",
            "stops": "integer (numerical) - Number of stops (0 = direct)",
            "flight_duration": "float (numerical) - Flight duration in hours",
            "departure_day_of_week": "integer (numerical) - Day of week (0=Monday, 6=Sunday)",
            "departure_is_weekend": "boolean (categorical) - Whether departure is on weekend",
            "departure_time_hour": "integer (numerical) - Departure hour (0-23)",
            "departure_time_minute": "integer (numerical) - Departure minute (0-59)",
            "arrival_time_hour": "integer (numerical) - Arrival hour (0-23)",
            "arrival_time_minute": "integer (numerical) - Arrival minute (0-59)",
            "origin_country": "string (categorical) - ISO country code of origin (e.g., 'FR', 'ES')",
            "destination_country": "string (categorical) - ISO country code of destination (e.g., 'GB', 'IT')",
            "origin_departure_holidays": "string (categorical) - Holiday name at origin or 'None'",
            "destination_departure_holidays": "string (categorical) - Holiday name at destination or 'None'",
        },
    }


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
