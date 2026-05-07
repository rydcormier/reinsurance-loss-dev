"""
FastAPI serving layer for loss reserve prediction.

Provides REST API for uploading triangles, generating predictions,
retrieving historical results, and managing models.
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd

# TODO: Import actual model and ingestion modules once implemented
# from ingestion.triangles import LossTriangle, parse_csv, parse_excel, validate
# from features.engineering import build_feature_panel
# from models.mlmodel import LossDevModel
# from models.chainladder import ChainLadder
# from evaluation.metrics import evaluate, compare_to_chainladder, divergence_flag


# ============================================================================
# Pydantic Response Models
# ============================================================================


class ValidationResult(BaseModel):
    """Result of triangle validation."""

    is_valid: bool
    errors: list[str] = []


class TriangleMetadata(BaseModel):
    """Metadata about an uploaded triangle."""

    triangle_id: str
    line_of_business: str
    origin_years: list[int]
    development_periods: list[int]
    upload_timestamp: str


class PredictionResult(BaseModel):
    """Prediction output from model."""

    triangle_id: str
    point_estimate: dict  # {origin_year: ultimate_loss}
    lower_bound_10pct: dict
    upper_bound_90pct: dict
    chain_ladder_estimate: dict
    divergence_flags: dict  # {origin_year: bool}
    ml_vs_cl_comparison: dict


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"


# ============================================================================
# Lifespan Context Manager
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown event handler for FastAPI app.

    Startup:
    - Load trained ML model from disk
    - Load or initialize chain-ladder baseline
    - Initialize triangle storage dictionary

    Shutdown:
    - Save any in-memory state
    - Close database connections if used

    TODO:
        - On startup:
            - Load LossDevModel from models/saved/model.pkl (if exists)
            - Initialize ChainLadder baseline
            - Set app.state.model = loaded_model
            - Set app.state.baseline = chainladder_model
            - Initialize app.state.triangles = {} (in-memory cache)
        - Log startup message
        - yield (resume app execution)
        - On shutdown:
            - Log shutdown message
    """
    # Startup
    print("Starting reinsurance-loss-dev API...")
    # TODO: Load models here
    yield
    # Shutdown
    print("Shutting down reinsurance-loss-dev API...")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="Reinsurance Loss Development API",
    description="Reserve adequacy prediction and model serving",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status and version

    TODO:
        - Return {"status": "healthy", "version": "0.1.0"}
    """
    raise NotImplementedError("GET /health stub — implement health check")


@app.post("/triangles/upload", response_model=dict)
async def upload_triangle(
    file: UploadFile = File(...), line_of_business: str = "Workers Compensation"
):
    """
    Upload a loss triangle from CSV or Excel file.

    Accepts CSV or Excel files, parses the triangle, validates it,
    stores it in memory/database, and returns triangle_id.

    Args:
        file: CSV or Excel file with loss triangle
        line_of_business: LOB for context (e.g., "Workers Compensation")

    Returns:
        {
            "triangle_id": "<uuid>",
            "line_of_business": "<lob>",
            "validation": ValidationResult,
            "origin_years": [2015, 2016, ...],
            "development_periods": [12, 24, 36, ...]
        }

    Raises:
        HTTPException 400 if validation fails
        HTTPException 500 if parsing fails

    TODO:
        - Generate unique triangle_id = uuid.uuid4()
        - Save uploaded file to temp location
        - Determine file type from file.filename (.csv or .xlsx)
        - Call parse_csv() or parse_excel()
        - Call validate() on parsed triangle
        - If validation fails, return 400 with errors
        - Store triangle in app.state.triangles[triangle_id]
        - Return triangle_id, LOB, metadata
    """
    raise NotImplementedError("POST /triangles/upload stub — implement file upload")


@app.post("/predict", response_model=PredictionResult)
async def predict(
    triangle_id: str,
    use_uncertainty: bool = True,
):
    """
    Generate predictions for an uploaded triangle.

    Takes a parsed triangle, generates features, runs both ML and chain-ladder
    models, compares results, and flags significant divergences.

    Args:
        triangle_id: ID from previous upload
        use_uncertainty: Include prediction intervals (default True)

    Returns:
        PredictionResult with ML predictions, CL baseline, divergence flags

    Raises:
        HTTPException 404 if triangle_id not found
        HTTPException 500 if prediction fails

    TODO:
        - Retrieve triangle from app.state.triangles[triangle_id]
        - Call features.engineering.build_feature_panel(triangle)
        - Run ML model: app.state.model.predict_with_uncertainty(features)
        - Run chain-ladder baseline: app.state.baseline.predict(triangle)
        - Call evaluation.metrics.evaluate() for ML predictions
        - Call evaluation.metrics.compare_to_chainladder()
        - For each origin_year, call divergence_flag()
        - Construct PredictionResult response
        - Store result in app.state.triangles[triangle_id]["last_prediction"]
    """
    raise NotImplementedError("POST /predict stub — implement prediction endpoint")


@app.get("/triangles", response_model=list[TriangleMetadata])
async def list_triangles():
    """
    List all uploaded triangles with metadata.

    Returns:
        List of TriangleMetadata objects

    TODO:
        - Iterate over app.state.triangles
        - For each, extract metadata (id, LOB, years, periods, timestamp)
        - Return list of TriangleMetadata
    """
    raise NotImplementedError("GET /triangles stub — implement triangle listing")


@app.get("/triangles/{triangle_id}", response_model=dict)
async def get_triangle(triangle_id: str):
    """
    Retrieve a specific triangle and its last prediction.

    Args:
        triangle_id: Triangle ID from upload

    Returns:
        {
            "triangle": TriangleMetadata,
            "last_prediction": PredictionResult or null
        }

    Raises:
        HTTPException 404 if triangle_id not found

    TODO:
        - Retrieve triangle from app.state.triangles[triangle_id]
        - If not found, raise HTTPException(404)
        - Construct response with metadata and last_prediction (if exists)
    """
    raise NotImplementedError("GET /triangles/{triangle_id} stub — implement retrieval")


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Generic HTTP exception handler."""
    return {"detail": exc.detail, "status_code": exc.status_code}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
