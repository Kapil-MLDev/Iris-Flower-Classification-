from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import joblib
import numpy as np
from typing import List, Optional
import os

# Global variable for model
model = None
label_encoder_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    try:
        model_path = "best_model.joblib"
        if not os.path.exists(model_path):
            print(f"WARNING: Model file '{model_path}' not found in current directory")
            print(f"Current directory: {os.getcwd()}")
            print(f"Please place 'best_model.joblib' in: {os.path.abspath(model_path)}")
            # Don't raise error, allow API to start for testing
        else:
            model = joblib.load(model_path)
            print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    yield  # API runs here
    
    # Shutdown: Cleanup (if needed)
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Iris Flower Classification API",
    description="API for predicting Iris flower species using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Input validation schema (updated for Pydantic V2)
class PredictionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2
            }
        }
    )
    
    SepalLengthCm: float = Field(..., description="Sepal length in centimeters", ge=0, le=10)
    SepalWidthCm: float = Field(..., description="Sepal width in centimeters", ge=0, le=10)
    PetalLengthCm: float = Field(..., description="Petal length in centimeters", ge=0, le=10)
    PetalWidthCm: float = Field(..., description="Petal width in centimeters", ge=0, le=10)

# Batch prediction input
class BatchPredictionInput(BaseModel):
    samples: List[PredictionInput] = Field(..., description="List of samples to predict")

# Output schema
class PredictionOutput(BaseModel):
    prediction: str = Field(..., description="Predicted iris species")
    prediction_label: int = Field(..., description="Numeric label (0, 1, or 2)")
    confidence: Optional[float] = Field(None, description="Prediction confidence/probability")
    all_probabilities: Optional[dict] = Field(None, description="Probabilities for all classes")

# Batch output schema
class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_samples: int

# Health check endpoint
@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy",
        "message": "Iris Flower Classification API is running",
        "model_loaded": model is not None,
        "current_directory": os.getcwd()
    }

# Main prediction endpoint
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: PredictionInput):
    """
    Predict the species of an Iris flower based on its measurements
    
    - **SepalLengthCm**: Length of the sepal in centimeters
    - **SepalWidthCm**: Width of the sepal in centimeters
    - **PetalLengthCm**: Length of the petal in centimeters
    - **PetalWidthCm**: Width of the petal in centimeters
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure 'best_model.joblib' is in the same directory as app.py"
        )
    
    try:
        # Convert input to numpy array in the correct order
        features = np.array([[
            input_data.SepalLengthCm,
            input_data.SepalWidthCm,
            input_data.PetalLengthCm,
            input_data.PetalWidthCm
        ]])
        
        # Make prediction
        prediction_label = model.predict(features)[0]
        prediction_name = label_encoder_classes[int(prediction_label)]
        
        # Get prediction probabilities if available
        confidence = None
        all_probs = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[int(prediction_label)])
            all_probs = {
                label_encoder_classes[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        
        return PredictionOutput(
            prediction=prediction_name,
            prediction_label=int(prediction_label),
            confidence=confidence,
            all_probabilities=all_probs
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
def predict_batch(input_data: BatchPredictionInput):
    """
    Predict species for multiple Iris flower samples at once
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure 'best_model.joblib' is in the same directory as app.py"
        )
    
    try:
        predictions = []
        for sample in input_data.samples:
            result = predict(sample)
            predictions.append(result)
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_samples=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

# Model information endpoint
@app.get("/model-info", tags=["Model Info"])
def model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure 'best_model.joblib' is in the same directory as app.py"
        )
    
    # Get model type name
    model_type = type(model).__name__
    
    # If it's a pipeline, get the final estimator
    if hasattr(model, 'named_steps'):
        final_step = list(model.named_steps.keys())[-1]
        model_type = f"Pipeline with {type(model.named_steps[final_step]).__name__}"
    
    return {
        "model_type": model_type,
        "problem_type": "classification",
        "features": [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ],
        "target_classes": label_encoder_classes,
        "number_of_features": 4,
        "number_of_classes": 3,
        "model_description": "Machine learning model trained to classify Iris flower species"
    }

# Get available species
@app.get("/species", tags=["Model Info"])
def get_species():
    """
    Get list of all possible Iris species that can be predicted
    """
    return {
        "species": label_encoder_classes,
        "encoding": {
            0: label_encoder_classes[0],
            1: label_encoder_classes[1],
            2: label_encoder_classes[2]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)