"""
Model training nodes for the Visionary pipeline.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
import shap
import matplotlib.pyplot as plt
import tempfile
import os
from dotenv import load_dotenv



def log_shap_feature_importance(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    params: Dict[str, Any],
    top_n: int = 5,
) -> None:
    """
    Generate and log SHAP feature importance plot to MLflow.
    
    Args:
        model: Trained CatBoost model
        X_test: Test dataset features
        params: Dictionary containing parameters (for random_state)
        top_n: Number of top features to display (default: 5)
    """
    logger.info("Computing SHAP values...")
    # Use a sample of test data for SHAP computation (for efficiency)
    sample_size = min(100, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=params.get("random_state", 42))
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": X_test_sample.columns,
        "shap_importance": mean_shap_values,
    }).sort_values("shap_importance", ascending=False)
    
    # Get top N features
    top_features = shap_importance_df.head(top_n)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features["shap_importance"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.title(f"Top {top_n} Features by SHAP Importance", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot to temporary file and log to MLflow
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp_file.name, "shap_feature_importance")
        os.unlink(tmp_file.name)
    
    plt.close()
    
    logger.info("SHAP feature importance plot logged to MLflow")


def train_model(
    tickets_train_data: pd.DataFrame,
    tickets_test_data: pd.DataFrame,
    params: Dict[str, Any],
) -> None:
    """
    Train a CatBoost model and log the experiment to MLflow.
    
    Args:
        tickets_train_data: Training dataset with features and target
        tickets_test_data: Test dataset with features and target
        params: Dictionary containing model training parameters
    """
    # Identify target column (assuming 'price' is the target)
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    target_col = "price"

    categorical_features = tickets_train_data.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    
    # Convert NaN values in categorical features to strings (CatBoost requirement)
    for col in categorical_features:
        if tickets_train_data[col].isna().any():
            tickets_train_data[col] = tickets_train_data[col].fillna('None').astype(str)
        if tickets_test_data[col].isna().any():
            tickets_test_data[col] = tickets_test_data[col].fillna('None').astype(str)
    
    if target_col not in tickets_train_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    # Separate features and target
    X_train = tickets_train_data.drop(columns=[target_col])
    y_train = tickets_train_data[target_col]
    X_test = tickets_test_data.drop(columns=[target_col])
    y_test = tickets_test_data[target_col]
    
    # Identify categorical features
    categorical_features = X_train.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {len(X_train.columns) - len(categorical_features)}")
    
    # Prepare CatBoost parameters
    catboost_params = {
        "iterations": params.get("n_estimators", 100),
        "depth": params.get("max_depth", 10),
        "random_state": params.get("random_state", 42),
        "loss_function": "RMSE",
        "verbose": False,
        "grow_policy": "Lossguide",
        "max_leaves": 64,
        "bootstrap_type":"Bayesian",
        "bagging_temperature": 0.7,
        "random_strength": 1.5,
        "iterations": 2500,
        "learning_rate": 0.02,
        "cat_features": categorical_features if categorical_features else None,
        "allow_writing_files": False,
    }
    
    # Remove None values from params0
    catboost_params = {k: v for k, v in catboost_params.items() if v is not None}
    
    # Configure MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    
    # Start MLflow experiment
    mlflow.set_experiment("visionary_price_prediction")
    
    with mlflow.start_run(description=params.get("run_description", "")):
        # Log parameters
        mlflow.log_params(catboost_params)
        mlflow.log_params({
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(X_train.columns),
            "n_categorical_features": len(categorical_features),
        })
        
            # Log features used for training
        all_features = X_train.columns.tolist()
        numerical_features = [f for f in all_features if f not in categorical_features]
        
        features_metadata = pd.DataFrame({
            "feature": all_features,
            "type": ["categorical" if f in categorical_features else "numerical" for f in all_features],
        })
        
        mlflow.log_table(
            data=features_metadata,
            artifact_file="training_features.json",
        )
        
        logger.info(f"Logged {len(all_features)} training features to MLflow")
        
        # Initialize and train model
        logger.info("Training CatBoost model...")
        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            verbose=False,
        )
        
        # Make predictions
        logger.info("Making predictions...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })
        
        # Log model
        mlflow.catboost.log_model(model, artifact_path="model")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        
        mlflow.log_table(
            data=feature_importance,
            artifact_file="feature_importance.json",
        )
        
        # Generate and log SHAP feature importance plot
        log_shap_feature_importance(model, X_test, params, top_n=5)
        
        logger.info("Model training completed!")
        logger.info(f"Train RMSE: {train_rmse:.2f}, Train R²: {train_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.4f}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"MLflow experiment: {mlflow.get_experiment_by_name('visionary_price_prediction').experiment_id}")
