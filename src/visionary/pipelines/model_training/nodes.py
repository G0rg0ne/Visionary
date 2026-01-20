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
    import pdb; pdb.set_trace()
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
        "cat_features": categorical_features if categorical_features else None,
    }
    
    # Remove None values from params
    catboost_params = {k: v for k, v in catboost_params.items() if v is not None}
    
    # Start MLflow experiment
    mlflow.set_experiment("visionary_price_prediction")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(catboost_params)
        mlflow.log_params({
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(X_train.columns),
            "n_categorical_features": len(categorical_features),
        })
        
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
        mlflow.catboost.log_model(model, name="model")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        
        mlflow.log_table(
            data=feature_importance,
            artifact_file="feature_importance.json",
        )
        
        logger.info("Model training completed!")
        logger.info(f"Train RMSE: {train_rmse:.2f}, Train R²: {train_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.4f}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"MLflow experiment: {mlflow.get_experiment_by_name('visionary_price_prediction').experiment_id}")
