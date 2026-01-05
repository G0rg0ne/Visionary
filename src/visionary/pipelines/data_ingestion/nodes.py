"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""
import pandas as pd
import plotly.express as px
import boto3
from typing import Dict, Any, Optional
from loguru import logger
import os
from pathlib import Path
from dotenv import load_dotenv


def connect_to_minio(
    minio_credentials: Dict[str, Any],
) -> boto3.client:
    """
    Connect to a MinIO server using credentials from config and/or environment variables.
    
    Args:
        minio_credentials: Dictionary containing MinIO configuration from config.
                          Required keys: endpoint_url, bucket_name
                          Optional keys: access_key, secret_key, prefix, region_name
                          Note: access_key and secret_key can also be provided via
                          environment variables (takes precedence over config)
    
    Returns:
        boto3.client: Configured S3-compatible client ready to use with MinIO
    
    Example parameters.yml structure:
        minio_credentials:
            endpoint_url: http://localhost:9000  # Required
            bucket_name: my-bucket  # Required
            prefix: data/  # Optional
            region_name: us-east-1  # Optional, defaults to us-east-1
            access_key: minioadmin  # Optional if provided via env var
            secret_key: minioadmin  # Optional if provided via env var
    
    Environment variables (loaded from .env file):
        access_key: MinIO access key (optional if in config)
        secret_key: MinIO secret key (optional if in config)
    """
    try:
        project_root =  Path(__file__).parent.parent.parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Environment file loaded from {env_path}")
        else:
            logger.warning(f"Environment file not found at {env_path}, using credentials from config")
        
        # Extract credentials - prefer environment variables, fallback to config
        access_key = os.getenv("access_key") or minio_credentials.get("access_key")
        secret_key = os.getenv("secret_key") or minio_credentials.get("secret_key")
        endpoint_url = minio_credentials.get("endpoint_url")
        bucket_name = minio_credentials.get("bucket_name")
        prefix = minio_credentials.get("prefix")
        
        # Validate required credentials
        if not access_key or not secret_key:
            raise ValueError(
                "Missing required credentials: access_key and secret_key. "
                "Provide them via environment variables or minio_credentials config."
            )
        
        # Validate endpoint URL (required for MinIO)
        if not endpoint_url:
            raise ValueError(
                "Missing required endpoint_url. MinIO requires an endpoint URL "
                "(e.g., http://localhost:9000 or https://your-minio-server:9000)"
            )
        
        # Validate bucket_name (required for bucket operations)
        if not bucket_name:
            raise ValueError(
                "Missing required bucket_name in minio_credentials config."
            )
        
        # Get optional parameters
        region = minio_credentials.get("region_name", "us-east-1")
        
        # Configure client kwargs for MinIO

        client_kwargs = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "endpoint_url": endpoint_url,
            "region_name": region,
        }
        
        logger.info(f"Connecting to MinIO endpoint: {endpoint_url}")
        if prefix:
            logger.info(f"Using prefix: {prefix}")
        
        # Create S3-compatible client for MinIO
        minio_client = boto3.client("s3", **client_kwargs)
        
        # Test connection by checking if bucket exists
        try:
            minio_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to MinIO bucket: {bucket_name}")
        except Exception as e:
            logger.warning(
                f"Could not verify bucket '{bucket_name}' exists. "
                f"Connection established but bucket check failed: {str(e)}"
            )
        
        return minio_client
        
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {str(e)}")
        raise


def load_csv_files_from_minio(
    minio_client: boto3.client,
    minio_credentials: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from MinIO bucket with the specified prefix.
    
    Args:
        minio_client: Configured boto3 S3-compatible client for MinIO
        minio_credentials: Dictionary containing MinIO configuration.
                          Required keys: bucket_name, prefix
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping filename to DataFrame.
                                 Keys are the CSV filenames (without path).
    
    Example:
        {
            'file1.csv': DataFrame(...),
            'file2.csv': DataFrame(...),
        }
    """
    try:
        bucket_name = minio_credentials.get("bucket_name")
        prefix = minio_credentials.get("prefix", "")
        
        if not bucket_name:
            raise ValueError("Missing required bucket_name in minio_credentials config.")
        
        logger.info(f"Listing CSV files in bucket '{bucket_name}' with prefix '{prefix}'")
        
        # List all objects with the prefix
        paginator = minio_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        csv_files = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Filter for CSV files only
                    if key.lower().endswith('.csv'):
                        csv_files.append(key)
        
        if not csv_files:
            logger.warning(f"No CSV files found in bucket '{bucket_name}' with prefix '{prefix}'")
            return {}
        
        logger.info(f"Found {len(csv_files)} CSV file(s) to load")
        
        # Load each CSV file into a DataFrame
        dataframes = {}
        for csv_key in csv_files:
            try:
                # Get the filename without the prefix path
                filename = csv_key.split('/')[-1] if '/' in csv_key else csv_key
                
                logger.info(f"Loading CSV file: {csv_key}")
                
                # Download the CSV file content
                response = minio_client.get_object(Bucket=bucket_name, Key=csv_key)
                
                # Read CSV into DataFrame
                df = pd.read_csv(response['Body'])
                
                dataframes[filename] = df
                logger.info(f"Successfully loaded '{filename}' with {len(df)} rows and {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Failed to load CSV file '{csv_key}': {str(e)}")
                # Continue loading other files even if one fails
                continue
        
        logger.info(f"Successfully loaded {len(dataframes)} CSV file(s)")
        return dataframes
        
    except Exception as e:
        logger.error(f"Failed to load CSV files from MinIO: {str(e)}")
        raise


def load_csv_files_from_minio_combined(
    minio_credentials: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Connect to MinIO and load all CSV files from the specified bucket and prefix.
    This is a combined function that handles both connection and loading to avoid
    serialization issues with boto3.client objects.
    
    Args:
        minio_credentials: Dictionary containing MinIO configuration from config.
                          Required keys: endpoint_url, bucket_name
                          Optional keys: access_key, secret_key, prefix, region_name
                          Note: access_key and secret_key can also be provided via
                          environment variables (takes precedence over config)
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping filename to DataFrame.
                                 Keys are the CSV filenames (without path).
    
    Example:
        {
            'file1.csv': DataFrame(...),
            'file2.csv': DataFrame(...),
        }
    """
    # Connect to MinIO
    minio_client = connect_to_minio(minio_credentials)
    
    # Load CSV files
    data = load_csv_files_from_minio(minio_client, minio_credentials)
    return data