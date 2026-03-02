import pandas as pd
import sys
from src.utils.logger import logger
from src.utils.custom_exception import CustomException


# ============================================
# CLEAN PRODUCTS DATASET
# ============================================

def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting products preprocessing")

        # Remove duplicates
        products = products.drop_duplicates()

        # Remove completely empty rows
        products = products.dropna(how="all")

        # Fill missing text columns
        text_columns = [
            "product_name",
            "category",
            "sub_category",
            "brand",
            "description"
        ]

        for col in text_columns:
            if col in products.columns:
                products[col] = products[col].fillna("")

        # Fill numeric columns
        if "price" in products.columns:
            products["price"] = products["price"].fillna(
                products["price"].median()
            )

        if "rating" in products.columns:
            products["rating"] = products["rating"].fillna(
                products["rating"].mean()
            )

        products = products.reset_index(drop=True)

        logger.info("Products preprocessing completed")

        return products

    except Exception as e:
        logger.error("Error during products preprocessing")
        raise CustomException(e, sys)


# ============================================
# CLEAN INTERACTIONS DATASET
# ============================================

def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting interactions preprocessing")

        # Convert timestamp safely
        if "timestamp" in interactions.columns:
            interactions["timestamp"] = pd.to_datetime(
                interactions["timestamp"],
                errors="coerce"
            )

        # Drop invalid timestamps
        interactions = interactions.dropna(subset=["timestamp"])

        # Remove duplicates
        interactions = interactions.drop_duplicates()

        # Remove remaining nulls
        interactions = interactions.dropna()

        interactions = interactions.reset_index(drop=True)

        logger.info("Interactions preprocessing completed")

        return interactions

    except Exception as e:
        logger.error("Error during interactions preprocessing")
        raise CustomException(e, sys)


# ============================================
# VALIDATE INTERACTIONS PRODUCT IDS
# ============================================

def validate_interactions(
    products: pd.DataFrame,
    interactions: pd.DataFrame
) -> pd.DataFrame:
    try:
        logger.info("Validating product IDs in interactions")

        interactions = interactions[
            interactions["product_id"].isin(products["product_id"])
        ]

        logger.info("Validation completed")

        return interactions

    except Exception as e:
        logger.error("Error during interaction validation")
        raise CustomException(e, sys)


# ============================================
# OPTIONAL MERGE FUNCTION
# ============================================

def merge_datasets(
    products: pd.DataFrame,
    interactions: pd.DataFrame
) -> pd.DataFrame:
    try:
        logger.info("Merging products and interactions")

        merged_df = interactions.merge(
            products,
            on="product_id",
            how="inner"
        )

        logger.info("Merge successful")

        return merged_df

    except Exception as e:
        logger.error("Error during dataset merge")
        raise CustomException(e, sys)

import os


def save_processed_data(products, interactions):
    try:
        logger.info("Saving processed datasets")

        os.makedirs("data/processed", exist_ok=True)

        products.to_csv(
            "data/processed/products_processed.csv",
            index=False
        )

        interactions.to_csv(
            "data/processed/interactions_processed.csv",
            index=False
        )

        logger.info("Processed data saved successfully")

    except Exception as e:
        logger.error("Error saving processed data")
        raise CustomException(e, sys)