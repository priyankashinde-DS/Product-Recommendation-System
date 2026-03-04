import os
import sys
import pickle
import pandas as pd
from pathlib import Path

from src.utils.logger import logger
from src.utils.custom_exception import CustomException


class Recommender:
    def __init__(
        self,
        artifacts_dir=None,
        products_file=None
    ):
        """
        Initialize Recommender class with paths.
        If paths are None, it will resolve automatically relative to project root.
        """
        try:
            # Resolve project root dynamically
            project_root = Path(__file__).resolve().parent.parent.parent

            # Artifacts directory and products CSV
            self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else project_root / "artifacts"
            self.products_file = Path(products_file) if products_file else project_root / "data" / "processed" / "products_processed.csv"

            self.similarity_matrix = None
            self.products = None

            # Validate paths exist
            if not self.artifacts_dir.exists():
                raise FileNotFoundError(f"Artifacts directory not found: {self.artifacts_dir}")
            if not self.products_file.exists():
                raise FileNotFoundError(f"Products file not found: {self.products_file}")

        except Exception as e:
            raise CustomException(e, sys)

    def load_artifacts(self):
        """
        Load precomputed artifacts: similarity matrix and products DataFrame
        """
        try:
            similarity_path = self.artifacts_dir / "cosine_similarity.pkl"
            if not similarity_path.exists():
                raise FileNotFoundError(f"Cosine similarity file not found: {similarity_path}")

            with open(similarity_path, "rb") as f:
                self.similarity_matrix = pickle.load(f)

            self.products = pd.read_csv(self.products_file)

            logger.info("Artifacts loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def recommend(self, product_id, top_k=5):
        """
        Recommend top_k similar products for a given product_id
        """
        try:
            if self.similarity_matrix is None or self.products is None:
                raise ValueError("Artifacts not loaded. Call load_artifacts() first.")

            if product_id not in self.products["product_id"].values:
                raise ValueError(f"Product ID {product_id} not found in products list.")

            idx = self.products[self.products["product_id"] == product_id].index[0]

            scores = list(enumerate(self.similarity_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]

            product_indices = [i[0] for i in scores]

            return self.products.iloc[product_indices][
                ["product_id", "product_name", "brand", "category"]
            ]

        except Exception as e:
            raise CustomException(e, sys)

