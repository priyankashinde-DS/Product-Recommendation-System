import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import logger
from src.utils.custom_exception import CustomException


class Vectorizer:

    def __init__(self, artifacts_path="artifacts"):
        self.artifacts_path = artifacts_path
        os.makedirs(self.artifacts_path, exist_ok=True)

    def run(self, products_path="data/processed/products_processed.csv"):
        try:
            logger.info("Starting feature engineering")

            # Load processed products
            products = pd.read_csv(products_path)

            # Fill missing values to avoid NaN issues
            products = products.fillna("")

            # Combine important text columns into one feature
            products["combined_features"] = (
                products["category"].astype(str) + " " +
                products["sub_category"].astype(str) + " " +
                products["brand"].astype(str) + " " +
                products["description"].astype(str)
            )

            # Initialize TF-IDF vectorizer
            tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2)
            )

            # Convert text into numerical vectors
            tfidf_matrix = tfidf.fit_transform(products["combined_features"])

            # Save vectorizer
            with open(os.path.join(self.artifacts_path, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(tfidf, f)

            # Save matrix
            with open(os.path.join(self.artifacts_path, "tfidf_matrix.pkl"), "wb") as f:
                pickle.dump(tfidf_matrix, f)

            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Save similarity matrix
            with open(os.path.join(self.artifacts_path, "cosine_similarity.pkl"), "wb") as f:
                pickle.dump(similarity_matrix, f)

            logger.info("Feature engineering completed successfully")

            return products, tfidf_matrix

        except Exception as e:
            raise CustomException(e, sys)