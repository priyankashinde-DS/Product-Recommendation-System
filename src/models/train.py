import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import logger
from src.utils.custom_exception import CustomException


class Trainer:
    def __init__(self, artifacts_path="artifacts"):
        self.artifacts_path = artifacts_path

    def load_tfidf_matrix(self):
        try:
            with open(os.path.join(self.artifacts_path, "tfidf_matrix.pkl"), "rb") as f:
                tfidf_matrix = pickle.load(f)

            logger.info("TF-IDF matrix loaded")
            return tfidf_matrix

        except Exception as e:
            raise CustomException(e, sys)

    def train(self):
        try:
            logger.info("Training similarity model")

            tfidf_matrix = self.load_tfidf_matrix()
            similarity_matrix = cosine_similarity(tfidf_matrix)

            os.makedirs(self.artifacts_path, exist_ok=True)

            with open(os.path.join(self.artifacts_path, "cosine_similarity.pkl"), "wb") as f:
                pickle.dump(similarity_matrix, f)

            logger.info("Similarity matrix saved successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()