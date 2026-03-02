import pandas as pd
import sys
from src.utils.logger import logger
from src.utils.custom_exception import CustomException


def load_products(path: str):
    try:
        logger.info("Loading products dataset")
        return pd.read_csv(path)
    except Exception as e:
        raise CustomException(e, sys)


def load_interactions(path: str):
    try:
        logger.info("Loading interactions dataset")
        return pd.read_csv(path)
    except Exception as e:
        raise CustomException(e, sys)


#if __name__ == "__main__":
    data = load_products("data/raw/products_10k.csv")
    print(data.head())