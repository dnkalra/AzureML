import os
import logging
import json
import tempfile
from typing import List

import numpy as np
import pandas as pd
import pyodbc
import faiss
import requests

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

# =========================================================
# GLOBAL LOGGER CONFIG
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("aml-faiss-pipeline")

# =========================================================
# CONFIG CLASS
# =========================================================

class Config:

    # Azure SQL
    SQL_SERVER = os.getenv("SQL_SERVER")
    SQL_DB = os.getenv("SQL_DB")
    SQL_TABLE = "account"

    # Cohere Serverless Endpoint
    EMBEDDING_ENDPOINT = os.getenv("EMBED4_ENDPOINT_URL")
    EMBEDDING_MODEL = "embed-v4"

    # ADLS
    ADLS_ACCOUNT = os.getenv("ADLS_ACCOUNT_NAME")
    ADLS_FILESYSTEM = os.getenv("ADLS_FILESYSTEM")
    ADLS_DIR = "faiss-index"

    VECTOR_DIMENSION = 1024   # change based on embed-v4 output

# =========================================================
# SQL READER
# =========================================================

class AzureSQLReader:

    def __init__(self, config: Config):
        self.config = config
        self.credential = DefaultAzureCredential()

    def _get_connection(self):
        try:
            token = self.credential.get_token(
                "https://database.windows.net/.default"
            ).token

            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                f"SERVER={self.config.SQL_SERVER};"
                f"DATABASE={self.config.SQL_DB};"
                "Encrypt=yes;"
                "TrustServerCertificate=no;"
                "Authentication=ActiveDirectoryAccessToken"
            )

            token_bytes = bytes(token, "utf-16-le")

            return pyodbc.connect(conn_str, attrs_before={1256: token_bytes})

        except Exception as e:
            logger.exception("Failed to create SQL connection")
            raise

    def read_accounts(self) -> pd.DataFrame:
        try:
            conn = self._get_connection()
            query = f"SELECT * FROM {self.config.SQL_TABLE}"

            df = pd.read_sql(query, conn)
            logger.info(f"Loaded {len(df)} rows from Azure SQL")
            return df

        except Exception:
            logger.exception("Error reading Azure SQL table")
            raise


# =========================================================
# EMBEDDING CLIENT (COHERE SERVERLESS)
# =========================================================

class CohereEmbeddingClient:

    def __init__(self, config: Config):
        self.config = config
        self.credential = DefaultAzureCredential()

    def _get_headers(self):
        try:
            token = self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token

            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        except Exception:
            logger.exception("Failed to obtain token for embedding endpoint")
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        try:
            payload = {
                "input": texts,
                "model": self.config.EMBEDDING_MODEL
            }

            response = requests.post(
                self.config.EMBEDDING_ENDPOINT,
                headers=self._get_headers(),
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            embeddings = [item["embedding"] for item in data["data"]]

            return np.array(embeddings).astype("float32")

        except Exception:
            logger.exception("Embedding request failed")
            raise


# =========================================================
# ADLS MANAGER
# =========================================================

class ADLSManager:

    def __init__(self, config: Config):
        self.config = config
        self.credential = DefaultAzureCredential()

        try:
            account_url = f"https://{config.ADLS_ACCOUNT}.dfs.core.windows.net"

            self.service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=self.credential
            )

            self.fs_client = self.service_client.get_file_system_client(
                config.ADLS_FILESYSTEM
            )

        except Exception:
            logger.exception("Failed to init ADLS client")
            raise

    def upload_file(self, local_path: str, file_name: str):

        try:
            file_client = self.fs_client.get_file_client(
                f"{self.config.ADLS_DIR}/{file_name}"
            )

            with open(local_path, "rb") as f:
                file_client.upload_data(f, overwrite=True)

            logger.info(f"Uploaded {file_name} to ADLS")

        except Exception:
            logger.exception("Failed uploading to ADLS")
            raise

    def download_file(self, local_path: str, file_name: str):

        try:
            file_client = self.fs_client.get_file_client(
                f"{self.config.ADLS_DIR}/{file_name}"
            )

            with open(local_path, "wb") as f:
                data = file_client.download_file().readall()
                f.write(data)

            logger.info(f"Downloaded {file_name} from ADLS")

        except Exception:
            logger.exception("Failed downloading from ADLS")
            raise


# =========================================================
# FAISS MANAGER
# =========================================================

class FaissManager:

    def __init__(self, dimension: int):
        self.dimension = dimension

    def create_index(self, vectors: np.ndarray):
        try:
            index = faiss.IndexFlatL2(self.dimension)
            index.add(vectors)
            logger.info(f"FAISS index built with {index.ntotal} vectors")
            return index
        except Exception:
            logger.exception("Failed creating FAISS index")
            raise

    def save_index(self, index, path):
        try:
            faiss.write_index(index, path)
        except Exception:
            logger.exception("Failed saving FAISS index")
            raise

    def load_index(self, path):
        try:
            return faiss.read_index(path)
        except Exception:
            logger.exception("Failed loading FAISS index")
            raise


# =========================================================
# MAIN PIPELINE ENTRYPOINT
# =========================================================

def main():

    try:
        config = Config()

        sql_reader = AzureSQLReader(config)
        embed_client = CohereEmbeddingClient(config)
        adls = ADLSManager(config)
        faiss_mgr = FaissManager(config.VECTOR_DIMENSION)

        # STEP 1 — Read Data
        df = sql_reader.read_accounts()

        if df.empty:
            logger.warning("No data found in account table")
            return

        texts = df.astype(str).agg(" ".join, axis=1).tolist()

        # STEP 2 — Generate Embeddings
        embeddings = embed_client.embed(texts)

        # STEP 3 — Build FAISS
        index = faiss_mgr.create_index(embeddings)

        with tempfile.TemporaryDirectory() as tmp:

            index_path = os.path.join(tmp, "index.faiss")

            faiss_mgr.save_index(index, index_path)

            # STEP 4 — Upload to ADLS
            adls.upload_file(index_path, "index.faiss")

        logger.info("Pipeline execution completed successfully")

    except Exception:
        logger.exception("Pipeline FAILED")
        raise


if __name__ == "__main__":
    main()
