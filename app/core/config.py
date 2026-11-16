
"""Centralised application configuration."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.core.keyvault import get_secret


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Load and validate configuration from environment variables."""

    app_name: str = "IT Support RAG Assistant"
    app_env: str = "development"
    rag_api_key: str | None = None

    data_dir: Path = PROJECT_ROOT / "data"

    # Select which vector store to use.
    vector_store_provider: Literal["chroma", "faiss"] = "chroma"

    # Directories for storing vector embeddings.
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    faiss_dir: Path = PROJECT_ROOT / "faiss_db"
    # Select the retrieval pipeline.
    # Standard preserves the existing RAG behaviour.
    retrieval_mode: Literal["standard", "enhanced"] = "standard"

    # Separate storage for parent-child FAISS embeddings.
    parent_child_faiss_dir: Path = PROJECT_ROOT / "faiss_parent_child_db"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_size: int = Field(default=500, ge=100)
    chunk_overlap: int = Field(default=75, ge=0)
    retrieval_top_k: int = Field(default=3, ge=1, le=20)

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    retrieval_max_distance: float = Field(
    default=0.60,
    ge=0.0,
    le=2.0,
    )

    def model_post_init(self, __context: object) -> None:
        """Validate settings that depend on one another."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )



@lru_cache
def get_settings() -> Settings:
    """Load configuration, including secrets from Azure Key Vault."""

    settings = Settings()

    use_keyvault = os.getenv(
        "USE_AZURE_KEY_VAULT", "false"
    ).lower() == "true"

    if use_keyvault:
        settings.azure_openai_api_key = get_secret(
            "AZURE-OPENAI-KEY"
        )

        settings.rag_api_key = get_secret(
            "RAG-API-KEY"
        )

    return settings
