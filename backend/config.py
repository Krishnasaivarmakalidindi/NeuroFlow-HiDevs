import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "neuroflow")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "neuroflow")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Redis
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    # Application
    ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    DEBUG: bool = os.getenv(
        "DEBUG",
        "false"
    ).lower() == "true"

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4317"
    )

    # Groq/OpenAI-compatible provider
    OPENAI_API_KEY: str = os.getenv(
        "OPENAI_API_KEY",
        ""
    )

    OPENAI_BASE_URL: str = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.groq.com/openai/v1"
    )

    DEFAULT_CHAT_MODEL: str = os.getenv(
        "DEFAULT_CHAT_MODEL",
        "llama-3.3-70b-versatile"
    )

    model_config = {
        "env_file": "../.env"
    }


settings = Settings()
