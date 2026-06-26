from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str
    REDIS_PASSWORD: str
    REDIS_URL: str
    MLFLOW_TRACKING_URI: str
    ENVIRONMENT: str
    DEBUG: bool
    OTEL_EXPORTER_OTLP_ENDPOINT: str

    class Config:
        env_file = "../.env"

settings = Settings()
