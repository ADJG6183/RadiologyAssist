from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_host: str = "localhost"
    database_port: int = 1433
    database_name: str = "RadiologyAI"
    database_user: str = "sa"
    database_password: str = ""

    # App behaviour
    test_mode: int = 0          # 1 = use SQLite in-memory (no MS SQL needed)
    llm_provider: str = "mock"  # mock | openai | anthropic
    audio_upload_dir: str = "/data/audio"
    log_level: str = "INFO"

    # Optional LLM keys (only needed if llm_provider != mock)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def db_url(self) -> str:
        """SQLAlchemy connection URL. Switches to SQLite when TEST_MODE=1."""
        if self.test_mode:
            return "sqlite:///:memory:"
        return (
            f"mssql+pyodbc://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
            f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
        )


settings = Settings()
