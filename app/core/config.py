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
    llm_provider: str = "mock"  # mock | anthropic
    transcription_provider: str = "mock"  # mock | openai
    audio_upload_dir: str = "/data/audio"
    log_level: str = "INFO"

    # API keys (only needed when the respective provider is active)
    openai_api_key: str = ""      # used by transcription_provider=openai (Whisper)
    anthropic_api_key: str = ""   # used by llm_provider=anthropic (Claude)

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
