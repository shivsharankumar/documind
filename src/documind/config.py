from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Your two providers
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # Which one to use by default
    default_provider: str = "groq"

    # Default models for each
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"

    cohere_api_key: str | None = None


settings = Settings()
