from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "OpenAI"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = "sk-xxxx"
