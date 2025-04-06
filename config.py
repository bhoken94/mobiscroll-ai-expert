import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from supabase import create_client
import logfire

# Carica variabili d'ambiente
load_dotenv()

class Config:
    """Classe di configurazione per le API e le variabili d'ambiente."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_MODEL_EMBEDDINGS = os.getenv("LLM_MODEL_EMBEDDINGS")

    @staticmethod
    def setup_openai():
        """Configura OpenAI."""
        return AsyncOpenAI(base_url = "http://localhost:11434/v1",api_key = "ollama")

    @staticmethod
    def setup_supabase():
        """Configura Supabase."""
        return create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )

    @staticmethod
    def setup_logfire():
        """Configura Logfire."""
        logfire.configure(send_to_logfire="never")
