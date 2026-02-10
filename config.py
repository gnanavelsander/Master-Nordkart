import sys
from pathlib import Path


import os
from dotenv import load_dotenv

load_dotenv()

PGCONN_STRING = os.getenv("PGCONN_STRING")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_MODEL_DEPLOYMENT = {
    
    "gpt-4o": "gpt-4o",
    "gpt-4": "gpt-4",
    "gpt-35-turbo": "gpt-35-turbo",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "GPT-4o-mini": "GPT-4o-mini"
}


AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)

