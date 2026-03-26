import os
from dotenv import load_dotenv

load_dotenv()

PGCONN_STRING = os.getenv("PGCONN_STRING")
PGCONN_MATRIKKEL = os.getenv("PGCONN_MATRIKKEL")


PGCONN_FLOMAKTSOMRADER = os.getenv("PGCONN_FLOMAKTSOMRADER_GIS")  
PGCONN_FLOMSONER       = os.getenv("PGCONN_FLOMSONER_GIS")        
PGCONN_BRANNSTASJONER  = os.getenv("PGCONN_BRANNSTASJONER_GIS")   

PGCONN_KVIKKLEIRE = os.getenv("PGCONN_KVIKKLEIRE_GIS")
PGCONN_SNOSKREDAKTSOMHETSKART = os.getenv("PGCONN_SNOSKREDAKTSOMHETSKART_GIS")
PGCONN_MATRIKKELENEIENDOMSKARTTEIG = os.getenv("PGCONN_MATRIKKELENEIENDOMSKARTTEIG_GIS")





PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED = os.getenv("PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED_GIS")

PGCONN_KOMMUNER = os.getenv("PGCONN_KOMMUNER_GIS")






AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_FOUNDRY_KEY = os.getenv("AZURE_FOUNDRY_KEY")
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")








AZURE_MODEL_DEPLOYMENT = {
    # Azure OpenAI
    "gpt-35-turbo":          "gpt-35-turbo",
    "gpt-35-turbo-16k":      "gpt-35-turbo-16k",
    "gpt-35-turbo-instruct": "gpt-35-turbo-instruct",
    "gpt-4":                 "gpt-4",
    "gpt-4o":                "gpt-4o",
    "gpt-4o-mini":           "gpt-4o-mini",
    "gpt-4.1-nano":          "gpt-4.1-nano",
    # Azure Foundry — GPT
    "gpt-4-32k":             "gpt-4-32k",
    "gpt-4.1":               "gpt-4.1",
    "gpt-4.1-mini":          "gpt-4.1-mini",
    "gpt-4.5-preview":       "gpt-4.5-preview",
    "gpt-5-mini":            "gpt-5-mini",
    "o1":                    "o1",
    "o3":                    "o3",
    "o3-mini":               "o3-mini",
    "o3-pro":                "o3-pro",
    "o4-mini":               "o4-mini",
    # Azure Foundry — andre
    "Mistral-Large-3":       "Mistral-Large-3",
    "Mistral-small-2503":    "mistral-small-2503",
    "Llama-4-Maverick":      "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "grok-3-mini":           "grok-3-mini",
    "grok-3":                "grok-3",
    "DeepSeek-V3-0324":      "DeepSeek-V3-0324",
    "Cohere-command-r":      "Cohere-command-r-08-2024",
    "Phi-4":                 "Phi-4",
}

AZURE_OPENAI_MODEL_IDS = {
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-35-turbo-instruct",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-nano",
}






AZURE_FOUNDRY_MODEL_IDS = set(AZURE_MODEL_DEPLOYMENT.keys()) - AZURE_OPENAI_MODEL_IDS
ALL_MODELS = list(AZURE_MODEL_DEPLOYMENT.keys())

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)



OLLAMA_MODEL_IDS = {
    "functiongemma",
}





#blob storage



AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = "llmevaluator"