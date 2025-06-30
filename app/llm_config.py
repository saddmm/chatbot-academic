from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

DEFAULT_MODEL_NAME = "mistral:instruct"
DEFAULT_EMBEDDING_MODEL_NAME = "nomic-embed-text"
DEFAULT_TEMPERATURE = 0.1

def get_llm(model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE):
    try:
        llm = OllamaLLM(model= model_name, temperature=temperature)
        return llm
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM with model {model_name}: {str(e)}") from e
    
def get_groq_llm(model_name: str, temperature: float):
    try:
        llm = ChatGroq(model=model_name, temperature=temperature, api_key=groq_api_key)
        return llm
    except Exception as e:
        raise ValueError(f"Failed to initialize Groq LLM with model {model_name}: {str(e)}") from e

def get_embedding(model_name : str = DEFAULT_EMBEDDING_MODEL_NAME):
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        return embeddings
    except Exception as e:
        raise ValueError(f"Failed to initialize embeddings with model {model_name}: {str(e)}") from e

if __name__ == "__main__":
    llm = get_llm()
    print(f"LLM initialized with model: {llm.model}")
    
    embeddings = get_embedding()
    print(f"Embeddings initialized with model: {embeddings.model}")    
