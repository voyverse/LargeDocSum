from src.summarization_pipeline import pipeline
from src.data.hf_cleaned_booksum import DataLoader
from openai import OpenAI



data_loader = DataLoader()
client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")




