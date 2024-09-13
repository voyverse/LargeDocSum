import pandas as pd
import json
import os
from src.summarization_pipeline import pipeline
from src.data.hf_cleaned_booksum import DataLoader
from openai import OpenAI
from src.logger.logger import get_logger

logger = get_logger(__file__)


# Initialize data loader
data_loader = DataLoader()
number_of_books = data_loader.get_number_of_books()

# Initialize OpenAI client
client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")

# Define common parameters
chunking_method = 'recursive'  # or 'semantic', depending on your requirement
chunking_params = {
    'max_length': 500,  # example value
    'overlap': 20      # example value
}
embed_model_name = "nomic-embed-text"
summ_model_name = "gpt-4o-mini"
top_k = 2  # example value

# Directory to save JSON files
output_dir = os.path.join('data' , 'summarization_results')
os.makedirs(output_dir, exist_ok=True)

# List to collect results for DataFrame
results = []

# Iterate over all books
for index_to_fetch in range(number_of_books):
    # Fetch book and summary
    book_summary_couple = data_loader.get_book_summary_couple(index_to_fetch)
    doc = book_summary_couple["book"]
    reference_summary = book_summary_couple["summary"]

    # Define prompts using the document content
    system_prompt_aggregate_summaries = (
        "You are an AI assistant designed to summarize text. To give you some context, the document starts this way: "
        + doc[:300]
    )
    system_prompt_docsummary = (
        "You are an advanced writing assistant. I will provide you with several summaries of important sections of a long document. "
        "Your task is to aggregate these summaries into one coherent paragraph. This is how the document from which the summaries were extracted starts: "
        + doc[:300]
    )
    llm_instructions_doc_summary = 'aggregate these summaries given to you.'

    # Run the pipeline
    output = pipeline(
        chunking_method=chunking_method,
        chunking_params=chunking_params,
        embed_model_name=embed_model_name,
        summ_model_name=summ_model_name,
        doc=doc,
        top_k=top_k,
        system_prompt_aggregate_summaries=system_prompt_aggregate_summaries,
        system_prompt_docsummary=system_prompt_docsummary,
        llm_instructions_doc_summary=llm_instructions_doc_summary,
        reference_summary=reference_summary,
        client=client,
        log=True  # Enable logging
    )

    # Create the result dictionary
    result = {
        'Book Index': index_to_fetch,
        'Reference Summary' : reference_summary , 
        'Summary': output['summary'],
        'ROUGE Scores': output['rouge'],
        'BERTScore': output['bertscore'],
        'Coherence Scores': output['coherence']
    }

    # Save the result to a JSON file
    json_filename = os.path.join(output_dir, f'book_summary_{index_to_fetch}.json')
    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    # Print the result
    logger.info(result)

    # Append the result to the list for the DataFrame
    results.append(result)

# Convert the list of results into a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('summarization_results.csv', index=False)

print("Results have been saved to 'summarization_results.csv' and individual JSON files in the 'summarization_results' directory.")
