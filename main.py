from src.summarization_pipeline import pipeline
from src.data.hf_cleaned_booksum import DataLoader
from openai import OpenAI


data_loader = DataLoader()


index_to_fetch = 0 
book_summary_couple = data_loader.get_book_summary_couple(index_to_fetch)

doc = book_summary_couple["book"]
reference_summary  = book_summary_couple["summary"]

chunking_method = 'recursive'  # or 'semantic', depending on your requirement
chunking_params = {
    'max_length': 50,  # example value
    'overlap': 1       # example value
}
embed_model_name = "nomic-embed-text"
summ_model_name = "gpt-4o-mini"
num_clusters = 20  # example value
top_k = 2  # example value
system_prompt_aggregate_summaries = "You are an AI assistant designed to summarize text. To give you some context , the document starts this way  : " + doc[:300]
system_prompt_docsummary = 'You are an advanced writing assistant.I will provide you with several summaries of important sections of a long document. Your task is to generate a concise and comprehensive aggregation of these summaries. this is how the document from which the sumaries where extracted starts :  ' + doc[:300]
llm_instructions_doc_summary = 'aggregate these summaries given to you.'

client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")
# Run the pipeline
output = pipeline(
    chunking_method=chunking_method,
    chunking_params=chunking_params,
    embed_model_name=embed_model_name,
    summ_model_name=summ_model_name,
    doc=doc,
    num_clusters=num_clusters,
    top_k=top_k,
    system_prompt_aggregate_summaries=system_prompt_aggregate_summaries,
    system_prompt_docsummary=system_prompt_docsummary,
    llm_instructions_doc_summary=llm_instructions_doc_summary,
    reference_summary=reference_summary,
    client = client ,
    log=True # Enable logging
)

# Print the output
print("Summary:", output['summary'])
print("ROUGE Scores:", output['rouge'])
print("BERTScore:", output['bertscore'])
print("Coherence Scores:", output['coherence'])



