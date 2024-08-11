from typing import Dict, List, Any, Union
import logging
import pandas as pd
import sys
import os
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.indexing import *
from src.cluster import *
from src.markov_sum import *
from src.evaluation.metrics import *
from openai import OpenAI








CHUNKING_METHODS = ["recursive", "semantic"]
def pipeline(
    chunking_method: str,
    chunking_params: Dict[str, Any],
    embed_model_name: str,
    summ_model_name: str,
    doc: str,
    num_clusters: int,
    top_k: int,  # top k closest to cluster centroid
    system_prompt_aggregate_summaries: str,
    system_prompt_docsummary: str,
    llm_instructions_doc_summary: str,
    reference_summary: str,
    log: bool = False
) -> Dict[str, Any]:
    """
    Summarization pipeline that includes chunking, embedding, clustering, 
    generating aggregate summaries, and producing a final document 
    summary. Also calculates various evaluation metrics.

    Parameters:
    - chunking_method (str): Method used for chunking the document ('recursive' or 'semantic').
    - chunking_params (Dict[str, Any]): Parameters required for the chosen chunking method.
    - embed_model_name (str): Name of the embedding model.
    - summ_model_name (str): Name of the summarization model.
    - doc (str): Document to be summarized.
    - num_clusters (int): Number of clusters to form.
    - top_k (int): Top k closest chunks to the cluster centroid.
    - system_prompt_aggregate_summaries (str): System prompt for aggregate summaries. (needs to contain how the doc starts)
    - system_prompt_docsummary (str): System prompt for the final document summary. (needs to contain how the doc starts)
    - llm_instructions_doc_summary (str): Instructions for the language model for the final summary.  
    - reference_summary (str): Reference summary for evaluation.
    - log (bool): Enable logging if True.

    Returns:
    - results (Dict[str, Any]): Dictionary containing the final summary and various evaluation metrics.
    """
    client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")

    if log:
        logging.info("Pipeline started")

    # Validate inputs
    assert chunking_method in CHUNKING_METHODS, "Invalid chunking method provided."
    if chunking_method == "recursive":
        assert "max_length" in chunking_params.keys() and "overlap" in chunking_params.keys(), \
            "'max_length' and 'overlap' must be provided for recursive chunking."
    elif chunking_method == "semantic":
        assert "threshold" in chunking_params.keys() and "model" in chunking_params.keys(), \
            "'threshold' and 'model' must be provided for semantic chunking."
    assert num_clusters > 0, "Number of clusters must be greater than 0."
    assert num_clusters > top_k and top_k > 0, "Number of clusters must be greater than top_k, and top_k must be positive."
    assert doc, "Document must not be None."

    if log:
        logging.info(f"Chunking method: {chunking_method}")
        logging.info(f"Embedding model: {embed_model_name}")
        logging.info(f"Summarization model: {summ_model_name}")
        logging.info("Parameters validated successfully")

    # Chunk the document based on the chosen method
    if chunking_method == 'recursive':
        words = doc.split()
        chunks, avg_chunk_length = iterative_chunk(words, **chunking_params)
        if log:
            logging.info(f"Document chunked using recursive method: {len(chunks)} chunks created")
    elif chunking_method == 'semantic':
        sentences = doc.split(".")
        chunks = semantic_chunking(sentences, **chunking_params)
        if log:
            logging.info(f"Document chunked using semantic method: {len(chunks)} chunks created")

    # Create vector database
    vector_database = create_vbd(chunks, emb_model=embed_model_name)
    if log:
        logging.info("Vector database created")

    # Perform clustering
    clustered_chunks, cluster_centroids = cluster_chunks_kmeans(vdb=vector_database, num_clusters=num_clusters)
    if log:
        logging.info(f"Clustering completed: {num_clusters} clusters formed")

    # Identify closest chunks to cluster centroids
    closest_chunks = find_closest_data_points_to_centroid(
        clustered_chunks=clustered_chunks,
        cluster_centroids=cluster_centroids,
        top_k=top_k
    )
    if log:
        logging.info(f"Identified top {top_k} closest chunks to cluster centroids")

    # Aggregate summaries for each cluster
    cluster_summaries = aggregate_summaries_for_each_cluster(
        closest_chunks,
        sys_prompt_content=system_prompt_aggregate_summaries, #has parameter called start -> how the document starts 
        model=summ_model_name,
        client=client
    )
    if log:
        logging.info("Generated aggregate summaries for each cluster")

    # Create transition matrix and directed graph
    transition_matrix = create_transition_matrix(clustered_chunks, n_clusters=num_clusters)
    graph = create_directed_graph(transition_matrix=transition_matrix, summary_by_cluster=cluster_summaries)
    if log:
        logging.info("Transition matrix and directed graph created")

    # Determine the most probable path
    first_node = cluster_summaries[clustered_chunks[0]["cluster"]]
    last_node = cluster_summaries[clustered_chunks[-1]["cluster"]]
    path = most_probable_path(graph, first_node, last_node)
    if log:
        logging.info("Most probable path identified -> path : {path}")

    # Prepare messages for the final summary call
    messages = [{'role': 'system', 'content': system_prompt_docsummary}]
    for summary in path:
        messages.append({'role': 'user', 'content': summary})
    messages.append({"role": "user", 'content': llm_instructions_doc_summary})
    if log : 
        print(f"messages for generating the final summary are  : \n{messages}")
    # Generate the final summary
    overall_summary = get_completion_response(messages, client=client, model=summ_model_name)
    if log:
        logging.info("Final document summary generated")

    # Calculate evaluation metrics
    candidates = [overall_summary]
    references = [[reference_summary]]

    if log:
        logging.info("Calculating ROUGE scores")
    rouge_results = calculate_rouge_scores(candidates, references)

    if log:
        logging.info("Calculating BERTScore")
    bert_results = calculate_bertscore(candidates, references)

    if log:
        logging.info("Calculating coherence scores")
    coherence_results = calculate_coherence(overall_summary)

    results = {
        'summary': overall_summary,
        'rouge': rouge_results,
        'bertscore': bert_results,
        'coherence': coherence_results
    }

    if log:
        logging.info("Pipeline completed")

    return results




if __name__ == "__main__":
    # Read the document
    file_path = r"data\ted_talk_1.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        doc = file.read()

    # Parameters for the pipeline
    chunking_method = 'recursive'  # or 'semantic', depending on your requirement
    chunking_params = {
        'max_length': 200,  # example value
        'overlap': 20       # example value
    }
    embed_model_name = "nomic-embed-text"
    summ_model_name = 'gemma2'
    num_clusters = 3  # example value
    top_k = 2  # example value
    system_prompt_aggregate_summaries = "You are an AI assistant designed to summarize text. To give you some context , the document starts this way  : " + doc[:300]
    system_prompt_docsummary = 'You are an advanced summarization assistant.I will provide you with several summaries of important sections of a long document. Your task is to generate a concise and comprehensive summary of the entire document based on these individual summaries. this is how the documane starts : ' + doc[:300]
    llm_instructions_doc_summary = 'Provide a coherent abstractive summary based on the provided parts.'
    reference_summary = 'ted talk about malaria'  

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
        log=True  # Enable logging
    )

    # Print the output
    print("Summary:", output['summary'])
    print("ROUGE Scores:", output['rouge'])
    print("BERTScore:", output['bertscore'])
    print("Coherence Scores:", output['coherence'])