from typing import List, Tuple, Dict, Any , Union
import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
import json
import logging

def cluster_chunks_kmeans(
    vdb: List[Dict[str, Any]], 
    num_clusters: int
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Clusters the embeddings using k-means++ and returns the clustered results with centroids.

    Args:
        vdb (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
            - "pos" (int): The chronological position of the chunk.
            - "chunk" (str): The chunk itself.
            - "embedding" (np.ndarray): The embedding of the chunk.
        num_clusters (int): The number of clusters to form.

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray]: 
            - List[Dict[str, Any]]: A list of dictionaries where each dictionary contains:
                - "cluster" (int): The cluster label assigned by k-means.
                - "chunk" (str): The chunk itself.
                - "embedding" (np.ndarray): The embedding of the chunk.
                - "pos" (int): The chronological position of the chunk.
            - np.ndarray: An array of cluster centroids.
    """
    # Extract embeddings from the provided dictionary list
    embeddings = np.array([d["embedding"] for d in vdb])

    # Perform k-means++ clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_

    # Create a list of dictionaries with cluster information
    clustered_chunks = [
        {
            "cluster": cluster_labels[i],
            "chunk": data_point["chunk"],
            "embedding": data_point["embedding"],
            "pos": data_point["pos"]
        }
        for i, data_point in enumerate(vdb)
    ]

    return clustered_chunks, cluster_centroids




def find_closest_data_points_to_centroid(
    clustered_chunks: List[Dict[str, Any]], 
    cluster_centroids: np.ndarray, 
    top_k: int
) -> Dict[int, List[str]]:
    """
    Finds the closest data points to each cluster centroid.

    Args:
        clustered_chunks (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
            - "cluster" (int): The cluster label assigned by k-means.
            - "chunk" (str): The chunk itself.
            - "embedding" (np.ndarray): The embedding of the chunk.
            - "pos" (int): The chronological position of the chunk.
        cluster_centroids (np.ndarray): An array of cluster centroids from k-means.
        top_k (int): The number of closest data points to return for each cluster.

    Returns:
        Dict[int, List[str]]: A dictionary where the keys are cluster indices, and the values are lists 
                               of the closest `top_k` chunks to the respective cluster centroids.
    """
    closest_sentences = {}

    for cluster_index, cluster_centroid in enumerate(cluster_centroids):
        distances = []

        for data in clustered_chunks:
            if data["cluster"] == cluster_index:
                distance = np.linalg.norm(cluster_centroid - data["embedding"])
                distances.append({
                    "distance": distance,
                    "chunk": data["chunk"]
                })

        # Sort distances in ascending order
        distances = sorted(distances, key=lambda x: x["distance"])
        # Store the top_k closest chunks for the current cluster
        closest_sentences[cluster_index] = [e["chunk"] for e in distances[:top_k]]

    return closest_sentences






def get_completion_response(
    messages: List[Dict[str, str]], 
    client: Any, model: str
) -> Union[Dict[str, Any], str]:
    """
    Generates a response from an AI model based on the input messages.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries containing:
            - "role" (str): The role of the message sender, e.g., "system", "user".
            - "content" (str): The content of the message.
        client (Any): The AI client used to generate the response.
        model (str): The AI model to use for generating the response.

    Returns:
        Union[Dict[str, Any], str]: The parsed JSON response if successful, otherwise the raw response content.
    """
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error occurred in json.loads: {e}, with output being {response.choices[0].message.content}")
        return response.choices[0].message.content

from typing import Dict, List

def aggregate_summaries_for_each_cluster(
    closest_chunks: Dict[int, List[str]], 
    sys_prompt_content: str , 
    client : OpenAI, 
    model : str 
) -> Dict[int, str]:
    """
    Aggregates summaries for each cluster based on the closest chunks and system prompt content.

    Args:
        closest_chunks (Dict[int, List[str]]): A dictionary where each key is a cluster index and the value is a list of sentences (chunks) closest to the cluster centroid.
        sys_prompt_content (str): The content of the system prompt to be used for summarization.

    Returns:
        Dict[int, str]: A dictionary where each key is a cluster index and the value is the summarized text for that cluster.
    """
    aggregated_messages = [{"role": "system", "content": sys_prompt_content}]
    cluster_summaries = {}

    for cluster_index, sentences in closest_chunks.items():
        chunks = " \n - ".join(sentences)
        format = "{ 'summary': 'your output goes directly here' }"
        prompt = (
            f"You are an AI assistant designed to summarize text. You will be provided with multiple sentences extracted "
            f"from a large document that may be incomplete or incoherent at first sight. You need to focus and try to extract "
            f"the hidden main ideas. Your task is to summarize these sentences into a coherent single paragraph that captures "
            f"the main ideas. The sentences are:\n - {chunks}\nYou answer in this format {format}, you only output the needed format no more no less please."
        )
        messages = aggregated_messages + [{"role": "user", "content": prompt}]
        summary = get_completion_response(messages, client, model)
        try:
            summary_text = summary["summary"]
        except KeyError as e:
            logging.error(f"Key error occurred: {summary}")
            summary_text = str(summary)

        cluster_summaries[cluster_index] = summary_text
        aggregated_messages.append({"role": "assistant", "content": summary_text})

    return cluster_summaries