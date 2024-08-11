from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import networkx as nx

def create_transition_matrix(
    processed_data: List[Dict[str, Any]], 
    n_clusters: int
) -> np.ndarray:
    """
    Creates a transition matrix based on the transitions between clusters.

    Args:
        processed_data (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
            - "cluster" (int): The cluster label assigned by k-means.
            - "chunk" (str): The chunk itself.
            - "embedding" (np.ndarray): The embedding of the chunk.
            - "pos" (int): The chronological position of the chunk.
        n_clusters (int): The number of clusters.

    Returns:
        np.ndarray: A transition matrix where each entry (i, j) represents the transition probability from cluster i to cluster j.
    """
    # Initialize the transition matrix with zeros
    transition_matrix = np.zeros(shape=(n_clusters, n_clusters))

    # Iterate over the processed_data to fill the transition matrix
    for i in tqdm(range(len(processed_data) - 1), desc="Calculating transition matrix..."):
        from_cluster = processed_data[i]["cluster"]
        to_cluster = processed_data[i + 1]["cluster"]
        transition_matrix[from_cluster][to_cluster] += 1

    # Calculate the sum of transitions from each cluster (row-wise sum)
    row_sums = np.sum(transition_matrix, axis=1, keepdims=True)

    # Normalize the transition matrix by row sums to get the probabilities
    transition_matrix = transition_matrix / row_sums

    return transition_matrix



def create_directed_graph(
    transition_matrix: np.ndarray, 
    summary_by_cluster: Dict[int, str]
) -> nx.DiGraph:
    """
    Creates a directed graph from a transition matrix and summaries by cluster.

    Args:
        transition_matrix (np.ndarray): A transition matrix where each entry (i, j) represents the transition probability from cluster i to cluster j.
        summary_by_cluster (Dict[int, str]): A dictionary where keys are cluster indices, and values are summaries corresponding to those clusters.

    Returns:
        nx.DiGraph: A directed graph with edges weighted by transition probabilities and nodes labeled by summaries.
    """
    # Initialize a directed graph
    G = nx.DiGraph()

    # Create a lookup to convert cluster index to its corresponding summary
    cluster_by_intent = {intent: int(cluster) for cluster, intent in summary_by_cluster.items()}

    # Add edges with weights
    for i, from_intent in summary_by_cluster.items():
        for j, to_intent in summary_by_cluster.items():
            # Add edge from from_intent to to_intent with weight from the transition matrix
            G.add_edge(from_intent, to_intent, weight=transition_matrix[int(i), int(j)])

    return G 



import math
import networkx as nx

def add_transformed_weights(graph, weight_attribute='probability', transformed_attribute='weight'):
    """
    Add transformed weights to the graph edges by taking negative logarithm of the given probability weights.

    Parameters:
    graph (networkx.Graph): The input graph with probability weights.
    weight_attribute (str): The edge attribute name for the probability weights.
    transformed_attribute (str): The edge attribute name for the transformed weights.
    """
    for u, v, data in graph.edges(data=True):
        prob = data.get(weight_attribute, 1)
        if prob > 0:
            data[transformed_attribute] = -math.log(prob)
        else:
            data[transformed_attribute] = float('inf')  # Handle zero probability edges

def most_probable_path(graph, start, end):
    """
    Finds the most probable path from the start node to the end node in a graph where edges represent probabilities.

    Parameters:
    graph (networkx.Graph): The input graph with nodes and edges representing probabilities of transitions.
    start (hashable): The starting node.
    end (hashable): The ending node.

    Returns:
    list: The most probable path from start to end node passing through all nodes, as a list of nodes. 
          The list will be empty if such a path does not exist.
    """
    add_transformed_weights(graph)

    try:
        # Compute the shortest path using the negative logarithm of the probabilities as weights
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        # If no path is found, return an empty list
        return []

# Example Usage
if __name__ == "__main__":
    graph = nx.DiGraph()
    graph.add_edge('A', 'B', probability=0.9)
    graph.add_edge('B', 'C', probability=0.8)
    graph.add_edge('A', 'C', probability=0.6)
    graph.add_edge('C', 'A', probability=0.7)
    graph.add_edge('C', 'D', probability=0.5)
    start_node = 'A'
    end_node = 'D'
    path = most_probable_path(graph, start_node, end_node)
    print(f"Most probable path: {path}")