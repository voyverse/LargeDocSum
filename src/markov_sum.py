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




def most_probable_path(graph, start, end):
    n = len(graph.nodes)
    
    node_to_index = {node: i for i, node in enumerate(graph.nodes)}
    index_to_node = {i: node for node, i in node_to_index.items()}
    
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    dp[1 << node_to_index[start]][node_to_index[start]] = 0
    
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if not mask & (1 << v) and graph.has_edge(index_to_node[u], index_to_node[v]):
                        new_mask = mask | (1 << v)
                        weight = graph[index_to_node[u]][index_to_node[v]].get('weight', 1)
                        if dp[mask][u] + weight < dp[new_mask][v]:
                            dp[new_mask][v] = dp[mask][u] + weight
                            parent[new_mask][v] = u
    
    full_mask = (1 << n) - 1
    u = node_to_index[end]
    min_cost = float('inf')
    end_node = -1

    for i in range(n):
        if dp[full_mask][i] < min_cost and graph.has_edge(index_to_node[i], end):
            min_cost = dp[full_mask][i]
            end_node = i

    if end_node == -1:
        return []

    path = [end]
    mask = full_mask
    u = end_node

    while u != -1:
        path.append(index_to_node[u])
        next_u = parent[mask][u]
        mask ^= (1 << u)
        u = next_u

    path.reverse()
    return path
