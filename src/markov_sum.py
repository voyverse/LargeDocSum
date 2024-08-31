from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import networkx as nx

def create_transition_matrix(
    processed_data: List[Dict[str, Any]], 
    n_clusters: int , 
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


import networkx as nx
import matplotlib.pyplot as plt

def plot_networkx_graph(G: nx.Graph, title: str = "NetworkX Graph", node_size: int = 300, font_size: int = 10) -> None:
    """
    Plots a NetworkX graph using Matplotlib.

    Args:
        G (nx.Graph): The NetworkX graph to plot.
        title (str, optional): Title of the graph plot. Defaults to "NetworkX Graph".
        node_size (int, optional): Size of the nodes in the plot. Defaults to 300.
        font_size (int, optional): Font size of the node labels. Defaults to 10.

    Returns:
        None
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # Positions the nodes using the spring layout algorithm

    # Draw nodes, edges, and labels
    nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=font_size, node_color='skyblue', edge_color='gray', font_weight='bold')

    plt.title(title)
    plt.show()



def create_directed_graph(
    transition_matrix: np.ndarray, 
    summary_by_cluster: Dict[int, str],
    plot : bool = True 
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
            G.add_edge(from_intent, to_intent, weight=transition_matrix[int(i), int(j)])
    
    if plot : 
        plot_networkx_graph(G)
    return G 




import networkx as nx

def find_path(graph):
    """
    Finds the path of length n (number of nodes) with the highest score in a complete directed graph
    represented as a NetworkX DiGraph.
    
    Parameters:
    graph (nx.DiGraph): A NetworkX directed graph where each edge has a weight attribute.
    
    Returns:
    tuple: A tuple containing the highest score path and its score.
    """
    # Number of nodes in the graph
    n = len(graph)
    highest_score = -float('inf')
    best_path = []

    # Iterate over each node as the starting point
    for start_node in graph.nodes:
        current_path = [start_node]
        current_node = start_node
        current_score = 0

        # Find a path of length n
        for _ in range(n - 1):
            # Get all outgoing edges from the current node
            neighbors = list(graph.successors(current_node))
            
            # Find the edge with the maximum weight
            if not neighbors:
                break
            
            next_node = max(neighbors, key=lambda neighbor: graph[current_node][neighbor]['weight'])
            
            # Add to the path and score
            current_path.append(next_node)
            current_score += graph[current_node][next_node]['weight']
            
            # Move to the next node
            current_node = next_node

        # If the path has the required length, check its score
        if len(current_path) == n:
            if current_score > highest_score:
                highest_score = current_score
                best_path = current_path

    return best_path, highest_score



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
    path = find_path(graph, start_node, end_node)
    print(f"Most probable path: {path}")