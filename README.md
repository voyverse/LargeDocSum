# Large Document Summarization  

## Overview  
This project introduces a novel technique to summarize large documents through a series of methodical steps aimed at extracting the essence of the content. The approach leverages chunking, embedding, clustering, and language models to produce coherent summaries.  

## Methodology  

1. **Chunking**: The large document is divided into manageable chunks.  
2. **Embedding**: Each chunk is embedded into a high-dimensional space using embedding techniques.  
3. **Clustering**: The embedded chunks are clustered to group similar content together.  
4. **Cluster Processing**:  
   - Outlier removal to ensure quality clusters.  
   - Reclustering for optimal grouping.  
5. **Centroid Extraction**: Chunks closest to each cluster's centroid are selected.  
6. **Common Meaning Extraction**: These centroid chunks are passed to a Language Learning Model (LLM) to extract common meanings, storylines, or abstract ideas.  
7. **Cluster Labeling**: Each cluster is labeled based on the LLM outputs.  
8. **Transition Matrix Construction**: A transition matrix is created to represent the probabilities of transitioning from one cluster to another.  
9. **Ranking and Path Finding**:  
   - Identify the cluster ranked first in the document's flow.  
   - Determine the most probable path through the transition graph.  
10. **Summary Generation**: The most probable path is sent to the LLM to generate a comprehensive summary of the document.  

## Installation & Setup  

To use this project, follow these steps:  

### **1. Clone the Repository with Submodules**  
Since the project depends on BLEURT for evaluation, make sure to clone the repository with submodules:  

```bash
git clone --recursive https://github.com/voyverse/LargeDocSum.git
```

If you already cloned the repository without `--recursive`, initialize and update the submodule manually:  

```bash
cd LargeDocSum
git submodule update --init --recursive
```

### **2. Create a Virtual Environment and Install Dependencies**  
It is recommended to create a virtual environment to manage dependencies:  

```bash
cd LargeDocSum
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Install BLEURT**  
Navigate to the BLEURT submodule and install it:  

```bash
cd third_party/bleurt
pip install .
```

### **4. Download BLEURT Checkpoint**  
BLEURT requires a pre-trained model checkpoint. Download and unzip it as follows:  

```bash
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
```

### **5. Run the Project**  
Now, you can start using the document summarization technique as described in the methodology.

## Usage  
This technique can be applied to any large document to create concise and meaningful summaries, making it easier to digest extensive information efficiently.
