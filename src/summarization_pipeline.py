from typing import Dict , List , Any , Union
CHUNKING_METHODS = ["recursive" , "semantic"]


def pipeline(
    chunking_method : str , 
    chunking_params : Dict , 
    embed_model_name : str  ,
    doc : str , 
    num_clusters : int , 
    top_k : int ,  # top k closest to cluster centroid 
    system_prompt : str , 
    
) :
    assert chunking_method in CHUNKING_METHODS
    if chunking_method == "recursive" : 
        assert "max_length" in chunking_params.keys() and "overlap" in chunking_params.keys()
    elif chunking_method == "semantic" : 
        assert "threshold" in chunking_params.keys()
    assert num_clusters > 0 
    assert num_clusters > top_k and top_k > 0 
    assert doc != None and system_prompt != None 
    
    
    