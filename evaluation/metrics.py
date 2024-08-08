import spacy
import textdescriptives as td
import evaluate
import bert_score
import pandas as pd

def calculate_rouge_scores(candidates, references):
    """
    Calculate ROUGE scores for given candidates and references.

    Args:
        candidates (list of str): List of candidate summaries.
        references (list of list of str): List of reference summaries for each candidate.

    Returns:
        dict: ROUGE scores including ROUGE-1, ROUGE-2, ROUGE-L.
    """
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=candidates, references=references)
    return results

def calculate_bertscore(candidates, references):
    """
    Calculate BERTScore for given candidates and references.

    Args:
        candidates (list of str): List of candidate summaries.
        references (list of list of str): List of reference summaries for each candidate.

    Returns:
        dict: BERTScore including Precision, Recall, and F1.
    """
    # Ensure the number of candidates matches the number of sets of reference summaries
    assert len(candidates) == len(references), "Different number of candidates and sets of references"

    # Flatten the list of references for each candidate
    flat_references = [' '.join(ref) for ref in references]

    # Compute BERTScore
    P, R, F1 = bert_score.score(candidates, flat_references, lang='en', verbose=True)
    
    # Average BERTScore values for candidates
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    return {
        'bert_precision': avg_precision,
        'bert_recall': avg_recall,
        'bert_f1': avg_f1
    }


def calculate_coherence(text):
    """
    Calculate first-order and second-order coherence for a given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: Coherence scores including first-order and second-order coherence.
    """
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textdescriptives/coherence")

    doc = nlp(text)
    coherence = doc._.coherence
    first_order_coherence = doc._.first_order_coherence_values
    second_order_coherence = doc._.second_order_coherence_values
    
    return {
        'first_order_coherence': coherence['first_order_coherence'],
        'second_order_coherence': coherence['second_order_coherence'],
        'first_order_values': first_order_coherence,
        'second_order_values': second_order_coherence
    }

def print_rouge_scores(results):
    """
    Print ROUGE scores from the results dictionary.

    Args:
        results (dict): ROUGE scores including ROUGE-1, ROUGE-2, ROUGE-L.
    """
    print(f"ROUGE-1 Score: {results['rouge1']:.4f}")
    print(f"ROUGE-2 Score: {results['rouge2']:.4f}")
    print(f"ROUGE-L Score: {results['rougeL']:.4f}")
    if 'rougeLsum' in results:
        print(f"ROUGE-L Summary Score: {results['rougeLsum']:.4f}")

def print_bertscore(results):
    """
    Print BERTScore from the results dictionary.

    Args:
        results (dict): BERTScore including Precision, Recall, and F1.
    """
    print(f"BERTScore Precision: {results['bert_precision']:.4f}")
    print(f"BERTScore Recall: {results['bert_recall']:.4f}")
    print(f"BERTScore F1: {results['bert_f1']:.4f}")

def print_coherence(results):
    """
    Print coherence scores from the results dictionary.

    Args:
        results (dict): Coherence scores including first-order and second-order coherence.
    """
    print(f"First-Order Coherence: {results['first_order_coherence']:.4f}")
    print(f"Second-Order Coherence: {results['second_order_coherence']:.4f}")
    print(f"First-Order Coherence Values: {results['first_order_values']}")
    print(f"Second-Order Coherence Values: {results['second_order_values']}")

def main():
    # Example usage
    candidates = [
        "Summarization is cool",
        "I love Machine Learning",
        "Good night"
    ]

    references = [
        ["Summarization is beneficial and cool", "Summarization saves time"],
        ["People are getting used to Machine Learning", "I think I love Machine Learning"],
        ["Good night everyone!", "Night!"]
    ]

    # Calculate and print ROUGE scores
    rouge_results = calculate_rouge_scores(candidates, references)
    print_rouge_scores(rouge_results)
    
    # Calculate and print BERTScore
    bert_results = calculate_bertscore(candidates, references)
    print_bertscore(bert_results)

    # Calculate and print coherence
    text = "The world is changed. I feel it in the water. I feel it in the earth. I smell it in the air. Much that once was is lost, for none now live who remember it."
    coherence_results = calculate_coherence(text)
    print_coherence(coherence_results)

if __name__ == "__main__":
    main()
