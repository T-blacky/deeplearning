from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_bleu(reference: str, prediction: str) -> float:
    reference_tokens = reference.lower().split()
    prediction_tokens = prediction.lower().split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing)

def compute_rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score['rougeL'].fmeasure