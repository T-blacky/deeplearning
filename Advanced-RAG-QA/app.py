from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import compute_bleu, compute_rouge_l
import json

def load_questions(path: str):
    """
    Expects a JSONL file with {"question": "...", "answer": "..."}
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        corpus_path="data/corpus.jsonl",
        use_query_rewriting=True,
        generator_model="t5-base"
    )

    print("Loading evaluation questions...")
    examples = load_questions("data/questions.jsonl")  # or use inline list

    bleu_scores, rouge_scores = [], []

    for i, ex in enumerate(examples):
        print(f"\nQ{i+1}: {ex['question']}")
        answer = rag.answer_question(ex['question'], top_k=5)
        print("→ Generated:", answer)
        print("✓ Ground Truth:", ex['answer'])

        bleu = compute_bleu(ex['answer'], answer)
        rouge = compute_rouge_l(ex['answer'], answer)
        print(f"BLEU: {bleu:.4f} | ROUGE-L: {rouge:.4f}")

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

    print("\nAverage BLEU:", sum(bleu_scores) / len(bleu_scores))
    print("Average ROUGE-L:", sum(rouge_scores) / len(rouge_scores))

if __name__ == "__main__":
    main()