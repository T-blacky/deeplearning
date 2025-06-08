from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import compute_rouge_l
from retriever.hybrid_retriever import HybridRetriever
import torch
import torch.nn as nn
import torch.optim as optim
import json

def load_questions(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def train_gate():
    questions = load_questions("data/questions.jsonl")
    pipeline = RAGPipeline(corpus_path="data/corpus.jsonl", use_query_rewriting=True)
    gate_net = pipeline.retriever.gate_net

    optimizer = optim.Adam(gate_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  # simple regression loss

    for epoch in range(3):  # tune as needed
        total_loss = 0

        for example in questions:
            question, reference = example['question'], example['answer']
            retrieved_docs = pipeline.retriever.retrieve(question, top_k=5)

            # Combine retrieved docs
            context = "\n".join([doc for _, doc, _ in retrieved_docs])
            generated = pipeline.generator.generate(question, context)

            with torch.no_grad():
                reward = compute_rouge_l(reference, generated)

            # Predict fusion weights again for loss
            loss = 0
            for doc_id, _, _ in retrieved_docs:
                bm25_score = torch.tensor([pipeline.retriever._bm25_score(doc_id)], dtype=torch.float32)
                faiss_score = torch.tensor([pipeline.retriever._faiss_score(doc_id)], dtype=torch.float32)
                input_vec = torch.cat([bm25_score, faiss_score]).unsqueeze(0)

                gate = gate_net(input_vec)
                pred_score = gate * bm25_score + (1 - gate) * faiss_score
                loss += loss_fn(pred_score, torch.tensor([[reward]]))

            loss = loss / len(retrieved_docs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(questions):.4f}")

if __name__ == "__main__":
    train_gate()
