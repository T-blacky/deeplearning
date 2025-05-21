from transformers import T5ForConditionalGeneration
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
import torch


def load_model(model_name, device):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return model


def train(model, train_dataset, dev_dataset, tokenizer, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=4)

    best_score = 0
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        score = test(model, dev_loader, tokenizer, device)
        print(f"Epoch {epoch+1}, ROUGE-L: {score:.4f}")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'best_model.pth')


@torch.no_grad()
def test(model, loader, tokenizer, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_score, count = 0, 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for p, r in zip(preds, refs):
            total_score += scorer.score(r, p)['rougeL'].fmeasure
            count += 1

    return total_score / count if count > 0 else 0.0
