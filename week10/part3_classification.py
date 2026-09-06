import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertModel
from datasets import load_dataset

# --- BERT-1: Load dataset ---
data = load_dataset("fancyzhx/ag_news")
train_data = data['train'].select(range(2000))
test_data = data['test'].select(range(500))
label_names = train_data.features['label'].names

# --- BERT-2: Tokenize ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True)
test_loader = DataLoader(tokenized_test, batch_size=16)

# --- BERT-3: Model setup ---
backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")

for param in backbone.parameters():
    param.requires_grad = False

classifier = nn.Linear(768, 4)

total = sum(p.numel() for p in backbone.parameters())
trainable_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
trainable_head = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print(f"Backbone: {total:,} total, {trainable_backbone} trainable")
print(f"Classifier: {trainable_head:,} trainable")

# --- BERT-3: Training loop ---
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

backbone.eval()  # backbone stays in eval mode — no dropout, no updates

for epoch in range(3):
    classifier.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        # Forward through frozen backbone
        with torch.no_grad():
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Grab CLS token output (first token)
        cls_vector = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Forward through trainable classifier
        logits = classifier(cls_vector)  # (batch, 4)

        loss = loss_fn(logits, labels)

        # Backward — only classifier weights get updated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/3 — Loss: {avg_loss:.4f}")

print("\nTraining complete. Ready for BERT-4.")