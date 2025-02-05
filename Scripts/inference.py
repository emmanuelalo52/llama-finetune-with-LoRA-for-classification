import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from Dataloader import ag_news
from Llamaclassification import LLamaCalssification, LlamaConfig
from torch.utils.data import DataLoader, Dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def preprocess_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, df):
        self.input_ids = df['text'].apply(lambda x: preprocess_data({"text": x})["input_ids"][0])
        self.attention_mask = df['text'].apply(lambda x: preprocess_data({"text": x})["attention_mask"][0])
        self.labels = df['label'].values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids.iloc[idx],
            "attention_mask": self.attention_mask.iloc[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Convert dataset to DataLoader
train_dataset = TextDataset(ag_news["train"])
test_dataset = TextDataset(ag_news["test"])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Configure QLoRA quantization
def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

# Load model
config = LlamaConfig(vocab_size=tokenizer.vocab_size, num_classes=4, device="cuda" if torch.cuda.is_available() else "cpu")
model = LLamaCalssification(config).to(config.device)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Run inference
def classify_text(input_ids, attention_mask):
    input_ids, attention_mask = input_ids.to(config.device), attention_mask.to(config.device)
    with torch.no_grad():
        logits = model(input_ids, start_pos=0)
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Evaluate on test data
correct = 0
total = 0
for batch in test_dataloader:
    predictions = [classify_text(i.unsqueeze(0), a.unsqueeze(0)) for i, a in zip(batch['input_ids'], batch['attention_mask'])]
    correct += sum(p == l for p, l in zip(predictions, batch['labels'].tolist()))
    total += len(batch['labels'])

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
