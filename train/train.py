import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# ==========================================
# 0. Forced GPU Check
# ==========================================
print("\n" + "="*50)
print("🔍 System Environment Check")
print("="*50)
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   VRAM: {vram:.2f} GB")
    print("   >> Status: Perfect! Training will use GPU.")
else:
    print("❌ CRITICAL WARNING: No GPU detected! Using CPU.")
    print("   >> Training will be extremely slow.")
    print("   >> Recommended: Stop (Ctrl+C) and fix PyTorch with:")
    print("      uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    input("\n⚠️ Press Enter to force CPU execution (Not Recommended)...")
print("="*50 + "\n")

# ==========================================
# 1. Smart Data Loading Strategy (IMPROVED)
# ==========================================
train_file = "train_data_topic.json"  # Your 50k synthetic file
test_file = "test_data_gold.json"     # Your MANUAL/HARD test file

# 1. Load Training Data
if not os.path.exists(train_file):
    print(f"❌ Missing {train_file}! Please run 'python script/generate_huge_data.py'")
    exit()

print(f"📂 Loading Training Data: {train_file}...")
with open(train_file, "r", encoding="utf-8") as f:
    train_raw = json.load(f)
dataset_train = Dataset.from_list(train_raw)

# 2. Load Test Data (Logic: Prefer separate file, fallback to split)
if os.path.exists(test_file):
    print(f"✅ Found Gold Standard Test Set: {test_file}")
    print("   >> Evaluating on REAL/HARD data. This is excellent!")
    with open(test_file, "r", encoding="utf-8") as f:
        test_raw = json.load(f)
    dataset_test = Dataset.from_list(test_raw)
else:
    print(f"⚠️ 'Gold Standard' test set ({test_file}) not found.")
    print("   >> Falling back to random 80/20 split from training data.")
    print("   >> (Note: Accuracy might be artificially high due to template similarity)")
    
    # Split the training data
    split = dataset_train.train_test_split(test_size=0.1) # 10% is enough for 50k samples
    dataset_train = split["train"]
    dataset_test = split["test"]

print(f"   - Train Size: {len(dataset_train)}")
print(f"   - Test Size:  {len(dataset_test)}")

# ==========================================
# 2. Label Definitions (12 Classes)
# ==========================================
id2label = {
    0: "COMPLAINT", 
    1: "QUESTION", 
    2: "PRAISE", 
    3: "SUGGESTION", 
    4: "STATUS_CHECK", 
    5: "FRAUD_REPORT", 
    6: "CANCELLATION", 
    7: "SALES_LEAD", 
    8: "HUMAN_AGENT", 
    9: "BUG_REPORT", 
    10: "GREETING", 
    11: "IRRELEVANT"
}
label2id = {v: k for k, v in id2label.items()}

# ==========================================
# 3. Model Setup
# ==========================================
model_name = "xlm-roberta-large" 
print(f"\n📥 Loading Model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=12,  # 12 Classes
    id2label=id2label,
    label2id=label2id
)

# LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False, 
    r=16,
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["query", "value"], 
    modules_to_save=["classifier"] 
)
model = get_peft_model(model, peft_config)
print("\n📊 Trainable Parameters:")
model.print_trainable_parameters()

# ==========================================
# 4. Training Arguments (IMPROVED)
# ==========================================
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("\n🔄 Tokenizing data...")
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_test = dataset_test.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./topic_classification_output",
    learning_rate=5e-5,
    
    # --- Hardware Optimization for RTX 4060 (8GB) ---
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, # Effective Batch Size = 16
    
    # --- Preventing Overfitting ---
    num_train_epochs=3, # Reduced from 10 to 3 (50k data is large enough)
    
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    fp16=torch.cuda.is_available(), 
    logging_steps=100, 
    dataloader_num_workers=0, 
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n🚀 Starting Training...")
trainer.train()

# ==========================================
# 5. Saving & Evaluation
# ==========================================
save_path = "./final_topic_model"
print(f"\n✅ Training Complete! Saving to {save_path} ...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("\n" + "="*50)
print("📈 Final Evaluation Report")
print("="*50)
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
print(classification_report(labels, preds, target_names=list(label2id.keys())))