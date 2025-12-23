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
# 1. 載入數據
# ==========================================
data_file = "train_data_topic.json"

if not os.path.exists(data_file):
    print(f"❌ 找不到 {data_file}！請先執行 'python script/generate_data.py'")
    exit()

print(f"正在讀取 {data_file}...")
with open(data_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
# 80% 訓練, 20% 測試
dataset = dataset.train_test_split(test_size=0.2) 

id2label = {0: "COMPLAINT", 1: "INQUIRY", 2: "PRAISE"}
label2id = {"COMPLAINT": 0, "INQUIRY": 1, "PRAISE": 2}

# ==========================================
# 2. 模型設定
# ==========================================
model_name = "xlm-roberta-large" 
print(f"正在載入模型: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

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
model.print_trainable_parameters()

# ==========================================
# 3. 訓練參數
# ==========================================
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./topic_classification_output",
    learning_rate=5e-5,
    per_device_train_batch_size=4, # 4060 8GB VRAM 安全設定
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    dataloader_num_workers=0,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("🚀 開始訓練...")
trainer.train()

# 儲存
save_path = "./final_topic_model"
print(f"✅ 訓練完成！正在儲存至 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 評估報告
print("\n=== 最終評估報告 ===")
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
print(classification_report(labels, preds, target_names=label2id.keys()))