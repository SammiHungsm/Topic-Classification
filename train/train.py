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
# 0. 強制 GPU 檢查
# ==========================================
print("\n" + "="*50)
print("🔍 系統環境檢查")
print("="*50)
if torch.cuda.is_available():
    print(f"✅ 成功偵測 GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   VRAM: {vram:.2f} GB")
    print("   >> 狀態：完美！將使用 GPU 加速訓練。")
else:
    print("❌ 嚴重警告：未偵測到 GPU！正在使用 CPU。")
    input("\n⚠️ 按 Enter 鍵強行用 CPU 繼續 (不建議)...")
print("="*50 + "\n")

# ==========================================
# 1. 載入數據 (Multi-Label Version)
# ==========================================
# ⚠️ 注意：這裡讀取的是你剛剛生成的 "multilabel" 數據
train_file = "train_data_multilabel.json" 

if not os.path.exists(train_file):
    print(f"❌ 找不到 {train_file}！請先執行 'python script/generate_multilabel_data.py'")
    exit()

print(f"📂 正在讀取 {train_file}...")
with open(train_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

# 90% 訓練, 10% 測試 (因為數據量大，10% 做 Evaluation 已經足夠)
split = dataset.train_test_split(test_size=0.1) 
dataset_train = split["train"]
dataset_test = split["test"]

print(f"   - Train Size: {len(dataset_train)}")
print(f"   - Test Size:  {len(dataset_test)}")

# ==========================================
# 2. 定義 12 個類別
# ==========================================
id2label = {
    0: "COMPLAINT", 1: "QUESTION", 2: "PRAISE", 3: "SUGGESTION", 
    4: "STATUS_CHECK", 5: "FRAUD_REPORT", 6: "CANCELLATION", 7: "SALES_LEAD", 
    8: "HUMAN_AGENT", 9: "BUG_REPORT", 10: "GREETING", 11: "IRRELEVANT"
}
label2id = {v: k for k, v in id2label.items()}

# ==========================================
# 3. 模型設定 (關鍵修改！)
# ==========================================
model_name = "xlm-roberta-large" 
print(f"\n📥 正在載入模型: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=12,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification" # 👈 關鍵：開啟 Sigmoid 模式
)

# LoRA 設定 (保持不變)
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
# 4. 訓練參數 & Metrics (關鍵修改！)
# ==========================================
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("\n🔄 正在處理數據 (Tokenization)...")
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_test = dataset_test.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ⚠️ 全新 Sigmoid 評估函數
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # 1. 將 Logits 轉為 0-1 機率 (Sigmoid)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # 2. 設定門檻 (Threshold) > 0.5 為 True
    predictions = (probs > 0.5).astype(int)
    
    # 3. 計算分數 (使用 'micro' average 適合多標籤)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )
    
    # Accuracy 在多標籤中是指 "Exact Match" (12個都要全中才算對)，所以分數通常較低，參考即可
    acc = accuracy_score(labels, predictions)
    
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./topic_multilabel_output", # 改名以免覆蓋舊模型
    learning_rate=5e-5,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4,
    num_train_epochs=3, # 3 Epochs 足夠
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

print("\n🚀 開始 Multi-Label 訓練...")
trainer.train()

# 儲存
save_path = "./final_multilabel_model" # 改名
print(f"\n✅ 訓練完成！正在儲存至 {save_path} ...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 評估報告
print("\n" + "="*50)
print("📈 最終評估報告")
print("="*50)
predictions = trainer.predict(tokenized_test)
# 預測轉換
probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
preds = (probs > 0.5).astype(int)
labels = predictions.label_ids

print(classification_report(labels, preds, target_names=list(label2id.keys()), zero_division=0))