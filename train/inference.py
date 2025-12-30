import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import time

# ==========================================
# 1. Setup Paths
# ==========================================
base_model_name = "xlm-roberta-large"
# ⚠️ 請確保這裡指向你訓練好的 LoRA 模型資料夾
lora_model_path = "./final_multilabel_model" 

# 12 Classes (需與訓練時一致)
id2label = {
    0: "COMPLAINT", 1: "QUESTION", 2: "PRAISE", 3: "SUGGESTION", 
    4: "STATUS_CHECK", 5: "FRAUD_REPORT", 6: "CANCELLATION", 7: "SALES_LEAD", 
    8: "HUMAN_AGENT", 9: "BUG_REPORT", 10: "GREETING", 11: "IRRELEVANT"
}
label2id = {v: k for k, v in id2label.items()}

# ==========================================
# 2. Load Model (Sigmoid Mode)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading Multi-Label model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=12, 
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )
    # 載入 LoRA 外掛
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.to(device)
    model.eval() 
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Load failed: {e}")
    print("💡 提示: 請檢查 'lora_model_path' 是否正確指向你的模型資料夾")
    exit()

# ==========================================
# 3. Prediction Function (Sigmoid)
# ==========================================
def predict_multilabel(text, threshold=0): # 預設門檻值設為 0.3，方便觀察多標籤
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # ⚠️ Use Sigmoid for independent probabilities
        probs = torch.sigmoid(logits)[0]
        
    results = []
    for i, score in enumerate(probs):
        results.append((id2label[i], score.item()))
    
    # Sort descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Filter Active Labels > Threshold
    active_labels = [r for r in results if r[1] > threshold]
    
    return active_labels, results 

# ==========================================
# 4. Test Cases (包含 Stress Test)
# ==========================================
test_cases = [
   
    # ==========================================
    # 🔥 F. Stress Test (5+ Topics / Complex)
    # ==========================================
    # 預期: GREETING, BUG_REPORT, FRAUD_REPORT, COMPLAINT, CANCELLATION, HUMAN_AGENT
    "早晨，你個 App 死機搞到我俾人盜用咗張卡，你地服務真係好差，我要即刻 Cut 卡，快啲駁去真人聽啦！",

    # 預期: GREETING, PRAISE, SUGGESTION, SALES_LEAD, QUESTION
    "Hi, your service is great and I love the new features, but maybe add dark mode? Also, I want to apply for a loan, what are the rates?",

    # 預期: BUG_REPORT, STATUS_CHECK, COMPLAINT, CANCELLATION, HUMAN_AGENT
    "個系統壞咗查唔到申請進度，仲扣錯錢？我想投訴同埋 Cut 左個戶口佢，搵個經理出黎解釋下！",

    # 預期: GREETING, FRAUD_REPORT, SALES_LEAD, QUESTION, HUMAN_AGENT
    "Hello, I lost my phone and card, can you block it? Also I want to check my balance and buy travel insurance, please help."
]

print("\n" + "="*60)
print(f"🚀 Running Multi-Label Batch Test ({len(test_cases)} samples)...")
print("="*60)

for text in test_cases:
    # 這裡 threshold 用 0.4 來過濾，你可以根據需要調低 (例如 0.2) 來捕捉更多微弱的意圖
    active, all_res = predict_multilabel(text, threshold=0)
    
    print(f"\n📝 Input: \"{text}\"")
    
    if not active:
        # If nothing > Threshold, show the top guess (Low Confidence)
        top_label, top_score = all_res[0]
        print(f"   ⚠️  Low Confidence (Top: {top_label} - {top_score:.2%})")
    else:
        # Show all active labels
        for label, score in active:
            print(f"   ✅ {label:<15} {score:.2%}")
            
    time.sleep(0.02) # Tiny delay for readability

print("\n" + "="*60)
print("🏁 Batch Test Complete!")
print("="*60)

# ==========================================
# 5. Interactive Loop
# ==========================================
print("\n🤖 Interactive Mode (Sigmoid)")
print("Type 'q' to quit")

while True:
    text = input("\nEnter sentence: ")
    if text.lower() in ['q', 'exit']: break
    
    active, all_res = predict_multilabel(text, threshold=0.3)
    
    print("-" * 50)
    if active:
        print(f"Detected {len(active)} Intents:")
        for label, score in active:
            bar = "█" * int(score * 20)
            print(f"{label:<15} {score:.2%} [{bar:<20}]")
    else:
        print("No strong intent detected.")
        print(f"Top guess: {all_res[0][0]} ({all_res[0][1]:.2%})")
    print("-" * 50)