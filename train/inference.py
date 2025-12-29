import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F
import time

# ==========================================
# 1. Setup Paths and Parameters
# ==========================================
base_model_name = "xlm-roberta-large"
lora_model_path = "./final_topic_model" 

# ⚠️ CRITICAL: Must match the 12 classes from training!
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
# 2. Setup Environment
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
# 3. Load Model (Base + LoRA Adapter)
# ==========================================
print("Loading model, please wait...")

try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=12, 
        id2label=id2label,
        label2id=label2id
    )
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.to(device)
    model.eval() 
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Load failed: {e}")
    print(f"Make sure you have re-trained the model with the new 12-class dataset.")
    exit()

# ==========================================
# 4. Prediction Function
# ==========================================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]
        
    results = []
    for i, score in enumerate(probs):
        results.append((id2label[i], score.item()))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ==========================================
# 5. PRE-DEFINED TEST CASES (自動測試清單)
# ==========================================
test_cases = [
    # --- A. Core Intents ---
    "I want to apply for a tax loan, what is the interest rate?",
    "你哋個 App 更新完之後完全入唔到，一直轉圈圈。",
    "我想 waive 咗個 annual fee 佢，得唔得？",
    "Cut card.",
    "I am quite disappointed with the waiting time at your Central branch.",
    
    # --- B. High Risk / Urgent ---
    "唔見咗張卡，快啲幫我停咗佢！",
    "I received an SMS for a transaction I didn't make.",
    "轉會啦，你哋啲息咁低，我要 cut 戶口。",
    
    # --- C. Mixed / Tricky ---
    "Although the queue was long, the staff Manager Chan was very helpful.",
    "交咗表成個禮拜都無聲氣，批左未呀？",
    "同個機械人講極都唔明，叫你經理出黎。",
    "我有筆閒錢，有無咩高息戶口推介？",
    
    # --- D. System / Suggestions ---
    "強烈建議你哋加返指紋登入，每次入密碼好煩。",
    "Error 503 keeps popping up when I try to transfer money.",
    
    # --- E. Extreme / Edge Cases ---
    "早晨。",
    "Buy Bitcoin now! Click here for free crypto.",
    "asdfghjkl",
    "個制撳極都無反應既？壞左呀？",
    "Help.",
    "I wnat to open acount pls."
]

CONFIDENCE_THRESHOLD = 0.60 

print("\n" + "="*60)
print(f"🚀 Running Automatic Batch Test ({len(test_cases)} sentences)...")
print("="*60)

for text in test_cases:
    # Run Prediction
    sorted_results = predict(text)
    top_label, top_score = sorted_results[0]
    
    # Formatting Output
    print(f"\n📝 Input:      {text}")
    print(f"🎯 Prediction: {top_label}")
    
    # Colorize confidence output (Simulated)
    conf_str = f"{top_score:.2%}"
    if top_score < CONFIDENCE_THRESHOLD:
        print(f"⚠️  Confidence: {conf_str} (LOW - Needs Review)")
        print(f"   ↳ 2nd Guess: {sorted_results[1][0]} ({sorted_results[1][1]:.2%})")
    else:
        print(f"✅ Confidence: {conf_str}")
    
    time.sleep(0.1) # Add tiny delay for visual flow

print("\n" + "="*60)
print("🏁 Batch Test Complete!")
print("="*60)

# ==========================================
# 6. Interactive Loop (Manual Mode)
# ==========================================
print("\n🤖 Switching to Interactive Mode...")
print("Type 'q' or 'exit' to quit")

while True:
    text = input("\nEnter sentence: ")
    
    if text.lower() in ['q', 'exit', 'quit']:
        print("Bye!")
        break
    
    if not text.strip():
        continue

    sorted_results = predict(text)
    top_label, top_score = sorted_results[0]
    
    print("-" * 40)
    print(f"Prediction: {top_label}")
    print(f"Confidence: {top_score:.2%}")
    
    if top_score < CONFIDENCE_THRESHOLD:
        print("⚠️  Warning: Low confidence.")
        print(f"   (Next best: {sorted_results[1][0]})")
        
    print("-" * 40)