import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F

# ==========================================
# 1. Setup Paths and Parameters
# ==========================================
base_model_name = "xlm-roberta-large"
lora_model_path = "./final_topic_model" 

# Define labels (English)
id2label = {0: "COMPLAINT", 1: "INQUIRY", 2: "PRAISE"}
label2id = {"COMPLAINT": 0, "INQUIRY": 1, "PRAISE": 2}

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
    # A. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # B. Load Base Model
    # num_labels must match your training (3 classes)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # C. Load LoRA Adapter
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.to(device)
    model.eval() 

    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Load failed: {e}")
    print(f"Please ensure '{lora_model_path}' exists and training is complete.")
    exit()

# ==========================================
# 4. Prediction Function
# ==========================================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Calculate Probabilities (Softmax)
        probs = F.softmax(logits, dim=-1)[0]
        
    # Create a list of (label_name, score) tuples
    results = []
    for i, score in enumerate(probs):
        results.append((id2label[i], score.item()))
    
    # Sort by score descending (Highest confidence first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

# ==========================================
# 5. Interactive Test Loop
# ==========================================
print("\n" + "="*50)
print("🤖 Topic Classification Mode")
print("Type 'q' or 'exit' to quit")
print("="*50)

while True:
    text = input("\nEnter sentence: ")
    
    if text.lower() in ['q', 'exit', 'quit']:
        print("Bye!")
        break
    
    if not text.strip():
        continue

    # Get sorted results
    sorted_results = predict(text)
    
    # Get the Top 1 result
    top_label, top_score = sorted_results[0]
    
    print("-" * 40)
    print(f"Input: {text}")
    print(f"Prediction: {top_label}")
    print(f"Confidence: {top_score:.2%}")
    print("-" * 40)
    
    # Show All Rankings (Top K)
    print("Detailed Distribution:")
    for rank, (label, score) in enumerate(sorted_results):
        print(f"  {rank+1}. {label:<15} : {score:.2%}")