import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig, pipeline

# ==========================================
# 1. Check Environment (GPU vs CPU)
# ==========================================
print("Checking environment...")

use_gpu = torch.cuda.is_available()

if use_gpu:
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print("Loading model in 4-bit quantization mode (saves VRAM).")
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("⚠️ No NVIDIA GPU detected, running in CPU mode.")
    print("Note: XLM-RoBERTa-XL will be very slow on CPU and requires significant RAM (>16GB).")

# ==========================================
# 2. Load XLM-RoBERTa-XL
# ==========================================
# model_id = "facebook/xlm-roberta-xl"
# If CPU cannot handle XL, uncomment below to use a smaller model:
model_id = "xlm-roberta-large"

print(f"Downloading/Loading model: {model_id} ...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if use_gpu:
        # GPU Mode: Use 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # CPU Mode
        model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float32
        )

except Exception as e:
    print(f"\n❌ Model loading failed: {e}")
    exit()

# Create Pipeline, set top_k to 10
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=10)
print("\n✅ Model loaded! Starting comprehensive test...")

# ==========================================
# 3. Define Deep Sense Test Function
# ==========================================
def test_model_deep_sense(text):
    print("="*80)
    print(f"📄 Original Text: {text}")
    print("="*80)

    # Define prompts
    # <mask> is where the model fills in the blank
    prompts = {
        "📊 Sentiment": f"'{text}' This text has a <mask> sentiment.",
        "🏷️  Topic (Chinese Prompt)": f"'{text}' 這是關於<mask>的。",
        "📂 Category (English Prompt)": f"Category: <mask>. Text: {text}",
        "😊 Emotion": f"The writer is feeling <mask>. Text: {text}",
        "🔥 Action (Chinese Prompt)": f"'{text}' 所以我應該<mask>。",
    }

    for label, prompt_text in prompts.items():
        print(f"\n>> {label}")
        print(f"   Prompt: {prompt_text}")
        print("-" * 40)
        
        try:
            results = fill_mask(prompt_text)
            # Handle list vs dict return types
            if isinstance(results, dict): results = [results]
            
            # List raw data
            for i, res in enumerate(results):
                token = res['token_str'].strip()
                score = res['score']
                # Filter empty tokens
                if len(token) > 0:
                    print(f"   {i+1:2d}. {token:<15} (Confidence: {score:.2%})")
                    
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")

    print("\n")

# ==========================================
# 4. Run Test Data (Mixed Chinese & English)
# ==========================================
test_sentences = [
    "你哋個 App 更新完之後狂彈 App，垃圾！",       # Expected: Negative / Complaint
    "陳經理服務好好，好細心，值得表揚。",           # Expected: Positive / Praise
    "請問分行星期六幾點開門？",                   # Expected: Question / Inquiry
    "我張卡無端端被人碌咗幾千蚊，有無人幫到我？",   # Expected: Fraud / Urgent / Bank
    "High efficiency and very helpful staff!"       # Expected: Positive / Good
]

print(f"Preparing deep analysis for {len(test_sentences)} sentences...\n")

for sentence in test_sentences:
    test_model_deep_sense(sentence)

print("Test completed!")