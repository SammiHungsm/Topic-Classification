import json
import random

def generate_huge_data(num_samples=50000):
    print(f"🔄 Generating {num_samples} high-quality synthetic data points (12 Topics)...")
    
    # ==========================================
    # 1. Expanded Vocabulary (English & Cantonese)
    # ==========================================
    slots = {
        # --- Common Nouns (English) ---
        "issue": ["App keeps crashing", "Frozen screen", "Cannot load", "White screen", "Transfer failed", "No OTP received", "Login failed", "Biometric fail", "Glitch", "Force close"],
        "status": ["No one answering", "Line busy", "Cut off", "Voicemail", "Waited 30 mins", "Just music", "Disconnected", "No reply"],
        "error": ["Error 404", "System Failure", "Timeout", "Connection Error", "Server Busy", "Code: 503", "Unknown Error", "Access Denied"],
        "behavior": ["Very rude", "Bad attitude", "Ignored me", "Hung up", "Impatient", "Unhelpful", "Unprofessional", "Angry tone"],
        "fee": ["Handling fee", "Annual fee", "Late charge", "Overdraft interest", "Admin fee", "Replacement fee", "Hidden charge"],
        "feeling": ["Disappointed", "Angry", "Unsatisfied", "Helpless", "Furious", "Shocked", "Unbelievable", "Desperate"],
        "item": ["Credit Card", "PIN letter", "ATM card", "Cheque book", "Statement", "Security token", "New card", "Welcome gift"],
        "time": ["Saturday", "Public Holiday", "Weekday", "Lunar New Year", "Black Rainstorm", "Typhoon 8", "Lunch hour", "After work"],
        "product": ["Credit Card", "Personal Loan", "Mortgage", "Insurance", "MPF", "Fixed Deposit", "Tax Loan", "Travel Insurance"],
        "info": ["Address", "Phone number", "Email", "Date of birth", "Signature", "Transfer limit", "Credit limit"],
        "feature": ["Overseas withdrawal", "FPS", "Online Banking", "Stock trading", "FX exchange", "PayMe Top-up", "Apple Pay", "Google Pay"],
        "staff": ["Manager Chan", "Teller", "Hotline staff", "Branch manager", "Relationship Manager", "That male staff", "Receptionist", "CS Agent"],
        "trait": ["Polite", "Patient", "Professional", "Attentive", "Friendly", "Efficient", "Clear", "Helpful"],
        "service": ["service", "support", "arrangement", "follow-up", "speed", "experience", "attitude"],
        "improvement": ["Much smoother", "Faster", "Easier to use", "More stable", "Nicer UI", "Convenient", "User Friendly"],
        "location": ["Mong Kok Branch", "Central HQ", "Kwun Tong Branch", "Shatin Branch", "Tuen Mun Branch", "Causeway Bay Branch"],
        "amount": ["Few hundred", "10k", "Few thousand", "Cents", "The fee"],
        
        # --- Specific for New Topics ---
        "fraud_keyword": ["Stolen", "Lost", "Unauthorzied transaction", "Hacked", "Phishing", "Scam", "Suspicious SMS"],
        "cancel_action": ["Cancel", "Terminate", "Close account", "Cut card", "Stop service", "Unsubscribe"],
        "sales_interest": ["Interested in", "Want to buy", "Looking for", "Apply for", "Open account"],
        "bug_tech": ["Button not working", "UI broken", "Font too small", "Battery drain", "Lagging"],
        "greeting_word": ["Hi", "Hello", "Good morning", "Good afternoon", "Hey there"],
        "junk_word": ["asdf", "test", "1234", "buy crypto", "click link", "promotion"],

        # --- Cantonese / Mix (New Strategy C) ---
        "issue_hk": ["狂彈 App", "Hang 機", "Load 唔到", "白畫面", "轉唔到數", "收唔到 OTP", "無法登入", "指紋 Login 失敗", "畫面卡住", "閃退"],
        "status_hk": ["無人聽", "長期繁忙", "Cut 線", "飛留言", "等咗半個鐘", "一直播音樂", "沒人接聽", "斷線"],
        "behavior_hk": ["態度極差", "黑口黑面", "十問九唔應", "掛我電話", "語氣好差", "完全唔熟書", "遊花園", "唔耐煩"],
        "fee_hk": ["手續費", "年費", "過期罰款", "透支利息", "行政費", "補卡費", "隱藏收費"],
        "feeling_hk": ["失望", "憤怒", "不滿", "無奈", "火滾", "O晒嘴", "難以置信", "絕望"],
        "trait_hk": ["有禮貌", "有耐性", "專業", "細心", "友善", "效率高", "清楚", "幫得手"],
        "cancel_action_hk": ["Cut", "取消", "停用", "唔用", "Terminate"],
        "fraud_keyword_hk": ["俾人盜用", "唔見咗", "被 Hack", "懷疑被騙", "不明交易"],
        "sales_interest_hk": ["想申請", "想買", "有興趣", "睇緊", "想開"],
    }

    # ==========================================
    # 2. Random Modifiers
    # ==========================================
    prefixes = [
        "", "", "", "Hi, ", "Hello, ", "Urgent: ", "Question: ", "Help, ", "Complaint: ", 
        "To Manager: ", "Excuse me, ", "Regarding: ", "Hey, ", "May I ask, ",
        "你好, ", "喂, ", "想問下, ", "救命, ", "急問: ", "唔該, "
    ]
    
    suffixes = [
        "", "", "", ".", "!", "...", "😡", "👍", "🙏", "😤", "🤬", "❤️", 
        " thx.", " please help.", " ASAP!", " waiting.", " thanks.", " pls follow up.",
        " 唔該.", " 煩請跟進.", " 快啲覆.", " 等緊你."
    ]

    # ==========================================
    # 3. Topic Templates (12 Categories) - NOW WITH MIXED LANGUAGES
    # ==========================================
    
    # 0. COMPLAINT (Mixed English & Cantonese)
    complaints = [
        "Your App {issue}, rubbish!", "Called many times, always {status}.", 
        "Cannot Login, shows {error}.", "Staff at {location} was {behavior}.",
        "Why charged me {fee}? Refund!", "Very {feeling} with your service.",
        "Waiting for {item} for too long!", "Never been so {feeling} before.",
        "System keeps showing {error}.", "Staff {behavior}, I want to complain.",
        # Cantonese / Mix
        "你哋個 App {issue_hk}，垃圾！", "打極電話都 {status_hk}，火都黎埋。",
        "分行職員 {behavior_hk}，完全唔想幫手。", "無端端收多我 {fee_hk}，回水！",
        "對你哋服務好 {feeling_hk}，以後唔用。", "個 App 更新完之後狂 {issue_hk}，用唔到呀！",
        "點解無端端扣我 {amount}？解釋下好沃。", "Crazy admin fee {fee}, 回水！"
    ]

    # 1. QUESTION
    questions = [
        "What time does {location} open on {time}?", "How long to approve {product}?",
        "How to change my {info}?", "Can I waive the {fee}?",
        "How to activate {feature}?", "Interest rate for {product}?",
        "Lost my {item}, procedure?", "Any promotion for {product}?",
        "Where is {location}?", "How to reset password?",
        # Cantonese / Mix
        "請問分行 {time} 幾點開門？", "申請 {product} 要幾耐批？",
        "我想更改 {info}，手續係點？", "有無得豁免 {fee_hk}？",
        "請問點樣啟動 {feature} 功能？", "我想問下 {product} 既利息係幾多？",
        "如果我唔見咗 {item}，應該點做？", "有無 {product} 既最新優惠？",
        "想問 {location} 有無得做 {feature}？", "FPS transfer limit 係幾多?"
    ]

    # 2. PRAISE
    praises = [
        "{staff} is very {trait}, good job.", "Thanks for solving {issue}, fast.",
        "CS was {trait}, explained clearly.", "Excellent {service}!",
        "App update is {improvement}, nice.", "Satisfied with {service}.",
        "Professional {staff}.", "Very {trait} staff at {location}.",
        "Great experience.", "Appreciate the {trait} service.",
        # Cantonese / Mix
        "{staff} 服務好好，好 {trait_hk}，值得表揚。", "多謝你哋幫我解決 {issue_hk}，效率好高。",
        "上次個客服好 {trait_hk}，解釋得好清楚。", "個 App 更新左之後 {improvement}，正！",
        "十分滿意你哋既 {service}。", "感受到你哋既專業，{staff} 好幫手。",
        "{staff} 好有禮貌，令我好開心。", "很少見到咁 {trait_hk} 既銀行職員。"
    ]

    # 3. SUGGESTION
    suggestions = [
        "Please add Dark Mode to the App.", "Suggest to add more {feature}.",
        "Better if you open on {time}.", "App needs {improvement}.",
        "Should waiver {fee} for old clients.", "Please improve {feature} UI.",
        "Hope to see more ATMs in {location}.", "Suggest to simplify {product} application.",
        "Can you make the font bigger?", "Please bring back the old layout.",
        # Cantonese / Mix
        "強烈建議加返指紋 Login 功能。", "希望 App 可以加個 Dark Mode。",
        "如果你哋分行可以多幾張櫈就好啦。", "建議 {product} 申請手續簡化啲。",
        "個字體太細，老人家睇唔到，建議改大啲。", "可唔可以加返舊版個 layout?"
    ]

    # 4. STATUS_CHECK
    status_checks = [
        "Applied for {product} last week, any news?", "Where is my {item}?",
        "Status of my application?", "Has my card been mailed?",
        "Is my {info} updated?", "Check status of case #1234.",
        "Still waiting for approval.", "Any update on my request?",
        "Did you receive my documents?", "When will I get the {item}?",
        # Cantonese / Mix
        "我上星期交咗表，而家審批成點？", "張卡寄出未呀？等咗好耐。",
        "有無人跟進緊我個 Case？", "我想 Check 下申請進度。",
        "究竟批左未？", "收到文件未？", "幾時先收到張卡？"
    ]

    # 5. FRAUD_REPORT
    frauds = [
        "I lost my {item}, help!", "My card was {fraud_keyword}.",
        "Saw a transaction I didn't make.", "Suspect {fraud_keyword} on my account.",
        "Please freeze my account immediately.", "Received suspicious SMS.",
        "Someone used my card.", "I think I got hacked.",
        "Unrecognized charge of {amount}.", "Report {fraud_keyword}.",
        # Cantonese / Mix
        "我唔見咗張信用卡，想報失！", "收到條 SMS 話我有交易，但我無碌過卡！",
        "懷疑被人盜用資料，快啲幫我凍結戶口。", "我張卡 {fraud_keyword_hk}。",
        "有單交易我唔認數。", "救命，我個戶口俾人 Hack 咗！"
    ]

    # 6. CANCELLATION
    cancellations = [
        "I want to {cancel_action} my card.", "How to {cancel_action} {product}?",
        "Close my account please.", "Don't want to use your service anymore.",
        "Process my cancellation.", "Stop the auto-renewal.",
        "I am switching to another bank.", "Cancel my application.",
        "Form for account closure?", "Terminate service now.",
        # Cantonese / Mix
        "我想 Cut 咗張白金卡佢。", "取消自動轉賬要點做？",
        "我要取消戶口，不想再用你哋服務。", "填邊張表可以 {cancel_action_hk} 服務？",
        "幫我停咗個 {product} 佢。", "我轉會啦，拜拜。"
    ]

    # 7. SALES_LEAD
    sales = [
        "I am {sales_interest} {product}.", "Tell me more about {product}.",
        "Want to open an investment account.", "Buying travel insurance.",
        "Looking for a mortgage plan.", "Any good offers for new clients?",
        "I want to borrow money.", "{sales_interest} personal loan.",
        "How to apply for {product}?", "Connect me to sales team.",
        # Cantonese / Mix
        "我想申請私人貸款，息率幾多？", "有無旅遊保險介紹？",
        "我想開個投資戶口。", "有無 {product} 迎新優惠？",
        "我有興趣買 {product}。", "我想借錢，有咩 plan?"
    ]

    # 8. HUMAN_AGENT
    agents = [
        "Talk to human.", "Connect me to agent.",
        "I want a real person.", "Customer service please.",
        "Chat with staff.", "Transfer to operator.",
        "Don't want AI.", "Live chat support.",
        "Speak to manager.", "Human help.",
        # Cantonese / Mix
        "我想同真人對話。", "叫你經理出黎。", "轉駁去客戶服務員。",
        "我要人聽電話！", "唔想同機械人講野。", "揾個人黎得唔得？"
    ]

    # 9. BUG_REPORT
    bugs = [
        "App crashing when I click {feature}.", "Error 500 on login page.",
        "Buttons not responding.", "Screen freezes at {feature}.",
        "Cannot upload document.", "Biometric login broken.",
        "Page layout is messed up.", "App drains battery.",
        "Cannot type in the field.", "White screen on startup.",
        # Cantonese / Mix
        "網上理財出 Error 503。", "個畫面卡住咗係 Loading 頁面。",
        "收唔到 SMS 驗證碼。", "個制撳唔到。", "每次入去都白畫面。",
        "App 食電食得好快。"
    ]

    # 10. GREETING
    greetings = [
        "Hi", "Hello", "Good morning", "Good evening",
        "Hey", "Anyone there?", "Greetings", "Hi bot",
        "Good afternoon", "Yo",
        # Cantonese / Mix
        "早晨。", "你好。", "喂。", "哈囉。", "有無人係度？"
    ]

    # 11. IRRELEVANT
    irrelevant = [
        "abcd", "123456", "testing", "blah blah",
        "buy bitcoin", "click this link", "nonsense",
        "what is the weather?", "tell me a joke", "spam message",
        # Cantonese / Mix
        "借錢梗要還，咪俾錢中介", "今天天氣如何？", "食咗飯未？", 
        "亂打一通", "測試測試"
    ]

    # Mapping Categories
    categories = [
        (complaints, 0), (questions, 1), (praises, 2), (suggestions, 3),
        (status_checks, 4), (frauds, 5), (cancellations, 6), (sales, 7),
        (agents, 8), (bugs, 9), (greetings, 10), (irrelevant, 11)
    ]

    data = []
    
    # Calculate samples per class
    samples_per_class = int(num_samples / len(categories)) + 50
    print(f"   - Target per class: {samples_per_class}")

    # --- 4. NOISE INJECTION FUNCTION (Strategy D) ---
    def add_typo(text):
        if len(text) < 5 or random.random() > 0.1: # Only 10% chance to typo
            return text
        
        # Simple typo logic: remove a char, or swap chars
        char_list = list(text)
        idx = random.randint(0, len(char_list) - 1)
        
        if random.random() > 0.5:
            # Delete char
            del char_list[idx]
        else:
            # Repeat char (e.g., "hello" -> "helllo")
            char_list.insert(idx, char_list[idx])
            
        return "".join(char_list)

    # --- Generation Function ---
    def create_samples(template_list, label_id, count):
        temp_data = []
        for _ in range(count):
            tmpl = random.choice(template_list)
            
            # Random Prefix/Suffix
            prefix = random.choice(prefixes) if random.random() > 0.4 else ""
            suffix = random.choice(suffixes) if random.random() > 0.4 else ""
            
            # Fill Slots
            text = tmpl
            for key, values in slots.items():
                if "{" + key + "}" in text:
                    text = text.replace("{" + key + "}", random.choice(values))
            
            final_text = f"{prefix}{text}{suffix}"
            
            # Apply Noise (Strategy D)
            final_text = add_typo(final_text)
            
            temp_data.append({"text": final_text, "label": label_id})
        return temp_data

    # Generate Data for each category
    for template_list, label_id in categories:
        print(f"   - Generating Label {label_id}...")
        data.extend(create_samples(template_list, label_id, samples_per_class))

    # Shuffle and Slice
    random.shuffle(data)
    data = data[:num_samples]
    
    return data

if __name__ == "__main__":
    TARGET_COUNT = 50000
    dataset = generate_huge_data(TARGET_COUNT)
    
    output_file = "train_data_topic.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Successfully generated {len(dataset)} high-quality samples!")
    print(f"📁 Saved to: {output_file}")