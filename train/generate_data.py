import json
import random

def generate_multilabel_data(num_samples=50000):
    print(f"🔄 Generating {num_samples} High-Quality Multi-Label Data (Unique & Conflict-Free)...")
    
    # ==========================================
    # 1. Expanded Slots (詞庫擴充)
    # ==========================================
    slots = {
        # --- Tech / Bug ---
        "tech_noun": ["Login page", "FaceID", "Submit button", "Loading screen", "Transfer page", "The App", "Fingerprint sensor", "Home screen", "Payment gateway", "QR Code scanner"],
        "tech_verb": ["crashes", "freezes", "won't load", "is lagging", "force closes", "is stuck", "drains battery", "is not responding", "keeps spinning"],
        "error_code": ["Error 503", "System Failure", "Connection Timeout", "Unknown Error", "Code: 999", "Access Denied", "404 Not Found", "Gateway Timeout"],
        
        # --- Service / Complaint ---
        "staff_role": ["Manager", "Teller", "CS staff", "Hotline agent", "Relationship Manager", "The guy at counter", "Branch staff", "Live chat agent"],
        "rude_behavior": ["very rude", "black face", "hung up on me", "ignored me", "impatient", "unhelpful", "shouted at me", "unprofessional", "lazy"],
        "complaint_topic": ["waiting time", "handling fee", "attitude", "efficiency", "queue", "service quality", "response time"],
        
        # --- Products ---
        "product": ["Credit Card", "Tax Loan", "Mortgage", "Travel Insurance", "P-Loan", "Fixed Deposit", "MPF", "Investment Account", "Business Account", "Savings Account"],
        "item": ["ATM Card", "PIN letter", "Cheque book", "Monthly Statement", "Security Token", "Welcome Gift", "Credit Card", "Advice slip"],
        "feature": ["FPS", "PayMe", "Stock Trading", "Currency Exchange", "Apple Pay", "Google Pay", "Bill Payment", "e-Statement"],
        
        # --- Actions ---
        "action_cancel": ["cancel", "terminate", "cut", "close", "stop", "suspend"],
        "action_apply": ["apply for", "open", "sign up for", "buy", "get"],
        
        # --- HK Cantonese Particles & Slang ---
        "hk_particles": ["啦", "囉", "呀", "喎", "之嘛", "架", "勒", "既", "傑"],
        "hk_complaint": ["搞錯", "不知所謂", "火滾", "離曬譜", "垃圾", "嬲到震", "廢", "頂唔順", "太過分"],
        "hk_verbs": ["搞唔掂", "入唔到", "用唔到", "死左", "壞左", "神神地", "Hang左", "Load唔到"],
        "hk_time": ["幾耐", "幾時", "好耐", "成個鐘", "等到頸都長"],
        
        # --- Misc ---
        "time": ["Saturday", "Sunday", "Public Holiday", "Lunch hour", "After work", "Weekend", "Christmas"],
        "location": ["Mong Kok", "Central", "Kwun Tong", "Shatin", "Tuen Mun", "Causeway Bay", "Tsuen Wan", "Yuen Long"],
        "money": ["$500", "10k", "handling fee", "annual fee", "interest", "charge", "hidden fee"]
    }

    # ==========================================
    # 2. Massive Templates (12 Classes)
    # ==========================================
    templates = {
        # 0. COMPLAINT (投訴)
        0: [
            "Your staff is {rude_behavior}, I want to complain!",
            "I waited for 2 hours at {location}, ridiculous.",
            "Why did you charge me {money}? Refund now!",
            "I am very disappointed with your service.",
            "The {staff_role} was extremely unhelpful.",
            "Never using this bank again, service is bad.",
            "投訴: 你地個 {staff_role} 態度好差。",
            "有無搞錯，收我 {money} 手續費？",
            "打極熱線都無人聽，真係 {hk_complaint}。",
            "你地分行排隊排咁耐，不知所謂。",
            "個 {staff_role} {rude_behavior}，叫你經理出黎。",
            "對於你地既服務，我好失望。",
            "無端端扣錢，離曬譜！",
            "This is the worst banking experience ever.",
            "Your hotline service is totally garbage.",
            "I've been holding the line for ages!",
            "What kind of service is this?",
            "Totally unacceptable behavior from your staff.",
            "真的忍無可忍，我要投訴。",
            "垃圾銀行，亂收費！",
            "個 CS 十問九唔應，想點呀？",
            "火都黎埋，你地做野咁慢。",
            "I demand an explanation for this bad service.",
            "Your efficiency is zero.",
            "System is down again? Are you kidding me?"
        ],

        # 1. QUESTION (查詢)
        1: [
            "What is the opening hour of {location} branch?",
            "How do I activate {feature}?",
            "Is there any annual fee for {product}?",
            "Can I change my address online?",
            "What is the interest rate for {product}?",
            "Do you open on {time}?",
            "May I ask about the {product} details?",
            "請問 {location} 分行 {time} 開唔開？",
            "我想問 {product} 既息口係幾多？",
            "點樣可以用 {feature}？教我。",
            "有無得 waive 左個 {money} 佢？",
            "我想改電話號碼，手續係點？",
            "FPS transfer limit 每日係幾多？",
            "申請 {item} 要帶咩文件？",
            "How can I update my personal info?",
            "Is {feature} available on weekends?",
            "Does {product} come with welcome offers?",
            "Can I increase my credit limit?",
            "Where can I find the ATM near {location}?",
            "What is the swift code?",
            "請問點樣設定外海提款？",
            "我想問定期存款有咩優惠？",
            "忘記密碼可以點樣 reset？",
            "How long does the transfer take?",
            "Can I link my account to WeChat Pay?"
        ],

        # 2. PRAISE (讚賞)
        2: [
            "Excellent service from {staff_role}!",
            "The App is much faster now, good job.",
            "Very helpful staff at {location}.",
            "Quick and efficient, thank you.",
            "I like the new design, very user friendly.",
            "Appreciate the help from the hotline team.",
            "個職員好有禮貌，值得表揚。",
            "多謝 {staff_role} 幫我搞掂，效率好高。",
            "今次體驗好好，俾個讚你地。",
            "個 App 順左好多，正！",
            "很少見到咁好既銀行職員。",
            "Good job, keep it up!",
            "上次去分行，個經理好幫手。",
            "Very professional handling of my case.",
            "Your service is the best in HK.",
            "Thanks for waiving the fee, really appreciate it.",
            "Customer support was top notch.",
            "The new feature is amazing.",
            "好滿意你地既效率。",
            "個 Staff 解釋得好清楚，抵讚。",
            "Thank you for the quick response.",
            "Professional and polite.",
            "Love the new update!",
            "解決問題好快手，多謝。",
            "Great user experience."
        ],

        # 3. SUGGESTION (建議)
        3: [
            "Please add Dark Mode to the App.",
            "Suggest to have more ATMs in {location}.",
            "Can you make the font bigger?",
            "It would be better if you open on Sunday.",
            "Please bring back the old layout.",
            "I suggest improving the login process.",
            "強烈建議加返指紋 Login 功能。",
            "希望 App 可以加個 {feature} 掣。",
            "如果你哋分行可以多幾張櫈就好啦。",
            "建議 {product} 申請手續簡化啲。",
            "個字體太細，老人家睇唔到，建議改大啲。",
            "可唔可以加返舊版個 layout？",
            "你地應該學下其他銀行加個 {feature}。",
            "Hope you can improve the UI soon.",
            "Ideally, the transfer limit should be higher.",
            "Suggest to extend hotline hours.",
            "Better add face recognition.",
            "App 既介面可以再好用啲。",
            "希望加多啲外幣選擇。",
            "建議將個制放係首頁。",
            "Please consider adding Apple Pay support.",
            "Would be nice to have a chat bot.",
            "Make the statement download easier.",
            "個 Search 功能可以再準啲。",
            "Please fix the navigation menu."
        ],

        # 4. STATUS_CHECK (進度)
        4: [
            "Where is my {item}? I haven't received it.",
            "I applied for {product} last week, any update?",
            "Check status of my application.",
            "Is my card approved yet?",
            "When will I get the welcome gift?",
            "Has my cheque been cleared?",
            "我上星期交咗表，而家審批成點？",
            "張卡寄出未呀？等左 {hk_time}。",
            "有無人跟進緊我個 Case？",
            "我想 Check 下申請進度。",
            "究竟批左未？收到文件未？",
            "幾時先收到張 {item}？",
            "My application is still pending, why?",
            "Any news on my loan approval?",
            "Check refund status.",
            "Track my card delivery.",
            "提交左資料好耐，無聲氣。",
            "我想知批核結果。",
            "Status update please.",
            "Have you processed my request?",
            "Is the money transferred?",
            "確認信寄出未？",
            "My case number is 12345, check status.",
            "Still waiting for approval.",
            "Did you receive my documents?"
        ],

        # 5. FRAUD_REPORT (詐騙/緊急)
        5: [
            "I lost my credit card, please block it!",
            "There is a transaction I didn't make.",
            "I think my account is hacked.",
            "Received a suspicious SMS code.",
            "Unknown charge of {money} on my card.",
            "I suspect a fraudulent activity.",
            "我唔見咗張信用卡，想報失！",
            "收到條 SMS 話我有交易，但我無碌過卡！",
            "懷疑被人盜用資料，快啲幫我凍結戶口。",
            "有單交易我唔認數。",
            "救命，我個戶口俾人 Hack 咗！",
            "我懷疑中左詐騙。",
            "Please freeze my account immediately.",
            "Unauthorized withdrawal detected.",
            "Someone used my card in Japan!",
            "I did not authorize this payment.",
            "Report lost card.",
            "Suspicious login alert.",
            "張卡俾人盜用左呀！",
            "快啲停左我張卡佢。",
            "有無人幫手？我唔見左銀包。",
            "Emergency: Account hacked.",
            "Fraud alert!",
            "Unrecognized transaction found.",
            "Help, money stolen!"
        ],

        # 6. CANCELLATION (取消)
        6: [
            "I want to close my account.",
            "How to {action_cancel} my credit card?",
            "Don't want to use your service anymore.",
            "Stop the auto-renewal please.",
            "I am switching to another bank.",
            "Please proceed with the cancellation.",
            "我想 Cut 咗張白金卡佢。",
            "取消自動轉賬要點做？",
            "我要取消戶口，不想再用你哋服務。",
            "填邊張表可以 {action_cancel} 服務？",
            "幫我停咗個 {product} 佢，唔該。",
            "我轉會啦，拜拜。",
            "Terminate my subscription now.",
            "I want to opt-out from this service.",
            "Cut card procedure.",
            "Cancel my application.",
            "Close savings account.",
            "我想退保，點做？",
            "唔想再用你地張卡。",
            "Stop the monthly charge.",
            "Cancel everything.",
            "Form for account closure?",
            "Process my termination.",
            "I found a better bank, closing this one.",
            "Can I cancel online?"
        ],

        # 7. SALES_LEAD (銷售)
        7: [
            "I want to {action_apply} a {product}.",
            "Tell me more about your investment plans.",
            "I am looking to borrow some money.",
            "Do you have any promotion for new clients?",
            "I want to buy travel insurance.",
            "Please ask a sales staff to call me.",
            "我想申請私人貸款，息率幾多？",
            "有無旅遊保險介紹？",
            "我想開個投資戶口。",
            "有無 {product} 迎新優惠？",
            "我有興趣買 {product}，揾人聯絡我。",
            "想問下做按揭既詳情。",
            "Interested in opening a business account.",
            "Any cash rebate for new credit card?",
            "Looking for a tax loan.",
            "I want to start investing.",
            "What mortgage plan do you offer?",
            "Apply for Visa card.",
            "我想借錢，有無平息？",
            "對你地個基金有興趣。",
            "Open new account.",
            "Want to sign up for MPF.",
            "Show me your best offer.",
            "I need a loan.",
            "Buying forex."
        ],

        # 8. HUMAN_AGENT (真人)
        8: [
            "I want to talk to a human.",
            "Connect me to an agent please.",
            "Don't want to chat with bot.",
            "Is there a real person available?",
            "Transfer to operator.",
            "Can I speak to the manager?",
            "我想同真人對話。",
            "叫你經理出黎。",
            "轉駁去客戶服務員。",
            "我要人聽電話！唔想同機械人講野。",
            "揾個人黎得唔得？",
            "人工客服係邊？",
            "Talk to staff.",
            "Live chat with agent please.",
            "Human support needed.",
            "Customer service representative please.",
            "Bot is useless, give me a human.",
            "我想找客服。",
            "接駁去熱線同事。",
            "Need real help.",
            "Speak to someone.",
            "Chat with operator.",
            "有無真人呀？",
            "轉台去人手。",
            "Staff please."
        ],

        # 9. BUG_REPORT (故障)
        9: [
            "The {tech_noun} {tech_verb} every time.",
            "I see {error_code} when I login.",
            "Cannot click the submit button.",
            "FaceID is not working.",
            "App closes immediately after opening.",
            "System is down again.",
            "網上理財出 Error 503。",
            "個畫面卡住咗係 Loading 頁面。",
            "收唔到 SMS 驗證碼。",
            "個 App 又 {tech_verb}，搞錯！",
            "每次入去都白畫面，{hk_verbs}。",
            "個指紋 Login 用唔到。",
            "Cannot load the transaction history.",
            "The screen freezes when I transfer money.",
            "Bug in the latest update.",
            "App keeps crashing on iPhone.",
            "Login failed: System Error.",
            "Unable to connect to server.",
            "個制撳極都無反應。",
            "閃退問題嚴重。",
            "FPS transfer failed.",
            "App is very laggy.",
            "Technical issue with the app.",
            "Blank screen error.",
            "Something wrong with the system."
        ],

        # 10. GREETING (打招呼)
        10: [
            "Hi", "Hello", "Good morning", "Good afternoon", "Good evening",
            "Hey there", "Greetings", "Hi bot", "Yo",
            "早晨。", "你好。", "喂。", "哈囉。", "有無人係度？",
            "Hi hi", "Hello testing", "Good day", "Hey", "Hi there",
            "Excuse me", "Anyone?", "Start chat", "Begin", "Hola"
        ],

        # 11. IRRELEVANT (無關)
        11: [
            "abcd", "123456", "testing", "blah blah",
            "buy bitcoin", "click this link", "nonsense",
            "what is the weather?", "tell me a joke",
            "do you like pizza?", "spam message",
            "借錢梗要還，咪俾錢中介", "今天天氣如何？", 
            "食咗飯未？", "亂打一通", "測試測試",
            "asdfghjkl", "Testing 123", "Wrong number",
            "How are you?", "Are you AI?", "Sing a song",
            "Promotion code 123", "Crypto scam", "Click here"
        ]
    }

    # ==========================================
    # 3. 衝突過濾器 (Conflict Filter)
    # ==========================================
    conflicts = {
        0: [2, 10], 2: [0, 5, 9, 6], 5: [2, 10, 3], 6: [2], 9: [2], 10: [0, 5, 6], 11: [0,1,2,3,4,5,6,7,8,9,10]
    }

    # ==========================================
    # 4. 連接詞 (Connectors)
    # ==========================================
    connectors = [". Also, ", ". Plus, ", " and ", " & ", "; ", "，同埋 ", "。另外 ", "，還有 ", "，仲有 ", "。 ", " ", " ", "\n"]
    prefixes = ["Hi, ", "Urgent: ", "喂, ", "唔該, ", "To Manager: ", "", "", ""]
    suffixes = [".", "!", "...", " 唔該。", " thx.", "", ""]

    data = []
    seen_texts = set() # ✅ ADDED: Unique Check

    def fill_slots(text):
        for k, v in slots.items():
            if "{"+k+"}" in text: text = text.replace("{"+k+"}", random.choice(v))
        return text

    # ==========================================
    # 5. 生成 Loop (Logic)
    # ==========================================
    count = 0
    max_attempts = num_samples * 5
    attempts = 0

    while count < num_samples and attempts < max_attempts:
        attempts += 1
        label_vec = [0.0] * 12
        final_text = ""
        
        # [A] 50% 機率：雙重意圖 (Double Intent)
        if random.random() < 0.5:
            idx1 = random.randint(0, 10)
            valid_seconds = [i for i in range(11) if i != idx1 and i not in conflicts.get(idx1, []) and idx1 not in conflicts.get(i, [])]
            
            if not valid_seconds:
                sent = fill_slots(random.choice(templates[idx1]))
                final_text = f"{random.choice(prefixes)}{sent}{random.choice(suffixes)}"
                label_vec[idx1] = 1.0
            else:
                idx2 = random.choice(valid_seconds)
                sent1 = fill_slots(random.choice(templates[idx1]))
                sent2 = fill_slots(random.choice(templates[idx2]))
                conn = random.choice(connectors)
                final_text = f"{random.choice(prefixes)}{sent1}{conn}{sent2}{random.choice(suffixes)}"
                label_vec[idx1] = 1.0
                label_vec[idx2] = 1.0

        # [B] 50% 機率：單一意圖
        else:
            idx = random.randint(0, 11)
            sent = fill_slots(random.choice(templates[idx]))
            final_text = f"{random.choice(prefixes)}{sent}{random.choice(suffixes)}"
            label_vec[idx] = 1.0

        # ✅ ADDED: Unique Check logic
        if final_text not in seen_texts:
            seen_texts.add(final_text)
            data.append({"text": final_text, "label": label_vec})
            count += 1
            if count % 10000 == 0:
                print(f"   ... Generated {count}/{num_samples}")

    return data

if __name__ == "__main__":
    dataset = generate_multilabel_data(50000)
    with open("train_data_multilabel.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Generated {len(dataset)} Conflict-Free Multi-Label samples!")
    print(json.dumps([d for d in dataset if sum(d['label']) > 1][:3], ensure_ascii=False, indent=2))