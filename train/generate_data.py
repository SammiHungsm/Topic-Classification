import json
import random

def generate_synthetic_data(num_samples=1000):
    print(f"🔄 正在生成 {num_samples} 條模擬數據...")
    
    # 模板庫
    complaints = [
        "你哋個 App {issue}，垃圾！", "打極電話都{status}，火都黎埋。", 
        "Login 唔到，顯示 {error}，搞錯呀？", "分行職員{behavior}，完全唔想幫手。",
        "無端端收多我{fee}，回水！", "對你哋服務好{feeling}，以後唔用。",
        "等咗好耐都未有{item}，效率極低！", "用咗咁耐從來未試過咁{feeling}！"
    ]
    inquiries = [
        "請問分行{time}幾點開門？", "申請{product}要幾耐批？",
        "我想更改{info}，手續係點？", "有無得豁免{fee}？",
        "請問點樣啟動{feature}功能？", "我想問下{product}既利息係幾多？",
        "如果我唔見咗{item}，應該點做？", "有無{product}既最新優惠？"
    ]
    praises = [
        "{staff}服務好好，好細心，值得表揚。", "多謝你哋幫我解決{issue}，效率好高。",
        "上次個客服好{trait}，解釋得好清楚。", "Excellent {service}, very helpful!",
        "個 App 更新左之後{improvement}，正！", "十分滿意你哋既{service}。",
        "感受到你哋既專業，{staff}好幫手。", "{staff}好有禮貌，令我好開心。"
    ]

    # 填充詞
    slots = {
        "issue": ["狂彈 App", "Hang 機", "Load 唔到", "白畫面", "轉唔到數"],
        "status": ["無人聽", "長期繁忙", "Cut 線", "飛留言"],
        "error": ["Error 404", "System Failure", "Timeout", "Connection Error"],
        "behavior": ["態度極差", "黑口黑面", "十問九唔應", "掛我電話"],
        "fee": ["手續費", "年費", "過期罰款", "透支利息"],
        "feeling": ["失望", "憤怒", "不滿", "無奈"],
        "item": ["信用卡", "密碼信", "提款卡", "支票簿"],
        "time": ["星期六", "公眾假期", "閒日", "年初一"],
        "product": ["信用卡", "私人貸款", "按揭", "保險"],
        "info": ["通訊地址", "電話號碼", "電郵", "出生日期"],
        "feature": ["海外提款", "轉數快", "網上理財", "股票買賣"],
        "staff": ["陳經理", "櫃位職員", "熱線同事", "分行經理", "客戶主任"],
        "trait": ["有禮貌", "有耐性", "專業", "細心"],
        "service": ["service", "support", "服務", "安排"],
        "improvement": ["順左好多", "快左", "好用左", "穩定左"]
    }

    data = []
    
    # 確保每個類別平均分佈
    samples_per_class = num_samples // 3
    
    # 生成數據
    for _ in range(samples_per_class):
        # Complaint (Label 0)
        tmpl = random.choice(complaints)
        text = tmpl.format(**{k: random.choice(v) for k, v in slots.items() if "{" + k + "}" in tmpl})
        data.append({"text": text, "label": 0})
        
        # Inquiry (Label 1)
        tmpl = random.choice(inquiries)
        text = tmpl.format(**{k: random.choice(v) for k, v in slots.items() if "{" + k + "}" in tmpl})
        data.append({"text": text, "label": 1})
        
        # Praise (Label 2)
        tmpl = random.choice(praises)
        text = tmpl.format(**{k: random.choice(v) for k, v in slots.items() if "{" + k + "}" in tmpl})
        data.append({"text": text, "label": 2})

    random.shuffle(data)
    return data

if __name__ == "__main__":
    # 生成 1000 條數據
    raw_data = generate_synthetic_data(1000)
    
    # 儲存為 JSON
    output_file = "train_data_topic.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 數據生成完成！已儲存至 {output_file}")