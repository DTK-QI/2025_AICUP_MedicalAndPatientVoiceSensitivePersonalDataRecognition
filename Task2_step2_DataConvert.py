import json

# 讀取 task1_answer.txt
task1_path = r"XXXX"
task2_path = r"XXX"
output_path = r"XXXX"

# 1. 讀取 task1，建立 id -> text
id2text = {}
with open(task1_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t', 1)
        if len(parts) < 2:
            continue
        id2text[parts[0]] = parts[1]

# 2. 讀取 task2，建立 id -> entities
id2entities = {}
with open(task2_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        id_, label, start, end, text = parts
        entity = {
            "category": label,
            "text": text
        }
        id2entities.setdefault(id_, []).append(entity)

# 3. 合併
merged = []
for id_, text in id2text.items():
    merged.append({
        "id": id_,
        "text": text,
        "entities": id2entities.get(id_, [])
    })

# 4. 輸出 json
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"已輸出 {output_path}")