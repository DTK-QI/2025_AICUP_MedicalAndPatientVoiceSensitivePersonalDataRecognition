import json
import os
import re
import requests
from dotenv import load_dotenv
import string
from typing import List, Dict, Any, Optional, Tuple
import difflib
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 在文件頂部添加導入
import opencc

# 在文件頂部添加轉換器初始化
try:
    converter_t2s = opencc.OpenCC('t2s')  # traditional to simplified
except Exception as e:
    print(f"Warning: opencc initialization failed: {e}")
    converter_t2s = None

# Load environment variables (for API key)
load_dotenv()
GROK_API_KEY = os.getenv('GROK_API_KEY')

# --- Configuration ---
MODEL_PATH = "./task2/qwen3_ner_finetuned_9_LORA_allDATA"
TIME_STEP_JSON_PATH = "result/task1/v15_Belle_large_v3_train_凍結編碼前20層參數/val_time_step.json"
OUTPUT_TSV_PATH = "result/task2/v20_8/task2_answer.txt" # Changed output path to task2
ENTITY_DICT_PATH = "e:\\Yuan\\SideProject\\WhisperNER\\task2\\entities.json" # Path to your entity dictionary
LLM_MODEL = "grok-3" # Grok model - use the appropriate one (e.g., llama3-70b-8192)
LLM_API_URL = 'https://api.x.ai/v1/chat/completions' #  API URL
# LLM_MODEL = "gemma3:27B" # Grok model - use the appropriate one (e.g., llama3-70b-8192)
# LLM_API_URL = 'http://localhost:11434/v1/chat/completions' #  API URL
TAG_List = [
  "PATIENT",
  "DOCTOR",
  "USERNAME",
  "FAMILYNAME",
  "PERSONALNAME",
  "PROFESSION",
  "ROOM",
  "DEPARTMENT",
  "HOSPITAL",
  "ORGANIZATION",
  "STREET",
  "CITY",
  "DISTRICT",
  "COUNTY",
  "STATE",
  "COUNTRY",
  "ZIP",
  "LOCATION-OTHER",
  "AGE",
  "DATE",
  "TIME",
  "DURATION",
  "SET",
  "PHONE",
  "FAX",
  "EMAIL",
  "URL",
  "IPADDRESS",
  "SOCIAL_SECURITY_NUMBER",
  "MEDICAL_RECORD_NUMBER",
  "HEALTH_PLAN_NUMBER",
  "ACCOUNT_NUMBER",
  "LICENSE_NUMBER",
  "VEHICLE_ID",
  "DEVICE_ID",
  "BIOMETRIC_ID",
  "ID_NUMBER",
  "OTHER"
]

# --- Filter Rules ---
# Define categories and the specific entity texts to remove within those categories
# Uses case-insensitive matching for the filter text.
FILTER_RULES: Dict[str, List[str]] = {
    "FAMILYNAME": [
        "DAD", "MOM", "HE", "IT", "SHE", "YOU", 
        "my mother", "my father", "my dad", "my mom", "my brother", "my sister","Pope",
        ],
    "DOCTOR": [
        "DAD", "MOM", "HE", "IT", "SHE", "YOU", 
        "my mother", "my father", "my dad", "my mom", "my brother", "my sister","Pope",
        ],
    "PATIENT": [
        "DAD", "MOM", "HE", "IT", "SHE", "YOU", 
        "my mother", "my father", "my dad", "my mom", "my brother", "my sister","Pope",
        ],
    "PERSONALNAME": [
        "DAD", "MOM", "HE", "IT", "SHE", "YOU", 
        "my mother", "my father", "my dad", "my mom", "my brother", "my sister","Pope",
        ],
    "PERSONALNAME": [
        "DAD", "MOM", "HE", "IT", "SHE", "YOU",
        "my mother", "my father", "my dad", "my mom", "my brother", "my sister","Pope",
        ],
    
    "PROFESSION": [
        "college acapella group","社工师","社工","doctor of surgery","Associate Professor"
        ],
    "DEPARTMENT": [
        "department", "Hunter Area", "Parks 8", "Parkes‑9 East Department","West"
        ],
    "HOSPITAL": ["All-Mail Dhistory Health", "Vincent", "Pathwest", "Francic","Reno Clinic"], # Example: Remove generic titles if needed
    "ORGANIZATION": ["Corrigent", "Fonsek Spiro","cure for life foundation neuro-oncology laboratory"],
    "DURATION":["that time"],
    "HOSPITAL":["hospital"]
    
}


# --- Load Entity Dictionary ---
entity_dictionary: List[Dict[str, List[str]]] = []
try:
    with open(ENTITY_DICT_PATH, 'r', encoding='utf-8') as f:
        entity_dictionary = json.load(f)
    print(f"Successfully loaded entity dictionary from: {ENTITY_DICT_PATH}")
except FileNotFoundError:
    print(f"Warning: Entity dictionary file not found at {ENTITY_DICT_PATH}. Dictionary search will be skipped.")
except json.JSONDecodeError as e:
    print(f"Warning: Could not decode JSON from {ENTITY_DICT_PATH}: {e}. Dictionary search will be skipped.")
except Exception as e:
    print(f"Warning: An unexpected error occurred while loading the entity dictionary: {e}. Dictionary search will be skipped.")


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for matching."""
    if not text:
        return ""
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def reconstruct_text_from_words(words: List[Dict[str, Any]]) -> str:
    """Reconstruct the full text from the word list."""
    # Filter out potential null entries or entries without 'word'
    return " ".join(word_data.get("word", "") for word_data in words if word_data and "word" in word_data).strip()

def reconstruct_text_from_words_zh(words: List[Dict[str, Any]]) -> str:
    """Reconstruct the full text from the word list."""
    # Filter out potential null entries or entries without 'word'
    return "".join(word_data.get("word", "") for word_data in words if word_data and "word" in word_data).strip()
# def fuzzy_find_entity_in_transcript(entity_text_norm, transcript_words_norm, cutoff=0.85):
#     """
#     Fuzzy match entity_text_norm (str) to a sequence in transcript_words_norm (list of str).
#     Returns (start_idx, end_idx) if found, else None.
#     """
#     entity_words = entity_text_norm.split()
#     n = len(entity_words)
#     for i in range(len(transcript_words_norm) - n + 1):
#         window = transcript_words_norm[i:i+n]
#         # Use difflib.SequenceMatcher to compare joined strings
#         ratio = difflib.SequenceMatcher(None, " ".join(window), " ".join(entity_words)).ratio()
#         if ratio >= cutoff:
#             return i, i+n-1
#     return None

def fuzzy_find_entity_in_transcript(entity_text_norm, transcript_words_norm, is_chinese=False, cutoff=0.85):
    """
    改進的模糊匹配：支援不同長度的窗口，並針對中文進行特殊處理
    """
    if not entity_text_norm or not transcript_words_norm:
        return None
    
    best_match = None
    best_ratio = 0
    
    if is_chinese:
        # 中文處理：直接比較字符串，不依賴空格分詞
        # 將 transcript_words 合併成連續字符串
        transcript_text = "".join(transcript_words_norm)
        
        try:
            entity_text_norm = converter_t2s.convert(entity_text_norm)
        except Exception as e:
            print(f"  Warning: Failed to convert text to simplified Chinese: {e}")
            entity_text_norm = entity_text_norm
        
        # 使用滑動窗口在字符級別進行匹配
        entity_len = len(entity_text_norm)
        
        for i in range(len(transcript_text) - entity_len + 1):
            substring = transcript_text[i:i + entity_len]
            ratio = difflib.SequenceMatcher(None, substring, entity_text_norm).ratio()
            
            if ratio >= cutoff and ratio > best_ratio:
                best_ratio = ratio
                
                # 需要將字符位置轉換回詞位置
                char_start = i
                char_end = i + entity_len
                
                # 找到對應的詞邊界
                word_start_idx = 0
                word_end_idx = 0
                char_pos = 0
                
                for word_idx, word in enumerate(transcript_words_norm):
                    word_start_char = char_pos
                    word_end_char = char_pos + len(word)
                    
                    if char_start >= word_start_char and char_start < word_end_char:
                        word_start_idx = word_idx
                    if char_end > word_start_char and char_end <= word_end_char:
                        word_end_idx = word_idx
                        break
                    
                    char_pos += len(word)
                
                best_match = (word_start_idx, word_end_idx)
                
                # 如果找到很高的匹配度，可以提早結束
                if ratio > 0.95:
                    break
    else:
        # 英文處理：使用原有的邏輯
        entity_words = entity_text_norm.split()
        entity_text_joined = " ".join(entity_words)
        n_entity_words = len(entity_words)
        
        # 嘗試不同的窗口大小
        min_size = max(1, n_entity_words // 2)
        max_size = min(len(transcript_words_norm), n_entity_words * 2)
        
        for window_size in range(min_size, max_size + 1):
            for i in range(len(transcript_words_norm) - window_size + 1):
                window = transcript_words_norm[i:i+window_size]
                window_text = " ".join(window)
                
                # 計算相似度
                ratio = difflib.SequenceMatcher(None, window_text, entity_text_joined).ratio()
                
                if ratio >= cutoff and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (i, i + window_size - 1)
                    
                    # 如果找到很高的匹配度，可以提早結束
                    if ratio > 0.95:
                        break
            
            # 如果已經找到很好的匹配，可以不用嘗試更大的窗口
            if best_ratio > 0.95:
                break
    
    return best_match

def get_entities_from_llm(text: str, tokenizer, model, device) -> Optional[List[Dict[str, str]]]:
    if not text:
        return []
    
    # Use the same prompt format as in training
    prompt = f"""Extract sensitive health information entities from the following medical dialogue.

        Target Entity Categories:
        - PATIENT: Patient names
        - DOCTOR: Doctor names   
        - FAMILYNAME: Family member names
        - PERSONALNAME: Personal names (not patient/doctor)
        - PROFESSION: Job titles or professions
        - DEPARTMENT: Hospital departments
        - HOSPITAL: Hospital or clinic names
        - ORGANIZATION: Organizations or institutions
        - STREET/CITY/DISTRICT/COUNTY/STATE/COUNTRY/ZIP: Location information
        - LOCATION-OTHER: Other location references
        - AGE: Age information
        - DATE/TIME/DURATION/SET: Temporal information
        - MEDICAL_RECORD_NUMBER/ID_NUMBER: Identification numbers

        Instructions:
        1. Identify ALL entities that match the categories above
        2. Extract exact text spans as they appear
        3. Return ONLY entities present in the text
        4. Use exact category names from the list above

        Output format:
        {{"entities": [{{"text": "<entity_text>", "category": "<entity_category>"}}]}}

        Example:
        Input: "Hello Dr. Chen, I am 65-year-old patient John Wang, here for a follow-up at Zhongshan Hospital Cardiology Department today."
        Output: {{"entities": [{{"text": "Dr. Chen", "category": "PROFESSION"}}, {{"text": "65", "category": "AGE"}}, {{"text": "John Wang", "category": "PATIENT"}}, {{"text": "Zhongshan Hospital", "category": "HOSPITAL"}}, {{"text": "Cardiology Department", "category": "DEPARTMENT"}}]}}

        Input Text: "{text}"
        Output: """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=3072,
            do_sample=False,
            # temperature=0.2,
            # top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
            
        )
    
    # Decode only the generated part (excluding the input prompt)
    generated_tokens = outputs[0][len(inputs.input_ids[0]):]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # print(f"Generated result: {result[:200]}...")  # Print first 200 characters for debugging
    
    # Try to extract JSON from the result
    json_match = re.search(r'\{[\s\S]*?\}', result)
    if json_match:
        try:
            parsed_json = json.loads(result)
            if "entities" in parsed_json and isinstance(parsed_json["entities"], list):
                valid_entities = []
                for entity in parsed_json["entities"]:
                    if isinstance(entity, dict) and "text" in entity and "category" in entity:
                        category = entity["category"].upper()
                        if category not in TAG_List:
                            # Map common variations or skip unknown categories
                            continue
                        valid_entities.append({"text": str(entity["text"]), "category": str(category)})
                return valid_entities
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw result: {result}")
            return []
        except Exception as e:
            print(f"Other error: {e}")
            return []
    
    print(f"No valid JSON found in result: {result}")
    return []

def find_entities_from_dictionary(text: str, dictionary: List[Dict[str, List[str]]], is_chinese: bool = False) -> List[Dict[str, str]]:
    """
    Find entities in the text based on a predefined dictionary.
    Uses case-insensitive search but preserves original casing from the text.
    """
    found_entities = []
    if not dictionary or not text:
        return found_entities

    # Create a lowercased version of the text for case-insensitive matching
    text_lower = text.lower()

    for category_entry in dictionary:
        for category, entity_list in category_entry.items():
            if category not in TAG_List: # Ensure category is valid
                continue
            for entity_text_dict in entity_list:
                entity_text_dict_lower = entity_text_dict.lower()
                
                try:
                    if is_chinese:
                        # 中文處理：不使用詞邊界，直接搜索子字符串
                        pattern = re.escape(entity_text_dict_lower)
                        
                        for match in re.finditer(pattern, text_lower):
                            start, end = match.span()
                            # 調整 start 和 end 以去除邊界字符
                            matched_full = text_lower[start:end]
                            
                            # 找到實際實體在匹配中的位置
                            entity_start_in_match = matched_full.find(entity_text_dict_lower)
                            if entity_start_in_match >= 0:
                                actual_start = start + entity_start_in_match
                                actual_end = actual_start + len(entity_text_dict_lower)
                                original_casing_text = text[actual_start:actual_end]
                                found_entities.append({"text": original_casing_text, "category": category})
                    else:
                        # 英文處理：使用原有邏輯
                        pattern = r'(?:^|\s)' + re.escape(entity_text_dict_lower) + r'(?:\.|,|$|\s)'
                        for match in re.finditer(pattern, text_lower):
                            start, end = match.span()
                            matched_full = text_lower[start:end]
                            
                            # 找到實際實體在匹配中的位置
                            entity_start_in_match = matched_full.find(entity_text_dict_lower)
                            if entity_start_in_match >= 0:
                                actual_start = start + entity_start_in_match
                                actual_end = actual_start + len(entity_text_dict_lower)
                                original_casing_text = text[actual_start:actual_end]
                                found_entities.append({"text": original_casing_text, "category": category})
                                
                except re.error as e:
                    print(f"Warning: Regex error for entity '{entity_text_dict}' in category '{category}': {e}")
                    # Fallback to simple substring search
                    start_index = 0
                    while start_index < len(text_lower):
                        pos = text_lower.find(entity_text_dict_lower, start_index)
                        if pos == -1:
                            break
                        
                        if is_chinese:
                            # 中文：簡單的子字符串匹配
                            original_casing_text = text[pos:pos + len(entity_text_dict)]
                            found_entities.append({"text": original_casing_text, "category": category})
                        else:
                            # 英文：檢查詞邊界
                            precedes_ok = (pos == 0) or (not text_lower[pos-1].isalnum())
                            follows_ok = (pos + len(entity_text_dict_lower) == len(text_lower)) or \
                                         (not text_lower[pos + len(entity_text_dict_lower)].isalnum())

                            if precedes_ok and follows_ok:
                                original_casing_text = text[pos:pos + len(entity_text_dict)]
                                found_entities.append({"text": original_casing_text, "category": category})
                        
                        start_index = pos + 1


    return found_entities

def get_pattern_match_category(text: str) -> list:
    """
    根據 PATTERNS_UPDATE 判斷 text 屬於哪一種時間/日期/頻率型別，回傳所有符合的類別名稱list
    """
    PATTERNS_UPDATE = {
        "ID_NUMBER": [
            re.compile(r"(?:^|\s)[0-9]{2}-[0-9]{7}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)[0-9]{2}[A-Za-z][0-9]{5}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)[0-9]{2}[A-Za-z][0-9]{6}[A-Z](?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)[0-9]{2}[A-Za-z][0-9]{2}-[0-9]{3}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)(?:lab\s+number)\s+[0-9]{3}-?[0-9]{5}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)(?:lab\s+number)\s+[0-9]{2}-?[0-9]{5}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)episode\s+number\s+[0-9]{3}-?[0-9]{6}-?[A-Za-z](?:\.|,|$|\s)", re.IGNORECASE)
            
        ],
        "MEDICAL_RECORD_NUMBER": [
            re.compile(r"(?:^|\s)[0-9]{6}\.[A-Z]{3}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)[0-9]{7}\.[A-Z]{3}(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)\d(?:-?\d){5,}\.[A-Z]{3}(?:\.|,|$|\s)", re.IGNORECASE),
        ],        
        "ZIP": [
            re.compile(r"(?:^|\s)ZIP\s*code\s*([0-9]{4})(?:\.|,|$|\s)", re.IGNORECASE), # 匹配 "ZIP code dddd"
            re.compile(r"(?:^|\s)ZIP\s*code\sof\s*([0-9]{4})(?:\.|,|$|\s)", re.IGNORECASE), # 匹配 "ZIP code dddd"
            re.compile(r"(?:^|\s)postal\s*code\s*([0-9]{4})(?:\.|,|$|\s)", re.IGNORECASE), # 匹配 "ZIP code dddd"
        ],
        "DOCTOR": [
            # 匹配 "Dr. " 後面的姓名部分，例如 "Dr. FC" 中的 "FC"，或 "Dr. John Smith" 中的 "John Smith"
            # ([A-Za-z][A-Za-z\s\.'-]*[A-Za-z0-9]|[A-Z]{1,5}) 捕獲姓名部分：
            #   [A-Za-z]                  - 名字以字母開頭
            #   [A-Za-z\s\.'-]*           - 名字中間可以包含字母、空格、點、撇號、連字符
            #   [A-Za-z0-9]               - 名字以字母或數字結尾 (處理像 Dr. J. R. Smith Jr. 的情況)
            #   或者 ([A-Z]{1,5})         - 捕獲1到5個大寫字母的縮寫 (例如 FC, J.D.)
            re.compile(r"(?:^|\s)dr\.\s+([A-Za-z]{1,10})(\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)drs\.\s+([A-Za-z]{1,10})(\.|,|$|\s)", re.IGNORECASE)
        
        ],
        "DATE": [
            # 匹配例如 "May 21, 2012", "Jan 1, 1999", "December 31, 2023"
            re.compile(
                r"(?:^|\s)(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}(?:\.|,|$|\s)",
                re.IGNORECASE
            ),
            re.compile(r"(?:^|\s)(?:first|second)\s+semester(?:\.|,|$|\s)", re.IGNORECASE),
            re.compile(r"(?:^|\s)(0?[1-9]|[12]\d|3[01])-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}(?:\.|,|$|\s)", re.IGNORECASE),
        ],
        "TIME": [
            # 匹配例如 "09:21 AM", "9.21 AM", "01:00 PM", "12:30"
            # (0?[1-9]|1[0-2]) - 小時 (1-12), 0 可選
            # [:\.]             - 分隔符 : 或 .
            # [0-5]\d           - 分鐘 (00-59)
            # (?:\s*(?:AM|PM))? - 可選的 AM/PM (前面可有空格)
            re.compile(r"(?:^|\s)(?:0[1-9]|1[0-2]):[0-5][0-9]\s*(?:[Aa]\.?[Mm]\.?|[Pp]\.?[Mm]\.?)(?:\.|,|$|\s)", re.IGNORECASE),
            # 您可以在此處為 TIME 類別添加其他時間格式的正規表示式
            # 例如24小時制: re.compile(r"\b(?:[01]\d|2[0-3])[:\.][0-5]\d\b")
            re.compile(r"(?:^|\s)(?:[1-9]|1[0-2])\s*(?:[Aa]\.?[Mm]\.?|[Pp]\.?[Mm]\.?)(?:\.|,|$|\s)", re.IGNORECASE), # 匹配 "9 AM", "12 PM" 等
            re.compile(r"(?:^|\s)(?:[1-9]|1[0-2])\s+(?:[Aa]\.?[Mm]\.?|[Pp]\.?[Mm]\.?)(?:\.|,|$|\s)", re.IGNORECASE), # 匹配 "9 AM", "12 PM" 等
            re.compile(r"(?:^|\s)([1-9]|1[0-2])\s+([1-9]|1[0-2])\s*(?:a\.m\.|p\.m\.)(?:\.|,|$|\s)", re.IGNORECASE), # 匹配 "09:21", "12:30" 等
            re.compile(r"(?:^|\s)((?:[01]?[0-9]|2[0-3]):[0-5][0-9])(?:\.|,|$|\s)", re.IGNORECASE),
        ],
        "DURATION": [
            # 匹配例如 "20 years", "3 months", "1 day", "2 weeks", "5 hours", "30 minutes", "10 seconds"
            # \d+             - 一個或多個數字
            # \s+             - 一個或多個空格
            # (?:...)         - 非捕獲組
            # years?          - "year" 或 "years"
            # months?         - "month" 或 "months"
            # weeks?          - "week" 或 "weeks"
            # days?           - "day" 或 "days"
            # hours?          - "hour" 或 "hours"
            # minutes?        - "minute" 或 "minutes"
            # seconds?        - "second" 或 "seconds"
            # |               - 或
            # \b              - 詞邊界
            re.compile(
                r"(?:^|\s)(?:(?:zero|one|two|three|four|five)|(?:[0-9]|[1-3][0-9]|4[0-4]))\s+(?:years|months|weeks|days|hours|minutes|seconds)(?:\.|,|$|\s)",
                re.IGNORECASE
            ),
            # 您可以根據需要添加更多時間單位或模式
        ],
        "SET":[
            re.compile(
                r"(?:^|\s)(?:one|two|three|four|five|six|seven|\d+)\s+days\s+a\s+week(?:\.|,|$|\s)",
                re.IGNORECASE
            ),
            re.compile(r"(?:^|\s)(\d+)\s+days\s+a\s+week(?:\.|,|$|\s)", re.IGNORECASE)
            
        ],
        "AGE":[
            re.compile(r"(?:^|\s)(?P<age>\d{1,3})(?:[\s-]+year(?:s)?[\s-]+old)(?:\.|,|$|\s)", re.IGNORECASE)
        ],
        "PHONE":[
            re.compile(r"(?:^|\s)(?<=\bcontact number is )\d{8}(?=\b)|(?<=\bcontact number )(\d{8})(?=\b)(?:\.|,|$|\s)", re.IGNORECASE), # 匹配電話號碼
        ],
        "PERSONALNAME": [# 中文姓氏 + 職位稱謂模式，只捕獲並提取姓氏
            re.compile(r"(刘)(?:老师|委员|主席|部長|部长|局長|局长|科長|科长|主任|經理|经理|總監|总监|處長|处长|廳長|厅长|署長|署长|司長|司长|院長|院长|校長|校长|組長|组长|課長|课长|股長|股长|副手|助理|秘書|秘书|顧問|顾问|專員|专员|執行長|执行长|理事長|理事长|董事長|董事长|會長|会长|社長|社长|總裁|总裁|總經理|总经理|副總|副总)",
                      re.UNICODE)
        ], 
        "FAMILYNAME": [# 中文姓氏 + 敬稱模式，只捕獲並提取姓氏
            re.compile(r"(陈)(?:先生|女士|小姐|太太|夫人|同學|同学|醫師|医师|護士|护士|同事|朋友)",
                      re.UNICODE)
            ]
    }
    matched = []
    for category, patterns in PATTERNS_UPDATE.items():
        for pat in patterns:
            match = pat.search(text)  # <--- 使用 search() 並保存結果
            if match:
                # 提取實際匹配的文字，去除前後的邊界字符
                matched_text = match.group(0).strip()
                
                # 如果匹配的文字以空格或標點開始/結束，去除它們
                # 因為我們的模式包含了邊界檢查 (^|\s) 和 (\.|$|\s)
                if matched_text.startswith(' '):
                    matched_text = matched_text[1:]
                if matched_text.endswith(' ') or matched_text.endswith('.'):
                    matched_text = matched_text[:-1]
                
                if matched_text:  # 確保不是空字符串
                    matched.append({"category": category, "text": matched_text})
                    # break  # 找到匹配就跳出這個類別的循環
    return matched

def remove_prefix_suffix_redundant_words(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    針對不同的實體標籤類別，移除前後的贅字
    """
    if not entities:
        return entities
    
    cleaned_entities = []
    
    # 定義每個類別需要移除的前綴和後綴詞彙
    CATEGORY_CLEANING_RULES = {
        "DOCTOR": {
            "prefixes_to_remove": ["dr.", "dr", "doctor", "drs.", "drs"],
            "suffixes_to_remove": [".", ","]
        },
        "AGE": {
            "prefixes_to_remove": [],
            "suffixes_to_remove": ["years old", "-year-old", "year-old", "years", "year", "old", ".", ","]
        },

    }
    
    for entity in entities:
        entity_text = entity.get("text", "").strip()
        entity_category = entity.get("category", "").upper()
        
        if not entity_text or not entity_category:
            cleaned_entities.append(entity)
            continue
        
        # 取得清理規則
        cleaning_rules = CATEGORY_CLEANING_RULES.get(entity_category, {})
        if not cleaning_rules:
            # 如果沒有特定規則，只做基本清理
            cleaned_text = entity_text.strip().rstrip(".,!?;:")
            cleaned_entities.append({"text": cleaned_text, "category": entity_category})
            continue
        
        cleaned_text = entity_text
        
        # 1. 移除前綴詞彙
        prefixes_to_remove = cleaning_rules.get("prefixes_to_remove", [])
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower() + " "):
                cleaned_text = cleaned_text[len(prefix):].strip()
            elif cleaned_text.lower().startswith(prefix.lower()) and len(cleaned_text) > len(prefix):
                # 處理沒有空格的情況
                next_char_idx = len(prefix)
                if next_char_idx < len(cleaned_text) and cleaned_text[next_char_idx] in " .,":
                    cleaned_text = cleaned_text[len(prefix):].strip()
        
        # 2. 移除後綴詞彙（按照定義的順序，從長到短）
        suffixes_to_remove = cleaning_rules.get("suffixes_to_remove", [])
        for suffix in suffixes_to_remove:
            # 檢查是否以 " " + suffix 結尾
            if cleaned_text.lower().endswith(" " + suffix.lower()):
                cleaned_text = cleaned_text[:-len(suffix)-1].strip()
                break  # 只移除第一個匹配的後綴
            # 檢查是否直接以 suffix 結尾
            elif cleaned_text.lower().endswith(suffix.lower()) and len(cleaned_text) > len(suffix):
                # 特殊處理：對於 AGE 類別，確保我們不會完全刪除數字部分
                if entity_category == "AGE":
                    remaining_text = cleaned_text[:-len(suffix)].strip()
                    # 確保剩餘文字包含數字
                    if remaining_text and any(c.isdigit() for c in remaining_text):
                        cleaned_text = remaining_text
                        break
                else:
                    cleaned_text = cleaned_text[:-len(suffix)].strip()
                    break  # 只移除第一個匹配的後綴
        
        # 3. 最終清理：移除多餘的空格
        final_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 4. 如果清理後文字為空，保留原始文字
        if not final_text:
            final_text = entity_text.strip()
        
        cleaned_entities.append({"text": final_text, "category": entity_category})
        
        # 如果文字有變化，輸出調試信息
        if final_text != entity_text:
            print(f"  Cleaned {entity_category}: '{entity_text}' -> '{final_text}'")
    
    return cleaned_entities

def filter_entities(entities: List[Dict[str, str]], rules: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Filters a list of entities based on predefined rules.
    Removes entities where the category matches a rule category and the
    entity text (case-insensitive) matches one of the forbidden texts for that category.
    """
    filtered_list = []
    # Prepare lowercased filter texts for efficient lookup
    lower_rules = {category: [text.lower() for text in texts]
                   for category, texts in rules.items()}

    for entity in entities:
        category = entity.get("category").upper()
        text = entity.get("text").lower()

        # Special handling for ZIP - allow only if it's exactly 4 digits
        if category == "ZIP":
            if not (len(text) == 4 and text.isdigit()):
                print(f"  Filtering out ZIP entity: {{'text': '{text}', 'category': '{category}'}} - not 4 digits")
                continue

        if not category or not text:
            continue # Skip invalid entities

        # Check if the category has filter rules
        if category in lower_rules:
            # Check if the entity text (lowercased) is in the forbidden list for this category
            if text.lower() in lower_rules[category]:
                print(f"  Filtering out entity: {{'text': '{text}', 'category': '{category}'}} based on rules.")
                continue # Skip this entity

        # If not filtered out, add it to the list
        filtered_list.append(entity)



    return filtered_list

def filter_subsumed_entities(
    entity_matches: List[Tuple[str, str, float, float]]
) -> List[Tuple[str, str, float, float]]:
    """
    過濾時間戳實體列表，移除那些被同一類別、時間重疊且文本更長的實體
    所包含的較短實體。文本比較不區分大小寫。

    例如：若同時存在 ('TIME', 'This morning', 1.0, 2.0) 和
                     ('TIME', 'morning', 1.5, 2.0)，
          則 'morning' 會被移除。
    """
    if not entity_matches:
        return []

    final_kept_entities = []
    indices_to_discard = set()

    for i in range(len(entity_matches)):
        if i in indices_to_discard:
            continue

        current_entity = entity_matches[i]
        current_category, current_text, current_start, current_end = current_entity
        
        # 檢查 current_entity 是否被其他更長的實體所包含
        # 注意：這裡的邏輯是，如果 current_entity 較短且被包含，則將其加入 discard 列表
        # 另一種寫法是，遍歷所有實體，決定每個實體是否應該被保留。下面的寫法更直接。

    # 決定每個實體是否應該被保留
    for i in range(len(entity_matches)):
        current_entity = entity_matches[i]
        current_category, current_text, current_start, current_end = current_entity
        
        is_subsumed_by_a_longer_entity = False
        should_be_filtered_due_to_age_duration_conflict = False
        for j in range(len(entity_matches)):
            if i == j:
                continue

            other_entity = entity_matches[j]
            other_category, other_text, other_start, other_end = other_entity

            # 檢查時間是否有重疊: max(start1, start2) < min(end1, end2)
            time_overlap = max(current_start, other_start) < min(current_end, other_end)

            if time_overlap:
                # 特殊處理：AGE 和 DURATION 的衝突
                if (current_category == "DURATION" and other_category == "AGE") or \
                   (current_category == "AGE" and other_category == "DURATION"):
                    # 如果當前實體是 DURATION 且有 AGE 重疊，則過濾掉 DURATION
                    if current_category == "DURATION":
                        should_be_filtered_due_to_age_duration_conflict = True
                        print(f"  Filtering DURATION '{current_text}' due to overlapping AGE '{other_text}'")
                        break
                    # 如果當前實體是 AGE，則保留（不做任何處理，繼續檢查其他條件）
                
                # 原有的同類別子字串過濾邏輯
                elif current_category == other_category:
                    # 檢查 current_text 是否是 other_text 的子字串 (不區分大小寫)
                    # 並且 current_text 的長度嚴格小於 other_text 的長度
                    if current_text.lower() in other_text.lower() and \
                       len(current_text) < len(other_text):
                        is_subsumed_by_a_longer_entity = True
                        print(f"  Filtering subsumed entity '{current_text}' contained in '{other_text}'")
                        break  # 找到一個更長的實體包含當前實體，無需再比較

        # 如果實體沒有被過濾，則保留
        if not is_subsumed_by_a_longer_entity and not should_be_filtered_due_to_age_duration_conflict:
            final_kept_entities.append(current_entity)
            
    return final_kept_entities

def get_entities_from_llm_with_confirmation(text: str,model, tokenizer, device,is_chinese) -> Optional[List[Dict[str, str]]]:
    

    
    """
    先請LLM抽取entities，再從字典查找，過濾不需要的實體，最後將entities highlight於原文，請LLM再次確認，回傳最終entities。
    """
    # Define categories that should be prioritized in sorting
    PRIORITY_CATEGORIES = ["PERSONALNAME","DOCTOR", "PATIENT","SET"]

    # 1. Collect all candidate entities from various sources
    print("  Searching entities in dictionary...")
    entities1 = find_entities_from_dictionary(text, entity_dictionary,is_chinese)
    print(f"  from dictionary found {len(entities1)} entities.")

    print("  Searching entities using regex patterns...")
    entities2 = get_pattern_match_category(text) # Assuming this returns list of {"text": ..., "category": ...}
    print(f"  from regex patterns found {len(entities2)} entities.")

    print("  Calling LLM ...")
    entities3 = get_entities_from_llm(text, tokenizer, model, device)
    print(f"  from LLM found {len(entities3)} entities.")

    all_raw_entities = entities1 + entities2 + entities3
    
    all_candidate_entities = []
    for ent in all_raw_entities:
        if isinstance(ent, dict) and ent.get("text") and ent.get("category") in TAG_List:
            all_candidate_entities.append({"text": str(ent["text"]), "category": str(ent["category"])})
        # else:
        #     print(f"Warning: Skipping invalid entity during collection: {ent}")
    
    print(f"Total valid candidate entities from all sources: {len(all_candidate_entities)}")

    # 2. Sort candidates:
    #    - Primary sort: entities in PRIORITY_CATEGORIES come first.
    #    - Secondary sort: by text length (descending).
    sorted_candidates = sorted(
        all_candidate_entities,
        key=lambda e: (
            e["category"] not in PRIORITY_CATEGORIES, # False (0) for priority, True (1) for others
            -len(e["text"])  # Sort by length descending
        )
    )

    # # 3. Initialize for merging based on non-overlapping spans
    # merged_entities_after_custom_logic = []
    # coverage_map = [False] * len(text) # True if char index is covered

    # # 4. Iterate through sorted candidates and select non-overlapping entities
    # for candidate in sorted_candidates:
    #     cand_text = candidate["text"]
    #     cand_category = candidate["category"]

    #     if not cand_text: # Skip empty text candidates
    #         continue

    #     try:
    #         # Find all occurrences of this candidate's text in the original text (case-insensitive for matching position)
    #         for match in re.finditer(re.escape(cand_text), text, flags=re.IGNORECASE):
    #             start_idx, end_idx = match.span()
                
    #             # Check if this specific span is already (fully or partially) covered
    #             is_span_covered = any(coverage_map[i] for i in range(start_idx, end_idx))

    #             if not is_span_covered:
    #                 # Add this entity instance using the exact text from the original document
    #                 merged_entities_after_custom_logic.append({
    #                     "text": text[start_idx:end_idx], 
    #                     "category": cand_category
    #                 })
    #                 # Mark this span as covered
    #                 for i in range(start_idx, end_idx):
    #                     coverage_map[i] = True
    #     except re.error as e:
    #         print(f"Warning: Regex error for candidate text '{cand_text}': {e}")
    #         # Continue with the next candidate if current one causes regex error
    #         continue
            
    # print(f"  Entities after new merging logic (length & non-overlap): {len(merged_entities_after_custom_logic)}")

    # 新增：對最終要返回的實體列表進行去重
    final_unique_entities_dict = {} # 鍵是 normalized_entity_text，值是 entity 字典
    
    # 為了方便查找 PRIORITY_CATEGORIES 的順序，可以將其轉換為帶索引的字典
    priority_order = {category: i for i, category in enumerate(PRIORITY_CATEGORIES)}

    for ent in sorted_candidates:
        if isinstance(ent, dict) and "text" in ent and "category" in ent:
            normalized_entity_text = normalize_text(ent["text"])
            if not normalized_entity_text:
                continue
            
            current_category = ent["category"]

            if normalized_entity_text not in final_unique_entities_dict:
                # 如果這個文本還沒有被記錄，直接記錄當前實體
                final_unique_entities_dict[normalized_entity_text] = ent
            else:
                # 如果文本已存在，則需要比較類別的優先級
                existing_entity = final_unique_entities_dict[normalized_entity_text]
                existing_category = existing_entity["category"]

                # 獲取當前類別和已存在類別的優先級順序
                # 如果類別不在 priority_order 中，給一個較大的值表示低優先級
                current_priority = priority_order.get(current_category, len(PRIORITY_CATEGORIES))
                existing_priority = priority_order.get(existing_category, len(PRIORITY_CATEGORIES))

                if current_priority < existing_priority:
                    # 如果當前實體的類別有更高的優先級（值更小），則替換
                    final_unique_entities_dict[normalized_entity_text] = ent
                elif current_priority == existing_priority:
                    # 如果優先級相同，我們已經進行了長度排序，
                    # sorted_candidates 保證了較長的實體會先被處理。
                    # 由於我們是迭代 sorted_candidates，並且在 normalized_entity_text 首次出現時就存儲，
                    # 所以這裡不需要額外比較長度，因為已存儲的 existing_entity 已經是該優先級下較優的（或第一個遇到的）。
                    # 如果需要嚴格按長度（即使優先級相同），可以取消下面註解
                    # if len(ent["text"]) > len(existing_entity["text"]):
                    #     final_unique_entities_dict[normalized_entity_text] = ent
                    pass # 保留已有的（因為它是在排序中先遇到的，或者長度更長）


    unique_entities_list = list(final_unique_entities_dict.values()) 
    print(f"  Entities after uniqueness filter (text-based, respecting PRIORITY_CATEGORIES): {len(unique_entities_list)}")

    filtered_combined_entities = filter_entities(unique_entities_list, FILTER_RULES)
    print(f"  Entities after filtering: {len(filtered_combined_entities)}")

    filtered_combined_entities = remove_prefix_suffix_redundant_words(filtered_combined_entities)
    print(f"  Entities after removing redundant prefixes/suffixes: {len(filtered_combined_entities)}")
    
    if not filtered_combined_entities:
        return [] # Return empty list if nothing left after filtering

    # # highlight entities in text for confirmation
    # highlight_text = text
    # # Sort by length descending to avoid partial highlights
    # # Use the filtered list for highlighting
    # for ent in sorted(filtered_combined_entities, key=lambda e: -len(e.get('text', '')), reverse=True):
    #     ent_text = ent.get('text')
    #     ent_category = ent.get('category')
    #     if not ent_text or not ent_category:
    #         continue
    #     # Use regex for safer replacement, escaping potential special chars in entity text
    #     try:
    #         # Replace only the first occurrence for this simple highlight method
    #         pattern = re.escape(ent_text)
    #         # Use a function in sub to avoid issues with overlapping matches being processed multiple times
    #         # This simple approach still has limitations with truly overlapping entities.
    #         highlight_text = re.sub(pattern, f"[[{ent_text}|{ent_category}]]", highlight_text, count=1)
    #     except re.error as e:
    #         print(f"Warning: Regex error during highlighting for entity '{ent_text}': {e}")
    #         # Fallback to simple string replacement (less safe)
    #         highlight_text = highlight_text.replace(ent_text, f"[[{ent_text}|{ent_category}]]", 1)


    # # 確認highlight的結果 (using the original text and the highlighted version based on filtered entities)
    # print("  Calling LLM for confirmation...")
    # confirmed_entities = duble_check(highlight_text, text)
    # print(f"  LLM confirmation returned {len(confirmed_entities)} entities.")

    # Optional: Apply filtering AGAIN after confirmation if LLM might re-introduce filtered items
    # print("  Applying filter rules after confirmation...")
    # final_filtered_entities = filter_entities(confirmed_entities, FILTER_RULES)
    # print(f"  Entities after final filtering: {len(final_filtered_entities)}")
    # return final_filtered_entities

    return filtered_combined_entities # Return entities confirmed by LLM

def find_entity_timestamps(
    words_with_times: List[Dict[str, Any]],
    entities: List[Dict[str, str]],
    is_chinese:bool
) -> List[Tuple[str, str, float, float]]:
    """
    Match entities to words and find start/end times (with fuzzy matching).
    Returns a list of tuples: (category, entity_text, start_time, end_time)
    """
    results = []
    if not words_with_times or not entities:
        return results

    # Create a list of normalized words and their original indices/data
    normalized_words_info = [
        {"norm": normalize_text(word_data.get("word")), "orig_idx": i, "data": word_data}
        for i, word_data in enumerate(words_with_times)
        if word_data and "word" in word_data and word_data.get("word") # Ensure word exists and is not empty
    ]

    

    transcript_words_norm = [w["norm"] for w in normalized_words_info]
    num_words = len(normalized_words_info)

    for entity in entities:
        entity_text_orig = entity.get("text", "")
        entity_category = entity.get("category", "UNKNOWN")
        entity_text_norm = normalize_text(entity_text_orig)

        if not entity_text_norm:
            continue

        entity_words_norm = entity_text_norm.split()
        num_entity_words = len(entity_words_norm)

        found_match = False
        # 檢查是否需要繼續尋找多個匹配
        should_continue_search = (is_chinese == True and entity_category == "COUNTRY")
        
        # 先精確比對
        for i in range(num_words - num_entity_words + 1):
            # Check if the sequence of normalized words matches
            match = True
            for j in range(num_entity_words):
                if transcript_words_norm[i + j].strip().lower() != entity_words_norm[j].strip().lower():
                    match = False
                    break

            if match:
                # Found a match, get start/end times from original data
                start_word_info = normalized_words_info[i]["data"]
                end_word_info = normalized_words_info[i + num_entity_words - 1]["data"]

                start_time = start_word_info.get("start")
                end_time = end_word_info.get("end")

                # Ensure times are valid floats
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    results.append((entity_category, entity_text_orig, float(start_time), float(end_time)))
                    found_match = True
                    # Optional: break here if you only want the first match
                    # break
                    
                else:
                     print(f"Warning: Invalid start/end time for matched entity '{entity_text_orig}' "
                           f"(Word indices {normalized_words_info[i]['orig_idx']} to "
                           f"{normalized_words_info[i + num_entity_words - 1]['orig_idx']}). "
                           f"Start: {start_time}, End: {end_time}")

        # 若精確比對失敗，進行fuzzy matching
        if not found_match:
            
            ### 特殊處理：對於中文國家名稱，可能需要多個匹配
            if(should_continue_search):
                temp_index = 0
                for i in range(2):
                    fuzzy_result = fuzzy_find_entity_in_transcript(entity_text_norm, transcript_words_norm,is_chinese)
                    if fuzzy_result:
                        i, j = fuzzy_result
                        
                        
                        start_word_info = normalized_words_info[temp_index+i]["data"]
                        end_word_info = normalized_words_info[temp_index+j]["data"]

                        start_time = start_word_info.get("start")
                        end_time = end_word_info.get("end")

                        # Ensure times are valid floats
                        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                            results.append((entity_category, entity_text_orig, float(start_time), float(end_time)))
                            found_match = True
                            transcript_words_norm = transcript_words_norm[j+1:]  # Remove matched words
                            temp_index = j + 1  # Update index for next search
            
            else:
                fuzzy_result = fuzzy_find_entity_in_transcript(entity_text_norm, transcript_words_norm,is_chinese)
                if fuzzy_result:
                    i, j = fuzzy_result
                    
                    
                    start_word_info = normalized_words_info[i]["data"]
                    end_word_info = normalized_words_info[j]["data"]

                    start_time = start_word_info.get("start")
                    end_time = end_word_info.get("end")

                    # Ensure times are valid floats
                    if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                        results.append((entity_category, entity_text_orig, float(start_time), float(end_time)))
                        found_match = True
                    
                    

        if not found_match:
            print(f"Warning: Could not find (even fuzzy) timestamp match for entity: '{entity_text_orig}' (Normalized: '{entity_text_norm}')")

    return results

# --- Main Processing Logic ---

def main():
    print(f"Loading time-step data from: {TIME_STEP_JSON_PATH}")
    try:
        with open(TIME_STEP_JSON_PATH, 'r', encoding='utf-8') as f:
            time_step_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {TIME_STEP_JSON_PATH}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {TIME_STEP_JSON_PATH}: {e}")
        return

    all_output_lines = []
    audio_ids = list(time_step_data.keys())
    
    # Sort audio_ids in ascending order
    audio_ids.sort()
    
    print(f"Found {len(audio_ids)} audio IDs to process.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model with same configuration as training
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
        model.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # audio_ids = audio_ids[:50]  # For testing, limit to first 5 audio IDs


    for audio_id in tqdm(audio_ids, desc="Processing audio IDs"):
        
        print(f"\nProcessing Audio ID: {audio_id}")
        
        # if int(audio_id) < 79999:
        #     print(f"  Skipping Audio ID {audio_id} as it is greater than 79999 (Chinese audio).")
        #     continue
        segments = time_step_data[audio_id].get("segments", [])
        if not segments:
            print(f"Warning: No segments found for audio ID {audio_id}")
            continue

        # Assuming entities are relevant to the whole audio file,
        # process words from all segments together if needed.
        # For simplicity here, we'll process the first segment's words,
        # or combine words if multiple segments exist.
        all_words_for_audio = []
        for segment in segments:
             words = segment.get("words", [])
             # Filter out invalid word entries before extending
             valid_words = [w for w in words if isinstance(w, dict) and "word" in w and "start" in w and "end" in w]
             all_words_for_audio.extend(valid_words)


        if not all_words_for_audio:
            print(f"Warning: No valid words with timestamps found for audio ID {audio_id}")
            continue

        # 1. Reconstruct text
        
        if(int(audio_id)) > 79999:
            full_text = reconstruct_text_from_words_zh(all_words_for_audio)
            is_chinese = True
        else:
            full_text = reconstruct_text_from_words(all_words_for_audio)
            is_chinese = False
        print(f"  Reconstructed Text (first 100 chars): {full_text[:100]}...")

        # 2. Get entities from LLM (with confirmation)
        print("  Calling LLM for NER...")
        entities = get_entities_from_llm_with_confirmation(full_text,model, tokenizer, device,is_chinese)

        if entities is None:
            print(f"  Failed to get entities from LLM for audio ID {audio_id}. Skipping.")
            continue
        if not entities:
            print(f"  No entities returned by LLM for audio ID {audio_id}.")
            continue

        print(f"  LLM returned entities :{len(entities)} .")

        # 3. Find timestamps for entities
        entity_matches = find_entity_timestamps(all_words_for_audio, entities,is_chinese)

        # 4. Filter subsumed entities AFTER finding timestamps
        filtered_entity_matches_after_subsumption = filter_subsumed_entities(entity_matches)
        print(f"  Retained entities after filtering subsumed ones :{len(filtered_entity_matches_after_subsumption)} .")

        # 4.1. Sort by start time (third column, index 2)
        filtered_entity_matches_after_subsumption.sort(key=lambda x: x[2])
        print(f"Return {filtered_entity_matches_after_subsumption}")
        
        # 4.2. Remove duplicates based on all four fields (category, text, start, end)
        unique_entity_matches = []
        seen_entities = set()
        
        for category, text, start, end in filtered_entity_matches_after_subsumption:
            # Create a tuple with all four fields for comparison
            entity_tuple = (category, text, round(start, 2), round(end, 2))
            
            if entity_tuple not in seen_entities:
                seen_entities.add(entity_tuple)
                unique_entity_matches.append((category, text, start, end))
            else:
                print(f"  Removing duplicate entity: {category}, '{text}', {start:.2f}, {end:.2f}")
        
        print(f"  Retained entities after removing duplicates: {len(unique_entity_matches)}")
        
        
        # 5. Format output lines
        for category, text, start, end in filtered_entity_matches_after_subsumption:
            # Format: audio_id\tcategory\tstart_time\tend_time\tentity_text
            line = f"{audio_id}\t{category}\t{start:.2f}\t{end:.2f}\t{text}"
            all_output_lines.append(line)
        print(f"  Found timestamps for {len(filtered_entity_matches_after_subsumption)} entities.")

    
    # 5. Save results to TSV
    print(f"\nSaving results to: {OUTPUT_TSV_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_TSV_PATH), exist_ok=True)
    try:
        with open(OUTPUT_TSV_PATH, 'w', encoding='utf-8',newline="\n") as f:
            # Optional: Write header
            # f.write("AudioID\\tCategory\\tStart\\tEnd\\tText\\n")
            for line in all_output_lines:
                f.write(line + "\n")
        print("Processing complete.")
    except IOError as e:
        print(f"Error: Could not write output file {OUTPUT_TSV_PATH}: {e}")


if __name__ == "__main__":
    main()

