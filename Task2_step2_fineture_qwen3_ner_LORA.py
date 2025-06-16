"""
Task2_finetune_qwen3_ner.py
Fine-tune Qwen3:4B for NER (Named Entity Recognition) in the Task2 format.
This script prepares a dataset, fine-tunes the model, and saves the result.
"""
import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, EarlyStoppingCallback, BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import random
# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B"  # HuggingFace model hub name
OUTPUT_DIR = "./task2/qwen3_ner_finetuned_10_LORA_allDATA_CombinePre"
DATA_PATH = "./task2/merged_output_CombinePre.json"  # Path to your NER training data (see below for format)
MAX_LENGTH = 3072
BATCH_SIZE  = 2
EPOCHS = 3
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.01  # 20% for validation
EARLY_STOPPING_PATIENCE = 3
ENABLE_QUANTIZATION = True  # Enable 4-bit quantization


# LoRA Configuration
LORA_R = 64  # LoRA rank
LORA_ALPHA = 128  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout

# --- Data Preparation ---
# The training data should be a JSONL file with lines like:
# {"text": "John Smith is 45 years old.", "entities": [{"text": "John Smith", "category": "PATIENT"}, {"text": "45", "category": "AGE"}]}

    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def load_or_create_dataset(data_path: str, answer_path=None) -> Dataset:

    data = json.load(open(data_path, "r", encoding="utf-8"))
        

    # Convert to HF Dataset
    return Dataset.from_list(data)

def format_text(line: Dict) -> Dict:
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


Input: "{line["text"]}"
Output: """
    # The label is the JSON string of entities
    label = json.dumps({"entities": line["entities"]}, ensure_ascii=False)
    return {"prompt": prompt, "label": label}

def preprocess_function(batch, tokenizer):
    prompt_text = batch["prompt"]
    label_text = batch["label"]
    full_text = prompt_text + label_text + tokenizer.eos_token

    # 更精確地處理 BOS token
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=False,
        add_special_tokens=False  # 不添加特殊 token
    )
    
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True  # 完整序列需要 BOS token
    )
    
    tokenized_full = {k: v.squeeze(0) for k, v in tokenized_full.items()}
    
    labels = tokenized_full["input_ids"].clone()
    
    # 考慮到 tokenizer 可能添加了 BOS token，調整 prompt_length
    # 如果 tokenizer 在 full_text 前添加了 BOS token，需要相應調整
    if tokenizer.bos_token_id is not None and tokenized_full["input_ids"][0] == tokenizer.bos_token_id:
        prompt_length_with_bos = len(tokenized_prompt["input_ids"]) + 1  # +1 for BOS
        labels[:prompt_length_with_bos] = -100
    else:
        labels[:len(tokenized_prompt["input_ids"])] = -100

    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
        
    tokenized_full["labels"] = labels
    return tokenized_full

def split_dataset(dataset: Dataset, validation_split: float = 0.1) -> tuple:
    """Split dataset into train and validation sets"""
    dataset_size = len(dataset)
    validation_size = int(dataset_size * validation_split)
    train_size = dataset_size - validation_size
    
    # Shuffle the dataset and split
    shuffled_dataset = dataset.shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, dataset_size))
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def compute_ner_metrics(eval_pred):
    """Compute NER metrics for evaluation"""
    predictions_logits, labels_ids = eval_pred # eval_pred is a tuple (predictions, labels)
                                        # predictions are logits, labels are the input_ids for labels
    
    # Move to CPU to free up GPU memory for argmax and further processing if needed
    # This is a trade-off: CPU processing is slower but uses system RAM
    # predictions_logits = predictions_logits.cpu()
    # labels_ids = labels_ids.cpu()

    # Get predicted token IDs
    predictions = np.argmax(predictions_logits, axis=-1) # Still on GPU if not moved to CPU
    
    # Flatten predictions and labels
    predictions = predictions.flatten()
    labels = labels_ids.flatten() # Assuming labels_ids is already flattened or compatible
    
    # Remove padding tokens (assuming -100 is used for padding)
    mask = labels != -100
    valid_predictions = predictions[mask]
    valid_labels = labels[mask]

    # Free up memory from large tensors if they are no longer needed
    del predictions_logits, labels_ids, predictions, labels, mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if len(valid_labels) == 0: # Handle case with no valid labels to prevent division by zero
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    accuracy = accuracy_score(valid_labels, valid_predictions)
    # For precision, recall, f1, ensure valid_labels and valid_predictions are on CPU if using sklearn
    # If they were on GPU, sklearn might implicitly move them, or you can do it explicitly.
    # If already moved to CPU earlier, this is fine.
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
# --- Main Training Script ---
def main():
    # Configure quantization if enabled
    quantization_config = None
    if ENABLE_QUANTIZATION:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # <--- 新增
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",  # <--- 新增 (如果支援)
        quantization_config=quantization_config,
        # device_map="auto" if ENABLE_QUANTIZATION else None
    )
    if ENABLE_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,  # LoRA rank
        lora_alpha=LORA_ALPHA,  # LoRA alpha
        target_modules=[  # 針對 Qwen 模型的注意力模組
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Load and preprocess dataset
    raw_dataset = load_or_create_dataset(DATA_PATH)
    formatted_dataset = raw_dataset.map(format_text)
    tokenized_dataset = formatted_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        remove_columns=formatted_dataset.column_names
    )
    # Split into train and validation sets
    train_dataset, val_dataset = split_dataset(tokenized_dataset, VALIDATION_SPLIT)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)# Training arguments
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,     # <--- 新增：模擬更大的有效批次
        eval_accumulation_steps=2,         # <--- 新增：減少評估時記憶體壓力
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_strategy="steps",         # <--- 修改
        save_steps=500,                # <--- 新增 (例如每 500 steps 保存一次)
        
        eval_strategy="steps",         # <--- 修改
        eval_steps=100, 
        save_total_limit=2,
        logging_steps=10,
        fp16=not ENABLE_QUANTIZATION,  # 如果使用量化，不要用 fp16
        bf16=False,  # 與量化衝突
        report_to=["tensorboard"],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        group_by_length=True,              # <--- 新增：將相似長度的樣本分組
        warmup_steps=20,                   # <--- 新增：學習率預熱
        lr_scheduler_type="cosine",        # <--- 新增：學習率調度器
        weight_decay=0.01,                 # <--- 新增：權重衰減
        dataloader_pin_memory=False        # <--- 在記憶體緊張時可以關閉
    )    # Trainer with early stopping and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate final model
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
    # Save evaluation results
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    set_seed(42)
    main()