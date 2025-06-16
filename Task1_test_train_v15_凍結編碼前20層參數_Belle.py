###!!!! 有限制whisper 448最大序列長度，超過448會報錯
### https://link.springer.com/article/10.1186/s13636-024-00349-3
### 冻结 Whisper 编码器前 20 层参数，其余层保持可训练。

import re
import os
import pandas as pd
import torch
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from jiwer import compute_measures
from tqdm import tqdm
import evaluate
from transformers import EarlyStoppingCallback
from whisper_normalizer.english import EnglishTextNormalizer
import opencc
# Initialize the WER metric
metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def Freeze_Parameters(model, n_frozen: int = 20):
    """
    冻结 Whisper 编码器前 n_frozen 层参数，其余层保持可训练。
    """
    # 遍历每一层
    for idx, layer in enumerate(model.model.encoder.layers):
        # 如果层索引小于 n_frozen，则冻结；否则解冻
        requires_grad = False if idx < n_frozen else True
        for param in layer.parameters():
            param.requires_grad = requires_grad
    return model


def mixed_text_normalizer(text: str) -> str:
    """
    一個適用於中英文混合文本的簡易標準化函式。
    - 英文轉換為小寫
    - 移除中英文常見標點符號
    - 標準化空格
    """
    if not isinstance(text, str):
        return ""

    # 1. 英文部分轉換為小寫 (對中文無影響)
    text = text.lower()

    # 2. 移除中英文常見標點符號
    punctuation_to_remove = (
        r"[.,?!\"';:()\\[\\]{}<>@#%&*+=~`|_/^\\—–…\u2014\u2013\u2026\u2018\u2019\u201c\u201d"  # 英文標點部分
        r"\uff0c\u3002\uff1f\uff01\u300c\u300d\u300e\u300f\uff1b\uff1a\uff08\uff09\u300a\u300b\u3008\u3009"  # 中文標點部分
        r"﹏—．‧＃＠＆＊％※／＼＋－＝～｀＿｜]" # 其他可能的中英文符號 (有些可能與英文部分重複，但不影響)
    )
    text = re.sub(punctuation_to_remove, "", text)



    # 3. 將多個空格替換為單個空格
    text = re.sub(r"\s+", " ", text)

    # 4. 移除文本前後的空格
    text = text.strip()

    return text
def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Convert predictions to a format suitable for batch_decode
    if isinstance(pred_ids[0], list):
        pred_ids = [item for sublist in pred_ids for item in sublist]
    
    # Convert predictions to tensor if they're not already
    if not isinstance(pred_ids, torch.Tensor):
        pred_ids = torch.tensor(pred_ids)
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids.reshape(-1, pred_ids.shape[-1]), skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [mixed_text_normalizer(x) for x in pred_str]
    label_str = [mixed_text_normalizer(x) for x in label_str]

    # Convert simplified Chinese to traditional Chinese
    try:
        converter = opencc.OpenCC('s2t')  # simplified to traditional
        pred_str = [converter.convert(x) for x in pred_str]
        label_str = [converter.convert(x) for x in label_str]
    except ImportError:
        print("Warning: opencc not installed. Skipping Chinese conversion.")
        pass


    # Calculate WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # Extract labels
        label_features = [feature["labels"] for feature in features]
        batch_labels = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        labels = batch_labels["input_ids"].masked_fill(batch_labels.attention_mask.ne(1), -100)
        
        # Add labels to batch
        batch["labels"] = labels
        
         # 保留 audio_file_name
        if "audio_file_name" in features[0]:
            batch["audio_file_name"] = [feature["audio_file_name"] for feature in features]


        return batch

def prepare_dataset(train_path: str, validation_path: str):
    # 讀取訓練資料集的metadata
    train_metadata_df = pd.read_csv(os.path.join(train_path, "task1_answer.txt"), sep="\t", header=None)

    converter = opencc.OpenCC('t2s')  # simplified to traditional

    for i in range(len(train_metadata_df[1])):
        train_metadata_df.loc[i, 1] = converter.convert(train_metadata_df.loc[i, 1])
        
    # 創建訓練資料集字典
    train_dict = {
        "audio": [os.path.join(train_path, str(audio_file)+".wav") for audio_file in train_metadata_df[0]],
        "text": train_metadata_df[1].tolist()
    }
    
    # 對於驗證集，只讀取音訊檔案
    val_audio_files = [f for f in os.listdir(validation_path) if f.endswith('.wav')]
    val_dict = {
        "audio": [os.path.join(validation_path, audio_file) for audio_file in val_audio_files],
        "text": ["" for _ in val_audio_files]  # 空字串作為預設文字
    }
    
    # 創建資料集
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    
    # 新增音訊載入功能
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
    train_test_dataset = train_dataset.train_test_split(test_size=0.2,shuffle=False)
    
    # 回傳兩個資料集
    return {
        "train": train_test_dataset["train"],
        "test": train_test_dataset["test"],
        "val": val_dataset
    }

def prepare_features(batch, processor):
    # Process audio
    audio = batch["audio"]
    
    # Compute log-mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        # language="en",   # or "zh"
        return_tensors="pt"
    ).input_features[0]
    
    lang_token_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
    sot_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    transcribe_token_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    
    # Encode target text to label ids 並限制最大長度為 448
    batch["labels"] = processor.tokenizer(
        batch["text"],
        max_length=445,
        truncation=True,
        padding="max_length",
        
    ).input_ids
    
    batch["labels"] = [sot_token_id, lang_token_id, transcribe_token_id] + batch["labels"]
    
    batch["audio_file_name"] = os.path.splitext(os.path.basename(batch["audio"]["path"]))[0]
    
    return batch

def setup_directories():
    # Create result directory and version subdirectory
    result_dir = "result"
    result_dir = os.path.join(result_dir, "task1")
    version_dir = os.path.join(result_dir, "v15_Belle_large_v3_train_凍結編碼前20層參數")
    model_dir = os.path.join(version_dir, "model")
    
    # Create directories if they don't exist
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return version_dir, model_dir


# Save predictions to TSV file  VALIDATION
def save_predictions_to_tsv(predictions,filenames, version_dir):
    task1_output_file = os.path.join(version_dir, "task1_answer.txt")
    output_file = os.path.join(version_dir, "val_results.txt")
    print("Saving predictions to file...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (pred) in enumerate(zip(predictions), 1):
            f.write(f"Predict: {pred}\n\n")
            
            
    print(f"Results saved to {output_file}")
    with open(task1_output_file, "w", encoding="utf-8", newline='\n') as f:
        for i, (pred,file) in enumerate(zip(predictions,filenames), 1):
            f.write(f"{file}\t{pred}\n")       
    
    print(f"Results saved to {task1_output_file}")

# Save predictions to TSV file  TEST
def save_predictions_to_tsv_1(predictions, references,ids,wers,cers, wer_score,cer_score, version_dir):
    output_file = os.path.join(version_dir, "test_results.txt")
    print("Saving predictions to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Overall Word Error Rate (WER): {wer_score:.2f}%\n\n")
        f.write(f"Overall Character Error Rate (CER): {cer_score:.2f}%\n\n")
        f.write(f"mix scored WER+CER/2: {(wer_score+cer_score)/2:.2f}%\n\n")
        f.write("Detailed Results:\n\n")
        f.write("--------------------------------------------------\n")
        for i, (ref, pred,id,wer,cer) in enumerate(zip(references, predictions,ids,wers,cers), 1):
            f.write(f"ID        :{id}\n")
            f.write(f"Reference :{ref}\n")
            f.write(f"Predict   :{pred}\n")
            f.write(f"WER       :{wer:.2f}%\n")
            f.write(f"CER       :{cer:.2f}%\n\n")
            
    print(f"Results saved to {output_file}")

def calculate_mer(trainer,test_dataloader,data_collator,version_dir):
    print("Calculating WER score...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = trainer.model.device
    predictions = []
    references = []
    ids = []
    wers = []
    cers = []
    
    i = 0
    for batch in tqdm(test_dataloader, desc="處理測試資料", unit="batch"):
        with torch.no_grad():
            collator_batch = data_collator([batch])
            # Move batch to device
            input_features = collator_batch["input_features"].to(device)
            labels = collator_batch["labels"]
            
            
            # Generate predictions
            generated_ids = trainer.model.generate(input_features=input_features, max_length=256)
            
            # Decode predictions and references
            pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(labels, skip_special_tokens=True)
            
            pred_str = [mixed_text_normalizer(x) for x in pred_str]
            label_str = [mixed_text_normalizer(x) for x in label_str]
            
            # Convert simplified Chinese to traditional Chinese
            try:
                converter = opencc.OpenCC('s2t')  # simplified to traditional
                pred_str = [converter.convert(x) for x in pred_str]
                label_str = [converter.convert(x) for x in label_str]
            except ImportError:
                print("Warning: opencc not installed. Skipping Chinese conversion.")
                pass
             
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)
            cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        
            predictions.extend(pred_str)
            references.extend(label_str)
            ids.extend([collator_batch["audio_file_name"]])  # 確保這行正確收集檔案名稱
            wers.append(wer)
            cers.append(cer)
            i += 1
    
    # Calculate WER
    wer = 100 * metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"mix scored WER+CER/2: {(wer+cer)/2:.2f}%")
    
    save_predictions_to_tsv_1(predictions, references,ids,wers,cers, wer,cer, version_dir)

def calculate_output(trainer, eval_dataloader,data_collator, version_dir):
    print("Calculating ouput score...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = trainer.model.device
    predictions = []
    filenames = []
    
    i = 0
    for batch in tqdm(eval_dataloader, desc="處理評估資料", unit="batch"):
        with torch.no_grad():
            collator_batch = data_collator([batch])
            
            # Move batch to device
            input_features = collator_batch["input_features"].to(device)
            
            # Generate predictions
            generated_ids = trainer.model.generate(input_features=input_features)
            
            # Decode predictions and references
            pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # normalizer = EnglishTextNormalizer()
            pred_str = [mixed_text_normalizer(x) for x in pred_str]
            
                        # Convert simplified Chinese to traditional Chinese
            try:
                converter = opencc.OpenCC('s2t')  # simplified to traditional
                pred_str = [converter.convert(x) for x in pred_str]
            except ImportError:
                print("Warning: opencc not installed. Skipping Chinese conversion.")
                pass
            
            predictions.extend(pred_str)
            filenames.extend([batch["audio_file_name"]])  # 確保這行正確收集檔案名稱
            i += 1
            
    
    # Save predictions to file
    save_predictions_to_tsv(predictions,filenames, version_dir)
    
    return 0

def main():
    global processor  # Make processor globally accessible for compute_metrics
    
    # Setup directories
    version_dir, model_dir = setup_directories()
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model and processor
    model_name = "BELLE-2/Belle-whisper-large-v3-zh"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                    device_map="auto",
                                                    offload_folder="offload",
                                                    offload_state_dict=True)
    
    model = Freeze_Parameters(model)  # Freeze parameters
    # Move model to GPU if available
    # model = model.to(device)
    
    # Prepare dataset
    dataset = prepare_dataset("XXX","XXX")

    # Process datasets
    
    processed_dataset = {}
    for split in ["train", "test", "val"]:
        processed_dataset[split] = dataset[split].map(
            lambda x: prepare_features(x, processor=processor),
            remove_columns=["audio", "text"],
            num_proc=1,  # Number of processes to use for parallel processing
        )
    
    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),  # Save checkpoints in model directory
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=100,
        #max_steps=20,
        num_train_epochs=10,
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_bnb_8bit",
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=True,
        dataloader_num_workers=2,      # 根據你的 CPU 和系統調整
        save_total_limit=1,      # Save only the best model
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)], # Early stopping after 10 evaluations without improvement
    )
    
    # Train model
    trainer.train()
    
    trainer.model.eval()
    
    # Load best model
    best_model_path = os.path.join(training_args.output_dir, "checkpoint-best")
    if (os.path.exists(best_model_path)):
        trainer.model = WhisperForConditionalGeneration.from_pretrained(best_model_path).to(device)
    
    # trainer.evaluate()
    # Calculate and display WER score
    calculate_mer(trainer, processed_dataset["test"],data_collator, version_dir)
    calculate_output(trainer, processed_dataset["val"],data_collator, version_dir)
    
    # Save the fine-tuned model
    print("\nFinal evaluation results:")
    
    final_model_path = os.path.join(model_dir, "final")
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()