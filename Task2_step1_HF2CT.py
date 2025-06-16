###!!!! 有限制whisper 448最大序列長度，超過448會報錯

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
import evaluate
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from ctranslate2.converters import TransformersConverter
import whisperx
from transformers import WhisperProcessor
import json
from jiwer import wer
from whisper_normalizer.english import EnglishTextNormalizer

# # Initialize the WER metric
metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

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

    # Calculate WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

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

        return batch

def prepare_dataset(train_path: str, validation_path: str):
    # 讀取訓練資料集的metadata
    # train_metadata_df = pd.read_csv(os.path.join(train_path, "task1_answer.txt"), sep="\t", header=None)
    
    
    
    # 對於驗證集，只讀取音訊檔案
    val_audio_files = [f for f in os.listdir(validation_path) if f.endswith('.wav')]
    val_dict = {
        "audio": [os.path.join(validation_path, audio_file) for audio_file in val_audio_files],
        "text": ["" for _ in val_audio_files]  # 空字串作為預設文字
    }
    
    val_dataset = Dataset.from_dict(val_dict)
    
    # 新增音訊載入功能
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    
    # 回傳兩個資料集
    return {
        "val": val_dataset
    }

def prepare_features(batch, processor):
    # Process audio
    audio = batch["audio"]
    
    # Compute log-mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        language="en",   # or "zh"
        return_tensors="pt"
    ).input_features[0]
    
    # Encode target text to label ids 並限制最大長度為 448
    batch["labels"] = processor.tokenizer(
        batch["text"],
        max_length=448,
        truncation=True,
        padding="max_length",
        
    ).input_ids
    batch["file_path"] = batch["audio"]["path"]
    batch["audio_file_name"] = os.path.splitext(os.path.basename(batch["audio"]["path"]))[0]
    
    return batch

def setup_directories(version: str = "v99"):
    # Create result directory and version subdirectory
    result_dir = "result"
    result_dir = os.path.join(result_dir, "task1")
    version_dir = os.path.join(result_dir, version)
    model_dir = os.path.join(version_dir, "model")
    
    # Create directories if they don't exist
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return version_dir, model_dir


# Save predictions to TSV file  VALIDATION
def save_predictions_to_tsv(predictions,filenames, version_dir,export_json_file):
    task1_output_file = os.path.join(version_dir, "task1_answer.txt")
    output_file = os.path.join(version_dir, "val_results.txt")
    print("Saving predictions to file...")
    
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for i, (pred) in enumerate(zip(predictions), 1):
    #         f.write(f"Predict: {pred}\n\n")
            
            
    # print(f"Results saved to {output_file}")
    # with open(task1_output_file, "w", encoding="utf-8", newline='\n') as f:
    #     for i, (pred,file) in enumerate(zip(predictions,filenames), 1):
    #         f.write(f"{file}\t{pred}\n")    

    json_output_file = os.path.join(version_dir, "val_time_step.json")
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(export_json_file, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {task1_output_file}")

# Save predictions to TSV file  TEST
def save_predictions_to_tsv_1(predictions, references, wer_score,cer_score, version_dir,export_json_file):
    output_file = os.path.join(version_dir, "test_results_TS.txt")
    print("Saving predictions to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Overall Word Error Rate (WER): {wer_score:.2f}%\n\n")
        f.write(f"Overall Character Error Rate (CER): {cer_score:.2f}%\n\n")
        f.write("Detailed Results:\n")
        for i, (ref, pred) in enumerate(zip(references, predictions), 1):
            f.write(f"Reference :{ref}\n")
            f.write(f"Predict   :{pred}\n\n")

    json_output_file = os.path.join(version_dir, "test_time_step.json")
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(export_json_file, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

def calculate_wer(model,test_dataloader,version_dir):
    print("Calculating WER score...")
    predictions = []
    references = []
    
    export_json_file = {}

    i = 0
    for batch in tqdm(test_dataloader, desc="處理測試資料", unit="batch"):
        with torch.no_grad():
            # Move batch to device
            audio = batch["file_path"]
            labels = batch["labels"]

            # Generate predictions
            # 3. 先做轉錄，保留原本的 segments
            result = model.transcribe(audio, batch_size=1)
            segments = result["segments"]  # 原本的句子層級 time-stamps
            transcript_dict = {
                "language": result["language"],
                "segments": []
            }
            # 4. 載入對齊模型（只做一次就好）
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device="cuda"
            )

            # 5. 做 forced-alignment，拿到 word_segments
            alignment = whisperx.align(segments, align_model, metadata, audio, device="cuda",return_char_alignments=False)

            # 6. 合併回原本 result
            result["word_segments"] = alignment["word_segments"]
            word_segments = alignment["word_segments"]

            for seg in segments:
                # 找出屬於這個 segment 的所有 word_segments
                words_in_seg = [
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "probability": w.get("probability", None)
                    }
                    for w in word_segments
                    if w["start"] >= seg["start"] and w["end"] <= seg["end"]
                ]
                transcript_dict["segments"].append({
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": words_in_seg
                })




            export_json_file[batch["audio_file_name"]] = transcript_dict
            label_str = processor.batch_decode([labels], skip_special_tokens=True)[0]

            normalizer = EnglishTextNormalizer()
            predictText = [normalizer(result["segments"][0]["text"])]
            label_str = [normalizer(label_str)]
              
            wer = 100 * metric.compute(predictions=predictText, references=label_str)
            cer = 100 * cer_metric.compute(predictions=predictText, references=label_str)
            # print("Predictions:" + str(pred_str))
            # print("Labels:" + str(label_str))
            print(f"Report {i}: WER: {wer:.2f}%")
            print(f"Report {i}: CER: {cer:.2f}%")
            print("--------------------------------------------------")
            predictions.extend(predictText)
            references.extend(label_str)
            i += 1
    
    # Calculate WER
    wer = 100 * metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)
    print(f"Word Error Rate (WER): {wer:.2f}%")
    
    save_predictions_to_tsv_1(predictions, references, wer,cer, version_dir,export_json_file)

def calculate_output(model, eval_dataloader, version_dir):
    print("Calculating ouput score...")
    predictions = []
    filenames = []
    export_json_file = {}
    
    # 4. 載入對齊模型（只做一次就好）
    align_model_zh, metadata_zh = whisperx.load_align_model(
        language_code="zh",
        device="cuda"
    )
    
    align_model_en, metadata_en = whisperx.load_align_model(
        language_code="en",
        device="cuda"
    )
    
    i = 0
    for batch in tqdm(eval_dataloader, desc="處理評估資料", unit="batch"):
        with torch.no_grad():
            
            # Move batch to device
            audio = batch["file_path"]
                    # Generate predictions
            # 3. 先做轉錄，保留原本的 segments
            result = model.transcribe(audio, batch_size=1)
            
            
            
            segments = result["segments"]  # 原本的句子層級 time-stamps
                
            # 檢查 segments 是否為空
            if not segments or len(segments) == 0:
                print(f"Warning: No segments found for audio {batch['audio_file_name']}")
                predictText = [""]  # 空字串
                export_json_file[batch["audio_file_name"]] = {
                    "language": result.get("language", "unknown"),
                    "segments": []
                }
            else:
                transcript_dict = {
                    "language": result["language"],
                    "segments": []
                }
            

            if (int(batch["audio_file_name"]) > 79999):
                alignment = whisperx.align(segments, align_model_zh, metadata_zh, audio, device="cuda",return_char_alignments=True)
            else:
                alignment = whisperx.align(segments, align_model_en, metadata_en, audio, device="cuda",return_char_alignments=False)

            # 6. 合併回原本 result
            result["word_segments"] = alignment["word_segments"]
            word_segments = alignment["word_segments"]

            for seg in segments:
                # 找出屬於這個 segment 的所有 word_segments
                words_in_seg = [
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "probability": w.get("probability", None)
                    }
                    for w in word_segments
                    if w["start"] >= seg["start"] and w["end"] <= seg["end"]
                ]
                transcript_dict["segments"].append({
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": words_in_seg
                })
            
            segments = result["segments"]  # 原本的句子層級 time-stamps       
            
            if not segments or len(segments) == 0:
                predictions.extend([""])
                filenames.extend([batch["audio_file_name"]])
            else:
                
                pred_str = result["segments"][0]["text"]
                export_json_file[batch["audio_file_name"]] = transcript_dict
            
                predictions.extend([pred_str])
                filenames.extend([batch["audio_file_name"]])  # 確保這行正確收集檔案名稱
            i += 1
            
    
    # Save predictions to file
    save_predictions_to_tsv(predictions,filenames, version_dir,export_json_file)
    
    return 0
def convert_hf_model_to_ct2(model_name_or_path: str, output_dir: str, quantization: str = "float32", trust_remote_code: bool = True, device: str = "cuda"):
    """
    將 Hugging Face 模型轉換為 CTranslate2 格式。

    參數:
        model_name_or_path (str): Hugging Face 模型的名稱或本地路徑。
        output_dir (str): 轉換後模型的輸出目錄。
        quantization (str): 量化方式，例如 "float16"、"int8" 等。
        trust_remote_code (bool): 是否信任遠端代碼，對於使用自定義代碼的模型需要設為 True。
    """
    # 初始化轉換器
    converter = TransformersConverter(model_name_or_path,copy_files=["preprocessor_config.json", "tokenizer.json"] , trust_remote_code=trust_remote_code)

    # 執行轉換
    converter.convert(output_dir=output_dir,force=True)

    model = whisperx.load_model(output_dir, device=device)

    return model

def main():
    global processor  # Make processor globally accessible for compute_metrics
    
    # Setup directories
    version = "v15_Belle_large_v3_train_凍結編碼前20層參數"
    version_dir, model_dir = setup_directories(version)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Move model to GPU if available
    model_name = "openai/whisper-large-v3"

    processor = WhisperProcessor.from_pretrained(model_name)

    # Prepare dataset
    dataset = prepare_dataset("XXXX","XXX")
    

    # Process datasets
    
    processed_dataset = {}
    for split in ["val"]:
        print(f"Processing {split} dataset...")
        processed_dataset[split] = dataset[split].map(
            lambda x: prepare_features(x, processor=processor),
            remove_columns= dataset[split].column_names,
            num_proc=1
        )
    
    # Initialize data collator
    final_model_path = model_dir
    final_ct2_model_path = os.path.join(model_dir,"..", "final_ct2")
    print(f"Model saved to {final_model_path}")


    print("Converting model to CTranslate2 format...")
    model = convert_hf_model_to_ct2(
        model_name_or_path=final_model_path,
        output_dir=final_ct2_model_path,
        # quantization="float16",
        trust_remote_code=True
    )
    # Calculate and display WER score
    # calculate_wer(model,processed_dataset["test"],version_dir)
    calculate_output(model, processed_dataset["val"], version_dir)

if __name__ == "__main__":
    main()