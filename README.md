# 2025_AICUP_MedicalAndPatientVoiceSensitivePersonalDataRecognition

Competition project for "2025 AICUP - Medical and Patient Voice Sensitive Personal Data Recognition". This project aims to identify and timestamp sensitive personal data within audio recordings of medical consultations or patient interactions.

## Project Overview

The project is structured into two main tasks:

1.  **Task 1: Speech-to-Text (ASR)**
    *   Transcribes audio files into text.
    *   Utilizes a fine-tuned Whisper-based model (`BELLE-2/Belle-whisper-large-v3-zh`).
    *   The ASR model is further processed using `whisperx` to generate word-level timestamps.

2.  **Task 2: Named Entity Recognition (NER)**
    *   Identifies sensitive entities (e.g., names, dates, locations, medical record numbers) within the transcribed text.
    *   Employs a fine-tuned `Qwen/Qwen3-4B` model using LoRA for NER.
    *   Combines results from the LLM, a predefined entity dictionary ([`entities.json`](entities.json)), and regular expression patterns.
    *   Applies various filtering and cleaning rules to refine the extracted entities.
    *   Outputs the identified entities along with their start and end timestamps in the audio.

## File Descriptions

### Core Scripts

*   **[`Task1_test_train_v15_凍結編碼前20層參數_Belle.py`](Task1_test_train_v15_凍結編碼前20層參數_Belle.py):**
    *   Handles the fine-tuning of the `BELLE-2/Belle-whisper-large-v3-zh` ASR model.
    *   Freezes the first 20 encoder layers during training.
    *   Performs transcription and evaluates using WER (Word Error Rate) and CER (Character Error Rate).
    *   !!!Since whisper Normalizer itself has problems and the competition will also go through a unified Normalizer when calculating the score, this program only implements a customized simplified version of Normalizer. If you want to implement experimental scores, it is recommended to replace it with whisper Normalizer for verification and testing.!!!

*   **[`Task2_step1_HF2CT.py`](Task2_step1_HF2CT.py):**
    *   Converts the Hugging Face ASR model (likely from Task 1) to the CTranslate2 format for optimized inference.
    *   Uses `whisperx` to perform transcription and word-level alignment, generating detailed timestamp information (e.g., `result/task1/.../val_time_step.json`).

*   **[`Task2_step2_DataConvert.py`](Task2_step2_DataConvert.py):**
    *   A data conversion script that merges transcribed text (output of Task 1) with entity annotations.
    *   Prepares the data in a JSON format suitable for training the NER model.

*   **[`Task2_step3_fineture_qwen3_ner_LORA.py`](Task2_step3_fineture_qwen3_ner_LORA.py):**
    *   Fine-tunes the `Qwen/Qwen3-4B` language model for Named Entity Recognition.
    *   Utilizes LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    *   The model is trained with a specific prompt structure to extract entities and their categories.

*   **[`Task2_step4_整合.py`](Task2_step4_整合.py):**
    *   The main integration script for Task 2.
    *   Loads the fine-tuned Qwen3 NER model.
    *   Combines entity extraction results from:
        *   The Qwen3 LLM.
        *   A predefined dictionary of entities ([`entities.json`](entities.json)).
        *   Regular expression patterns.
    *   Applies filtering rules (`FILTER_RULES`) and various cleaning functions (e.g., `remove_prefix_suffix_redundant_words`, `filter_subsumed_entities`).
    *   Matches identified entities to their corresponding timestamps from the ASR output.
    *   Handles both English and Chinese text (using `opencc` for Chinese text normalization).
    *   Outputs the final list of entities with timestamps to a TSV file (e.g., `result/task2/.../task2_answer.txt`).
       
    *   ![image](https://github.com/user-attachments/assets/ff4c8df3-f9ff-4edf-9e61-fe8475189ea2)


### Key Data Files

*   **[`entities.json`](entities.json):**
    *   A JSON file containing predefined lists of entities for various categories (e.g., `DATE`, `PERSONALNAME`, `DOCTOR`, `HOSPITAL`, `ID_NUMBER`).
    *   Used by `Task2_step4_整合.py` for dictionary-based entity lookup.

*   **[`requirements.txt`](requirements.txt):**
    *   Lists all Python dependencies required to run the project.

### Output Directories

*   **`result/`**: Root directory for all output files.
    *   **`result/task1/v15_Belle_large_v3_train_凍結編碼前20層參數/`**:
        *   Contains outputs from the ASR task, including the fine-tuned Belle-Whisper model, CTranslate2 converted model (`final_ct2/`), and JSON files with word-level timestamps (`val_time_step.json`).
    *   **`result/task2/qwen3_ner_finetuned_.../`**:
        *   Stores the fine-tuned Qwen3 NER model artifacts.
    *   **`result/task2/v20_8/`**:
        *   Contains the final NER output file `task2_answer.txt`, which lists detected sensitive entities with their timestamps.

## Score

| 模型版本           | WER    | CER    |
|--------------------|--------|--------|
| v3（只訓練最後20層）   | 9.31%  | 4.44%  |
| v3-zh（只訓練最後20層） | 7.86%  | 4.17%  |
| v3-zh（全部微調）      | 8.51%  | 4.23%  |


## Setup

1.  Clone the repository.
2.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```
3.  Ensure that necessary models are downloaded or paths are configured correctly within the scripts.

## Usage

The project involves a pipeline of scripts:

1.  **Run Task 1 scripts:**
    *   Execute [`Task1_test_train_v15_凍結編碼前20層參數_Belle.py`](Task1_test_train_v15_凍結編碼前20層參數_Belle.py) to fine-tune/run the ASR model.
    *   Then, run [`Task2_step1_HF2CT.py`](Task2_step1_HF2CT.py) to convert the ASR model and generate detailed word-level timestamps. This will produce files like `val_time_step.json`.

2.  **Prepare data for Task 2 NER model:**
    *   Run [`Task2_step2_DataConvert.py`](Task2_step2_DataConvert.py) to create the training dataset for the NER model, using outputs from Task 1.

3.  **Fine-tune the Task 2 NER model:**
    *   Execute [`Task2_step3_fineture_qwen3_ner_LORA.py`](Task2_step3_fineture_qwen3_ner_LORA.py) to fine-tune the Qwen3 model for NER.

4.  **Run the final NER integration and extraction:**
    *   Execute [`Task2_step4_整合.py`](Task2_step4_整合.py) to perform NER on new transcriptions (using the `val_time_step.json` from Task 2, Step 1) and generate the final `task2_answer.txt`.

*Note: Paths for input/output files and models might need to be adjusted within the scripts based on your local setup.*
