import os
import re
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob2 import glob
from natsort import natsorted
from datetime import datetime
import time
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

class Config:
    REPETITIONS = 5
    ENV_LOC = "../../.env"
    TEST_SET_LOC = "../../data/test.csv"
    TRAIN_SET_LOC = "../../data/train.csv"
    VAL_SET_LOC = "../../data/val.csv"
    SUMMARY_TEST_SET_LOC = "../../extras/test-summarization/test-summarization-2024-09-26_20-41-24.csv"
    SUMMARY_TRAIN_SET_LOC = "../../extras/train-summarization/train-summarization-2024-09-26_20-06-20.csv"
    MODEL_ID = "/midtier/sablab/scratch/ras4037/workspace/LLM_path/fine-tuned-models/llama3.1_pathrep_v1.0-merged"
    OUTPUT_DIR = "./pathllama3.1-8b"
    MAX_SEQ_LEN = 4096
    DISEASE_LIST = ['Adrenocortical carcinoma',
                'Bladder Urothelial Carcinoma',
                'Breast invasive carcinoma',
                'Cholangiocarcinoma',
                'Colon adenocarcinoma',
                'Esophageal carcinoma',
                'Head and Neck squamous cell carcinoma',
                'Kidney Chromophobe',
                'Kidney renal clear cell carcinoma',
                'Kidney renal papillary cell carcinoma',
                'Liver hepatocellular carcinoma',
                'Lung adenocarcinoma',
                'Lung squamous cell carcinoma',
                'Mesothelioma',
                'Pancreatic adenocarcinoma',
                'Rectum adenocarcinoma',
                'Skin Cutaneous Melanoma',
                'Stomach adenocarcinoma',
                'Testicular Germ Cell Tumors',
                'Thyroid carcinoma',
                'Uveal Melanoma']
    
def load_and_prepare_data(data_path, summary_path):
    df = pd.read_csv(data_path)
    summarization_df = pd.read_csv(summary_path)
    df['text_summarized'] = summarization_df['preds']
    df = df[['text', 'text_summarized', 'stage_overall', 'type_name',
             'age_at_initial_pathologic_diagnosis', 'gender', 'DSS.time']].dropna().reset_index(drop=True)
    df['DSS.time'] = df['DSS.time'].astype(float)
    df['DSS.time'] = np.round(df['DSS.time'] / 365, 3)
    return df


def add_survival_info(df, disease_times, disease_list):
    survival_times = [disease_times[disease_list.index(
        df['type_name'].iloc[i])] for i in range(len(df))]
    df['Survival_times'] = survival_times
    df['survival_over_mean'] = (
        df['DSS.time'] > df['Survival_times']).astype(str)
    return df
    
def fetch_answers(model, tokenizer, reports, mean_times, system_prompt):
    answers = []
    for i in tqdm(range(len(reports))):
        try:
            user_content = f"Can you analyze the pathology report and determine if the patient is expected to survive beyond {mean_times[i]}?\n" + \
                reports[i] + "\nOptions - \n(A) True \n(B) False \n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            
            output = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
            chat_response = tokenizer.batch_decode(output)[0].split('|im_start|>assistant\n')[-1].split('<|im_end|>')[0]
            answers.append(chat_response)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing report: {e}")
            answers.append("NOT COMPLETED")
    return answers

def main():
    max_seq_length = Config.MAX_SEQ_LEN
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_ID,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.for_inference(model)
    
    test_df = load_and_prepare_data(
        Config.TEST_SET_LOC, Config.SUMMARY_TEST_SET_LOC)
    train_df = load_and_prepare_data(
        Config.TRAIN_SET_LOC, Config.SUMMARY_TRAIN_SET_LOC)

    train_full_df = pd.read_csv(Config.TRAIN_SET_LOC)
    val_df = pd.read_csv(Config.VAL_SET_LOC)
    train_full_df['DSS.time'] = train_full_df['DSS.time'].astype(float)
    val_df['DSS.time'] = val_df['DSS.time'].astype(float)
    full_df = pd.concat([train_full_df, val_df])

    disease_list = Config.DISEASE_LIST
    disease_times = [np.round((np.mean(full_df[full_df['type_name'] == disease]['DSS.time']) / 365), 2)
                     for disease in disease_list]

    test_df = add_survival_info(test_df, disease_times, disease_list)
    train_df = add_survival_info(train_df, disease_times, disease_list)
    
    for d in disease_list:
        disease_type_name = d
        print(f"Processing Disease - {d}")
        disease_df = test_df[test_df['type_name'] ==
                             disease_type_name].reset_index(drop=False)
        all_reports = list(disease_df['text'])
        mean_times = list(disease_df['Survival_times'])

        role_content = f"""
You are an expert medical pathology AI assistant. You are provided with a question whether a patient will surivive after a particular given time or not along with the patient's pathology report with multiple answer choices. You will only output it as a JSON Object and nothing else.  You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()

        for i in range(Config.REPETITIONS):
            answers = fetch_answers(model, tokenizer, all_reports, mean_times,
                                    system_prompt=role_content)
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            filename = f"{Config.OUTPUT_DIR}/{disease_type_name}-prognosis-predictions-{timestamp}.csv"
            pd.DataFrame({"slno": range(len(answers)), "preds": answers}).to_csv(
                filename, index=False)
            print(f"Completed Run {i+1}")


if __name__ == "__main__":
    main()