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
    

def fetch_answers(model, tokenizer, reports, system_prompt):
    answers = []
    for rep in tqdm(reports):
        try:
            user_content = "Can you identify the AJCC Stage of the Cancer from the following Pathology Report? Please output it as a JSON object. \n" + rep + "\n\n Options - \n (A) Stage I \n (B) Stage II \n (C) Stage III \n (D) Stage IV \n"
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
    
    test_df = pd.read_csv(Config.TEST_SET_LOC)[
        ['text', 'type_name', 'stage_overall']].dropna()
    disease_list = Config.DISEASE_LIST
    
    for d in disease_list:
        disease_type_name = d
        print(f"Processing Disease - {d}")
        disease_df = test_df[test_df['type_name'] ==
                             disease_type_name].reset_index(drop=False)
        all_reports = list(disease_df['text'])

        role_content = """
You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to understand the patholgy report and extract the stage of the cancer from the pathology report provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip() 

        for i in range(Config.REPETITIONS):
            answers = fetch_answers(model, tokenizer, all_reports,
                                    system_prompt=role_content)
            
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            filename = f"{Config.OUTPUT_DIR}/{disease_type_name}-stage-predictions-{timestamp}.csv"
            pd.DataFrame({"slno": range(len(answers)), "preds": answers}).to_csv(
                filename, index=False)
            print(f"Completed Run {i+1}")
            

if __name__ == "__main__":
    main()