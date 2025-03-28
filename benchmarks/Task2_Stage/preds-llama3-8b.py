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
from transformers import pipeline
import torch

class Config:
    REPETITIONS = 5
    ENV_LOC = "../../.env"
    TEST_SET_LOC = "../../data/test.csv"
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    OUTPUT_DIR = "./llama3.1-8b"
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


def fetch_answers(pipe, reports, system_prompt):
    answers = []
    for rep in tqdm(reports):
        try:
            user_content = "Can you identify the AJCC Stage of the Cancer from the following Pathology Report?\n" + rep + "\n\n Options - \n (A) Stage I \n (B) Stage II \n (C) Stage III \n (D) Stage IV \n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            outputs = pipe(
                    messages,
                    max_new_tokens=500,
                    temperature=0.001,
            )
            chat_response = outputs[0]["generated_text"][-1]["content"]
            answers.append(chat_response)
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing report: {e}")
            answers.append("NOT COMPLETED")
    return answers

def main():
    pipe = pipeline(
            "text-generation",
            model=Config.MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
    
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
You are an expert medical pathology AI assistant. You are provided with a question about which stage of cancer does the patient have along with the patient's pathology report with multiple answer choices.
Your goal is to think through the question carefully and explain your reasoning step by step before selecting the final answer as a JSON.
Respond only with the reasoning steps and answer as specified below.
Below is the format for each question and answer:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
(model generated chain of thought explanation)
Therefore, the answer is - 
{"stage" : ANSWER (e.g. Stage I, Stage II, Stage III, Stage IV)}
""".strip() 

        for i in range(Config.REPETITIONS):
            answers = fetch_answers(pipe, all_reports,
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