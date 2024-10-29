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
    MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
    OUTPUT_DIR = "./llama3-70b"
    
def fetch_answers(pipe, reports, system_prompt):
    answers = []
    for rep in tqdm(reports):
        try:
            user_content = "What is the diagnosis from this text? Please output it as a JSON object, just generate the JSON object without explanations. \n" + rep
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            outputs = pipe(
                    messages,
                    max_new_tokens=50,
                    temperature=0.001
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
            device_map="auto"
        )

    test_df = pd.read_csv(Config.TEST_SET_LOC)
    all_reports = test_df['text']

    role_content = """
    You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to extract the patient's diagnosis from the text provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
    'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
    """.strip()

    for i in range(Config.REPETITIONS):
        answers = fetch_answers(pipe, all_reports,
                                system_prompt=role_content)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        filename = f"{Config.OUTPUT_DIR}/disease-predictions-{timestamp}.csv"
        pd.DataFrame({"slno": range(len(answers)), "preds": answers}).to_csv(
            filename, index=False)
        print(f"Completed Run {i+1}")


if __name__ == "__main__":
    main()