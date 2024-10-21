import os
import re
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob2 import glob
from natsort import natsorted
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI


class Config:
    REPETITIONS = 5
    ENV_LOC = "../../.env"
    TEST_SET_LOC = "../../data/test.csv"
    API_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::AJY5i5Q4"
    OUTPUT_DIR = "./gpt-4o-mini-ft"

    @staticmethod
    def load_env():
        load_dotenv(Config.ENV_LOC)
        return os.getenv("OPENAI_API_KEY")


def fetch_answers(client, reports, system_prompt):
    answers = []
    for rep in tqdm(reports):
        try:
            user_content = "What is the diagnosis from this text? Please output it as a JSON object, just generate the JSON object without explanations. \n" + rep
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            chat_response = client.chat.completions.create(
                model=Config.API_MODEL, messages=messages, max_tokens=50, temperature=0.001)
            answers.append(chat_response.choices[0].message.content)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing report: {e}")
            answers.append("NOT COMPLETED")
    return answers


def main():
    api_key = Config.load_env()
    client = OpenAI(api_key=api_key)
    test_df = pd.read_csv(Config.TEST_SET_LOC)
    all_reports = test_df['text']

    role_content = """
    You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to extract the patient's diagnosis from the text provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
    'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
    """.strip()

    for i in range(Config.REPETITIONS):
        answers = fetch_answers(client, all_reports,
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