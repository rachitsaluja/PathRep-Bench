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
    API_MODEL = "gpt-4o-2024-08-06"
    OUTPUT_DIR = "./gpt-4o"
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

    @staticmethod
    def load_env():
        load_dotenv(Config.ENV_LOC)
        return os.getenv("OPENAI_API_KEY")


def fetch_answers(client, reports, system_prompt):
    answers = []
    for rep in tqdm(reports):
        try:
            user_content = "Can you identify the AJCC Stage of the Cancer from this text? Take a step-by-step approach in your response, cite sources and give reasoning before sharing final answer in the format -  {stage : ANSWER} \n" + rep
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
    test_df = pd.read_csv(Config.TEST_SET_LOC)[
        ['text', 'type_name', 'stage_overall']].dropna()
    disease_list = Config.DISEASE_LIST

    for d in disease_list:
        disease_type_name = d
        disease_df = test_df[test_df['type_name'] ==
                             disease_type_name].reset_index(drop=False)
        all_reports = list(disease_df['text'])

        role_content = f"""
        You are a highly knowledgeable and intelligent pathology AI assistant. Your main responsibility is to understand the pathology report and identify the AJCC stage from the information provided in the report. This report is of a {disease_type_name} patient. You will only output it as a JSON Object. Here are the set of options - 
        'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail. You as an assistant cannot lie.
        """.strip()

        for i in range(Config.REPETITIONS):
            answers = fetch_answers(client, all_reports,
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
