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
    SUMMARY_TEST_SET_LOC = "../../extras/test-summarization/test-summarization-2024-09-26_20-41-24.csv"
    TRAIN_SET_LOC = "../../data/train.csv"
    VAL_SET_LOC = "../../data/val.csv"
    SUMMARY_TRAIN_SET_LOC = "../../extras/train-summarization/train-summarization-2024-09-26_20-06-20.csv"
    API_MODEL = "gpt-4o-mini-2024-07-18"
    OUTPUT_DIR = "./gpt-4o-mini"
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


def create_fewShot_prompt(disease_name, train_df, max_samples=4):
    a_df = train_df[train_df['type_name'] == disease_name].groupby('survival_over_mean').apply(
        lambda x: x.sample(min(max_samples, len(x)))).reset_index(drop=True)

    shot_data = zip(a_df['text_summarized'],
                    a_df['Survival_times'], a_df['survival_over_mean'])

    prompt_template = ('Can you determine if the patient will survive after {years} years '
                       'from the following Pathology Report?\n{summary}\nOptions - \n'
                       '(A) True \n(B) False \n \nAnswer - {{"Survival": "{category}"}}\n\n')

    shots = ''.join([prompt_template.format(years=years, summary=summary, category=category)
                     for summary, years, category in shot_data])

    return shots


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


def fetch_answers(client, reports, mean_times, system_prompt):
    answers = []
    for i in tqdm(range(len(reports))):
        try:
            user_content = f"Can you determine if the patient will survive after {mean_times[i]} years from the following Pathology Report?\n" + \
                reports[i] + "\nOptions - \n(A) True \n(B) False \n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            chat_response = client.chat.completions.create(
                model=Config.API_MODEL, messages=messages, max_tokens=500, temperature=0.001)
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
        all_reports = list(disease_df['text_summarized'])
        mean_times = list(disease_df['Survival_times'])

        role_content = f"""
You are an expert medical pathology AI assistant. You are provided with a question whether a patient will surivive after a particular given time or not along with the patient's pathology report with multiple answer choices.
Your goal is to think through the question carefully and explain your reasoning step by step before selecting the final answer as a JSON.
Respond only with the reasoning steps and answer as specified below.
When answering user questions follow these examples- \n
{create_fewShot_prompt(disease_name=disease_type_name, train_df=train_df)}
""".strip()

        for i in range(Config.REPETITIONS):
            answers = fetch_answers(client, all_reports, mean_times,
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
