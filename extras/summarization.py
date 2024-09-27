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
    REPETITIONS = 1
    ENV_LOC = "../.env"
    TEST_SET_LOC = "../data/test.csv"
    API_MODEL = "gpt-4o-mini-2024-07-18"
    OUTPUT_DIR = "./test-summarization"

    @staticmethod
    def load_env():
        load_dotenv(Config.ENV_LOC)
        return os.getenv("OPENAI_API_KEY")


def fetch_answers(client, reports, system_prompt):
    answers = []
    for rep in tqdm(reports):
        try:
            user_content = "Summarize the following Pathology Report. \n" + rep
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
    test_df = pd.read_csv(Config.TEST_SET_LOC)
    all_reports = test_df['text']

    role_content = """You are an expert medical pathology AI assistant. You are provided with a pathology report and you main responsibility is to summarize the report in 3 sentences. 
You will always be truthful and will not say anything this is factually false. You will only provide the summary and nothing else. 
""".strip()

    for i in range(Config.REPETITIONS):
        answers = fetch_answers(client, all_reports,
                                system_prompt=role_content)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        filename = f"{Config.OUTPUT_DIR}/test-summarization-{timestamp}.csv"
        pd.DataFrame({"slno": range(len(answers)), "preds": answers}).to_csv(
            filename, index=False)
        print(f"Completed Run {i+1}")


if __name__ == "__main__":
    main()
