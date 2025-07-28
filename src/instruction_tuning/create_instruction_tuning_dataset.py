import pandas as pd
import numpy as np
import random
import json

train_df = pd.read_csv("../../../data/train.csv")
val_df = pd.read_csv("../../../data/val.csv")

# Creating Disease Instructions

system_disease_prompt1 = """
You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to extract the patient's diagnosis from the text provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
""".strip()

system_disease_prompt2 = """
You are a smart pathology AI assistant. Your key responsibility is to deduce the patient's diagnosis from the supplied text and deliver a response solely from the given set of options, outputting only in the form of JSON Objects. Here are the set of options - 
'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
""".strip()

system_disease_prompt3 = """
You are a highly skilled and intelligent pathology AI assistant, it is your main task to interpret the patient's diagnosis from the text provided and limit your answers to a predefined set of options. Responses should be exclusively in JSON Object format. Here are the set of options - 
'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
""".strip()

system_disease_prompt4 = """
As an advanced and intelligent pathology AI assistant, your primary duty is to analyze the patient's diagnosis from the provided text and respond exclusively from a predetermined set of options. Your responses must be formatted strictly as JSON Objects. Here are the set of options - 
'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
""".strip()

system_disease_prompt5 = """
You are a highly capable and intelligent pathology AI assistant, your principal obligation is to extract the diagnosis from the patient's provided text and respond only from a selected set of options, with all outputs formatted as JSON Objects. Here are the set of options - 
'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie. 
""".strip()


system_disease_prompt_list = [
    system_disease_prompt1,
    system_disease_prompt2,
    system_disease_prompt3,
    system_disease_prompt4,
    system_disease_prompt5
]

path_reports = list(train_df['text'])
disease_type_name = list(train_df['type_name'])

disease_type_name_json = []
for item in disease_type_name:
    formatted_string = '{\n "diagnosis": "' + item + '" \n}'
    disease_type_name_json.append(formatted_string)
    
disease_instruction_prompts = [
    "What is the diagnosis from this text? Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Can you determine the diagnosis from the following text? Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Can you extract the diagnosis from the text provided? Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Identify the diagnosis from the following text. Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Determine the diagnosis from this text. Please output it as a JSON object, just generate the JSON object without explanations. \n\n"
]

disease_instruction_list = []
for i in range(len(path_reports)):
    instruction_dict = {}
    sys_prompt = random.sample(system_disease_prompt_list, k=1)[0]
    user_prompt = random.sample(disease_instruction_prompts, k=1)[0].strip()
    assistant_output = disease_type_name_json[i]
    
    instruction_dict["system_prompt"] = sys_prompt
    instruction_dict["user_prompt"] = user_prompt + "\n" + path_reports[i]
    instruction_dict["assistant_output"] = assistant_output
    disease_instruction_list.append(instruction_dict)    
    
train_disease_df = pd.DataFrame(disease_instruction_list)

path_reports = list(val_df['text'])
disease_type_name = list(val_df['type_name'])

disease_type_name_json = []
for item in disease_type_name:
    formatted_string = '{\n "diagnosis": "' + item + '" \n}'
    disease_type_name_json.append(formatted_string)
    
disease_instruction_list = []
for i in range(len(path_reports)):
    instruction_dict = {}
    sys_prompt = random.sample(system_disease_prompt_list, k=1)[0]
    user_prompt = random.sample(disease_instruction_prompts, k=1)[0].strip()
    assistant_output = disease_type_name_json[i]
    
    instruction_dict["system_prompt"] = sys_prompt
    instruction_dict["user_prompt"] = user_prompt + "\n" + path_reports[i]
    instruction_dict["assistant_output"] = assistant_output
    disease_instruction_list.append(instruction_dict)    
    
    
val_disease_df = pd.DataFrame(disease_instruction_list)


# Creating Stage Instructions

system_stage_prompt1 = """
You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to understand the patholgy report and extract the stage of the cancer from the pathology report provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()

system_stage_prompt2 = """
You are a deeply knowledgeable and intelligent pathology AI assistant, your primary task is to comprehend the pathology report, identify the cancer stage from the given report, and deliver responses strictly from a predefined set of options. Your outputs must be exclusively in JSON Object format. Here are the set of options - 
'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()

system_stage_prompt3 = """
You are a highly informed and intelligent pathology AI assistant, your key job is to process the pathology report, ascertain the stage of the cancer, and provide responses only from a certain set of options, exclusively outputting in JSON Object format. Here are the set of options - 
'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()

system_stage_prompt4 = """
You are a proficient and intelligent pathology AI assistant, you role primarily is to interpret the pathology report, extract the stage of the cancer detailed within, and reply using only a specified set of options. All responses must be formatted as JSON Objects and nothing else. Here are the set of options - 
'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()

system_stage_prompt5 = """
You are an expert and highly intelligent pathology AI assistant. Your chief responsibility involves interpreting the pathology report to determine the cancer stage and providing answers only from an established set of options, with responses formatted solely as JSON Objects. Here are the set of options - 
'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()


system_stage_prompt_list = [
    system_stage_prompt1,
    system_stage_prompt2,
    system_stage_prompt3,
    system_stage_prompt4,
    system_stage_prompt5
]

train_stage_df = train_df[['text', 'stage_overall']].dropna()
val_stage_df = val_df[['text', 'stage_overall']].dropna()


path_reports = list(train_stage_df['text'])
stage_name = list(train_stage_df['stage_overall'])

stage_name_json = []
for item in stage_name:
    formatted_string = '{\n "stage": "' + item + '" \n}'
    stage_name_json.append(formatted_string)
    
stage_instruction_prompts = [
    "Can you identify the AJCC Stage of the Cancer from the following Pathology Report? Please output it as a JSON object"
    "What is the AJCC Stage of the cancer from this text? Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Can you determine the AJCC Stage of the cancer from the following text? Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Can you extract the AJCC Stage of the cancer from the text provided? Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Identify the AJCC stage of the cancer based on the information in this text. Please output it as a JSON object, just generate the JSON object without explanations. \n\n",
    "Extract the AJCC cancer stage from the text provided. Please output it as a JSON object, just generate the JSON object without explanations. \n\n ",
    "Determine the AJCC stage of the cancer from this text. Please output it as a JSON object, just generate the JSON object without explanations. \n\n"
]

stage_instruction_list = []
for i in range(len(path_reports)):
    instruction_dict = {}
    sys_prompt = random.sample(system_stage_prompt_list, k=1)[0]
    user_prompt = random.sample(stage_instruction_prompts, k=1)[0].strip()
    assistant_output = stage_name_json[i]
    
    instruction_dict["system_prompt"] = sys_prompt
    instruction_dict["user_prompt"] = user_prompt + "\n" + path_reports[i] + "\n\n Options - \n (A) Stage I \n (B) Stage II \n (C) Stage III \n (D) Stage IV \n"
    instruction_dict["assistant_output"] = assistant_output
    stage_instruction_list.append(instruction_dict) 
    
    
train_stage_df = pd.DataFrame(stage_instruction_list)

path_reports = list(val_stage_df['text'])
stage_name = list(val_stage_df['stage_overall'])

stage_name_json = []
for item in stage_name:
    formatted_string = '{\n "stage": "' + item + '" \n}'
    stage_name_json.append(formatted_string)
    
stage_instruction_list = []
for i in range(len(path_reports)):
    instruction_dict = {}
    sys_prompt = random.sample(system_stage_prompt_list, k=1)[0]
    user_prompt = random.sample(stage_instruction_prompts, k=1)[0].strip()
    assistant_output = stage_name_json[i]
    
    instruction_dict["system_prompt"] = sys_prompt
    instruction_dict["user_prompt"] = user_prompt + "\n" + path_reports[i] + "\n\n Options - \n (A) Stage I \n (B) Stage II \n (C) Stage III \n (D) Stage IV \n"
    instruction_dict["assistant_output"] = assistant_output
    stage_instruction_list.append(instruction_dict) 
    
    
val_stage_df = pd.DataFrame(stage_instruction_list)


# Creating Prognosis Instructions

system_prognosis_prompt1 = """
You are an expert medical pathology AI assistant. You are provided with a question whether a patient will surivive after a particular given time or not along with the patient's pathology report with multiple answer choices. You will only output it as a JSON Object and nothing else.  You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
""".strip()

system_prognosis_prompt2 = """
You are an AI assistant specialized in medical pathology. You are tasked with determining whether a patient will survive after a specified period, based on their pathology report and a set of multiple-choice answers. Your response must be strictly a JSON Object, and no other output is allowed. You must select one of the provided options without exception, and you are not permitted to fabricate information.
""".strip()

system_prognosis_prompt3 = """
As a medical pathology AI assistant, you are given a pathology report and asked to assess whether a patient will survive after a certain time, using a set of provided answer choices. You will respond solely in JSON format and must choose one of the given options. You cannot generate any other text, and you must always provide truthful answers.
""".strip()

system_prognosis_prompt4 = """
You are an AI assistant in the field of medical pathology. Your task is to determine whether a patient will survive after a specific time, based on the pathology report and a list of answer options. Your output must be a JSON Object, and no other text is allowed. You must choose one of the answers and cannot lie.
""".strip()

system_prognosis_prompt5 = """
You serve as a medical pathology AI assistant. Your role is to evaluate the survival of a patient after a given time, relying on their pathology report and a set of possible answers. Your output will be limited to a JSON Object only, and you must select one of the provided choices without deviation. As an assistant, you are bound to give truthful and accurate responses.
""".strip()

system_prognosis_prompt6 = """
You are a dedicated AI assistant specializing in medical pathology. You are presented with a question about a patient's survival after a specific timeframe and their pathology report, along with multiple answer options. You must respond only with a JSON Object, choosing one of the given answers, and you are prohibited from producing any other text or giving false information.
""".strip()


system_prognosis_prompt_list = [
    system_prognosis_prompt1,
    system_prognosis_prompt2,
    system_prognosis_prompt3,
    system_prognosis_prompt4,
    system_prognosis_prompt5,
    system_prognosis_prompt6
]

def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    df = df[['text', 'stage_overall', 'type_name',
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



disease_list = ['Adrenocortical carcinoma',
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



train_full_df = pd.read_csv('../../../data/train.csv')
val_df = pd.read_csv('../../../data/val.csv')
train_full_df['DSS.time'] = train_full_df['DSS.time'].astype(float)
val_df['DSS.time'] = val_df['DSS.time'].astype(float)
full_df = pd.concat([train_full_df, val_df])


train_df = load_and_prepare_data(
    '../../../data/train.csv')

val_df = load_and_prepare_data(
    '../../../data/val.csv')


val_df = val_df[val_df['type_name'].isin(disease_list)]


disease_times = [np.round((np.mean(full_df[full_df['type_name'] == disease]['DSS.time']) / 365), 2)
                 for disease in disease_list]

train_df = add_survival_info(train_df, disease_times, disease_list)
val_df = add_survival_info(val_df, disease_times, disease_list)


train_prognosis_df = train_df[['text', 'Survival_times','survival_over_mean']].dropna()
val_prognosis_df = val_df[['text', 'Survival_times','survival_over_mean']].dropna()


path_reports = list(train_prognosis_df['text'])
prog_name = list(train_prognosis_df['survival_over_mean'])
prog_time = list(train_prognosis_df['Survival_times'])

prog_name_json = []
for item in prog_name:
    formatted_string = '{\n "survival": "' + item + '" \n}'
    prog_name_json.append(formatted_string)
    
    
prognosis_instruction_prompts = [
    f"Can you assess if the patient will survive beyond {{}} years based on the following pathology report?\n",
    f"Is it possible to determine if the patient will survive after {{}} years from the provided pathology report?\n",
    f"Can you evaluate the likelihood of the patient surviving past {{}} years from this pathology report?\n",
    f"Based on the following pathology report, can you predict if the patient will survive for more than {{}} years?\n",
    f"Can you analyze the pathology report and determine if the patient is expected to survive beyond {{}} years?\n"
]

prog_instruction_list = []
for i in range(len(path_reports)):
    instruction_dict = {}
    sys_prompt = random.sample(system_prognosis_prompt_list, k=1)[0]
    user_prompt = random.sample(prognosis_instruction_prompts, k=1)[0].strip()
    assistant_output = prog_name_json[i]
    
    instruction_dict["system_prompt"] = sys_prompt
    instruction_dict["user_prompt"] = user_prompt.format(prog_time[i])+ "\n" + path_reports[i] + "\nOptions - \n(A) True \n(B) False \n"
    instruction_dict["assistant_output"] = assistant_output
    prog_instruction_list.append(instruction_dict) 
    
train_prognosis_df = pd.DataFrame(prog_instruction_list)


path_reports = list(val_prognosis_df['text'])
prog_name = list(val_prognosis_df['survival_over_mean'])
prog_time = list(val_prognosis_df['Survival_times'])

prog_name_json = []
for item in prog_name:
    formatted_string = '{\n "survival": "' + item + '" \n}'
    prog_name_json.append(formatted_string)
    
prog_instruction_list = []
for i in range(len(path_reports)):
    instruction_dict = {}
    sys_prompt = random.sample(system_prognosis_prompt_list, k=1)[0]
    user_prompt = random.sample(prognosis_instruction_prompts, k=1)[0].strip()
    assistant_output = prog_name_json[i]
    
    instruction_dict["system_prompt"] = sys_prompt
    instruction_dict["user_prompt"] = user_prompt.format(prog_time[i])+ "\n" + path_reports[i] + "\nOptions - \n(A) True \n(B) False \n"
    instruction_dict["assistant_output"] = assistant_output
    prog_instruction_list.append(instruction_dict) 
    
val_prognosis_df = pd.DataFrame(prog_instruction_list)


train_disease_df.to_csv(
    "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/train_disease.csv", index=False)
train_stage_df.to_csv(
    "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/train_stage.csv", index=False)
train_prognosis_df.to_csv(
    "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/train_prognosis.csv", index=False)


val_disease_df.to_csv(
    "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/val_disease.csv", index=False)
val_stage_df.to_csv(
    "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/val_stage.csv", index=False)
val_prognosis_df.to_csv(
    "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/val_prognosis.csv", index=False)