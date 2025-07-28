import os
import contextlib
import os
import json

with contextlib.redirect_stdout(open(os.devnull, "w")):
    from unsloth import FastLanguageModel


class DataExtractor:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.MAX_SEQ_LEN = 4096
        self.MODEL_ID = "rsjx/pathllama-3.1-8b"

        with contextlib.redirect_stdout(open(os.devnull, "w")):
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.MODEL_ID,
                max_seq_length=self.MAX_SEQ_LEN,
                load_in_4bit=True,
                dtype=None,
            )

        self.model = FastLanguageModel.for_inference(self.model)

        self.system_prompt_diag = """
        You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to extract the patient's diagnosis from the text provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
        'Adrenocortical carcinoma','Bladder Urothelial Carcinoma','Brain Lower Grade Glioma','Breast invasive carcinoma','Cervical squamous cell carcinoma and endocervical adenocarcinoma','Cholangiocarcinoma','Colon adenocarcinoma','Esophageal carcinoma','Glioblastoma multiforme','Head and Neck squamous cell carcinoma','Kidney Chromophobe','Kidney renal clear cell carcinoma','Kidney renal papillary cell carcinoma','Liver hepatocellular carcinoma','Lung adenocarcinoma','Lung squamous cell carcinoma','Lymphoid Neoplasm Diffuse Large B-cell Lymphoma','Mesothelioma','Ovarian serous cystadenocarcinoma','Pancreatic adenocarcinoma','Pheochromocytoma and Paraganglioma','Prostate adenocarcinoma','Rectum adenocarcinoma','Sarcoma','Skin Cutaneous Melanoma','Stomach adenocarcinoma','Testicular Germ Cell Tumors','Thymoma','Thyroid carcinoma','Uterine Carcinosarcoma','Uterine Corpus Endometrial Carcinoma','Uveal Melanoma'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
        """.strip()

        self.system_prompt_ajcc = """
        You are highly knowledgeable and intelligent pathology AI assistant. Your main responsibilty is to understand the patholgy report and extract the stage of the cancer from the pathology report provided and give an answer only from the set of options. You will only output it as a JSON Object and nothing else. Here are the set of options - 
        'Stage I','Stage II','Stage III', 'Stage IV'. You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
        """.strip()

        self.system_prompt_prog = """
        You are an expert medical pathology AI assistant. You are provided with a question whether a patient will surivive after a particular given time or not along with the patient's pathology report with multiple answer choices. You will only output it as a JSON Object and nothing else.  You have to pick one these options without fail and you cannot print any other text. You as an assistant cannot lie.
        """.strip()

        self.survival_time_dict = {
            'Adrenocortical carcinoma': '3.98',
            'Bladder Urothelial Carcinoma': '2.24',
            'Breast invasive carcinoma': '3.49',
            'Cholangiocarcinoma': '2.35',
            'Colon adenocarcinoma': '2.2',
            'Esophageal carcinoma': '1.59',
            'Head and Neck squamous cell carcinoma': '2.53',
            'Kidney Chromophobe': '4.85',
            'Kidney renal clear cell carcinoma': '3.55',
            'Kidney renal papillary cell carcinoma': '3.03',
            'Liver hepatocellular carcinoma': '2.37',
            'Lung adenocarcinoma': '2.48',
            'Lung squamous cell carcinoma': '2.62',
            'Mesothelioma': '1.78',
            'Pancreatic adenocarcinoma': '1.59',
            'Rectum adenocarcinoma': '2.1',
            'Skin Cutaneous Melanoma': '1.43',
            'Stomach adenocarcinoma': '1.54',
            'Testicular Germ Cell Tumors': '3.82',
            'Thyroid carcinoma': '3.43',
            'Uveal Melanoma': '2.11'
        }

    def extract_CancerType(self, report: str) -> str:
        user_content = (
            "What is the diagnosis from this text? Please output it as a JSON object, "
            "just generate the JSON object without explanations.\n" + report
        )

        messages = [
            {"role": "system", "content": self.system_prompt_diag},
            {"role": "user", "content": user_content}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            use_cache=True
        )

        chat_response = self.tokenizer.batch_decode(output)[0].split(
            '|im_start|>assistant\n')[-1].split('<|im_end|>')[0]

        return json.loads(chat_response)

    def extract_AJCCstage(self, report: str) -> str:
        user_content = (
            "Can you identify the AJCC Stage of the Cancer from the following Pathology Report?"
            "Please output it as a JSON object. \n" + report +
            "\n\n Options - \n (A) Stage I \n (B) Stage II \n (C) Stage III \n (D) Stage IV \n"
        )

        messages = [
            {"role": "system", "content": self.system_prompt_ajcc},
            {"role": "user", "content": user_content}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            use_cache=True
        )

        chat_response = self.tokenizer.batch_decode(output)[0].split(
            '|im_start|>assistant\n')[-1].split('<|im_end|>')[0]

        return json.loads(chat_response)

    def extract_Prognosis(self, report: str, cancer_type: str) -> str:

        if cancer_type not in self.survival_time_dict:
            raise ValueError(f"[InputError] Unknown cancer type: '{cancer_type}'. Please provide a valid cancer type from the following list. Adrenocortical carcinoma', 'Bladder Urothelial Carcinoma', 'Breast invasive carcinoma', 'Cholangiocarcinoma', 'Colon adenocarcinoma', 'Esophageal carcinoma', 'Head and Neck squamous cell carcinoma', 'Kidney Chromophobe', 'Kidney renal clear cell carcinoma', 'Kidney renal papillary cell carcinoma', 'Liver hepatocellular carcinoma', 'Lung adenocarcinoma', 'Lung squamous cell carcinoma', 'Mesothelioma', 'Pancreatic adenocarcinoma', 'Rectum adenocarcinoma', 'Skin Cutaneous Melanoma', 'Stomach adenocarcinoma', 'Testicular Germ Cell Tumors', 'Thyroid carcinoma', 'Uveal Melanoma'")

        mean_time = self.survival_time_dict[cancer_type]
        user_content = (
            f"Can you analyze the pathology report and determine if the patient is expected to survive beyond {mean_time}?\n" +
            report + "\nOptions - \n(A) True \n(B) False \n"
        )

        messages = [
            {"role": "system", "content": self.system_prompt_ajcc},
            {"role": "user", "content": user_content}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            use_cache=True
        )

        chat_response = self.tokenizer.batch_decode(output)[0].split(
            '|im_start|>assistant\n')[-1].split('<|im_end|>')[0]

        print(f"The patient's survival after {mean_time} years - \n")
        print(chat_response)

        return json.loads(chat_response)
