# PathRep-Bench

Official implementation of [Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs](https://www.nature.com/articles/s41598-025-10709-4)

## Features

This repository provides a benchmarking framework for evaluating large language models (LLMs) on pathology report analysis, specifically for:
- Cancer Type Identification: Determining the type of cancer from pathology reports.
- AJCC Cancer Staging: Identifying cancer stage (Stage I, Stage II, Stage III, and Stage IV) based on pathology reports.
- Prognosis Prediction: Assessing whether a patient has a "good" or "bad" prognosis from pathology reports.

We also provide -

- Full Dataset: The dataset used in our study is publicly available [here](https://huggingface.co/datasets/rosenthal/tcga-path-notes) 🤗.
- Pretrained Model for Inference: We provide an instruction-tuned model that can be used for inference on custom pathology reports via a simple Python API, [model](https://huggingface.co/rsjx/pathllama-3.1-8b) 🤗. 
- Fine-Tuning Code: Scripts for generating instruction-tuning data and training your own fine-tuned model.
- Google Colab Implementation: A ready-to-use Google Colab notebook for easy experimentation | [![Explore UniverSeg in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/NEEDTODO)<br>

> ⚠️ Note-1: Our model and evaluations are based on a specific set of cancer types. Please keep this in mind when using the model, as your pathology reports may contain cancer types that were not included in our analysis.

> ⚠️⚠️ Note-2: The outputs of this model are LLM generated, please don't use it you are unwell. Visit a doctor to get the right information. This project is for data mining.   

## Running Inference

We have provided a simple python inference API to extract the Cancer Type, AJCC Stage and the Prognosis. It will automatically download the model shards and will be able to run inference. 

To get started, first, clone the repository:

```bash
git clone https://github.com/rachitsaluja/PathRep-Bench.git
```

Create a conda environment. Here it is called Path-LLM.

```bash
conda create -n Path-LLM python=3.11 -y
```

Then initialize the environment and start the installation of the other libraries

```bash
conda activate Path-LLM

# Go into Repo
cd PathRep-Bench

# Installation
pip install -r requirements.txt
```

Once complete, you can run the API via the following commands - 

### For Cancer Type

```python
import sys
sys.path.append("loc/to/PathRep-Bench")

from src.inference.DataExtractor import DataExtractor

report = "Pathology Report Contents ......"

extractor = DataExtractor()
report_text = report
response = extractor.extract_CancerType(report_text)
```

The output is a dictionary, like the following

```
{'diagnosis': 'Colon adenocarcinoma'}
```

### For AJCC Stage

```python
import sys
sys.path.append("loc/to/PathRep-Bench")

from src.inference.DataExtractor import DataExtractor

report = "Pathology Report Contents ......"

extractor = DataExtractor()
report_text = report
response = extractor.extract_AJCCstage(report_text)
```

The output is a dictionary, like the following

```
{'stage': 'Stage III'}
```

### For Prognosis

```python
import sys
sys.path.append("loc/to/PathRep-Bench")

from src.inference.DataExtractor import DataExtractor

report = "Pathology Report Contents ......"

extractor = DataExtractor()
report_text = report
response = extractor.extract_Prognosis(report_text, cancer_type='Colon adenocarcinoma')
```

The output is a dictionary, but it also prints a statement like the following

```
The patient's survival after 2.2 years - 

{
 "survival": "False" 
}
```

⚠️ Please be careful while using the prognosis, as it is a model derived value. 


## Google Colab

Coming soon....

## Instruction Tuning

You will have to create an instruction tuning dataset first and then you can use the trainer. Examples of the instruction tuning dataset is in `src/instruction_tuning/`. 

`src/instruction_tuning/trainer.py`, will help you run supervised fine tuning. 


## Running Benchmarks

Setting up the API keys: To run the benchmarks, you need to create a `.env` file in the `PathRep-Bench` directory. Replace `{MISTRAL_API_KEY_VALUE}` and `{OPENAI_API_KEY_VALUE}` with your actual API keys.

```
echo -e "MISTRAL_API_KEY={MISTRAL_API_KEY_VALUE}\nOPENAI_API_KEY={OPENAI_API_KEY_VALUE}" > .env
```

#### Cancer Type Identification

To start benchmarking just run the following command:

```
python ./benchmarks/Task1_Disease/preds-gpt-4o.py
```

You can replace `preds-gpt-4o.py` with a different model script as needed. The script will automatically run the model five times and save the outputs in a new folder.

If you need to modify the configuration (e.g., number of runs, output directory), you can adjust the settings directly in the script.

#### AJCC Stage Identification

Similarly, you can run 

```
python ./benchmarks/Task2_Stage/preds-gpt-4o.py
```

#### Prognosis Assessment

Finally, you can run the following command to benchmark the prognosis task. 

```
python ./benchmarks/Task3_Prognosis/preds-gpt-4o.py
```

## Citation

If you find our work or any of our materials useful, please cite our paper:

```
@article{saluja2025cancer,
  title={Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs},
  author={Saluja, Rachit and Rosenthal, Jacob and Artzi, Yoav and Pisapia, David J and Liechty, Benjamin L and Sabuncu, Mert R},
  journal={arXiv preprint arXiv:2503.01194},
  year={2025}
}
```