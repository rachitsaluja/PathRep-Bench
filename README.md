# PathRep-Bench

Official implementation of [Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs](https://arxiv.org/abs/2503.01194)

This repository provides a benchmarking framework for evaluating large language models (LLMs) on pathology report analysis, specifically for:
- Cancer Type Identification: Determining the type of cancer from pathology reports.
- AJCC Cancer Staging: Identifying cancer stage (Stage I, Stage II, Stage III, and Stage IV) based on pathology reports.
- Prognosis Prediction: Assessing whether a patient has a "good" or "bad" prognosis from pathology reports.

We also provide -

- Full Dataset: The dataset used in our study is publicly available [here](https://huggingface.co/datasets/rosenthal/tcga-path-notes) 🤗.
- Pretrained Model for Inference: We provide an instruction-tuned model that can be used for inference on custom pathology reports via a simple Python API.
- Fine-Tuning Code: Scripts for generating instruction-tuning data and training your own fine-tuned model.
- Google Colab Implementation: A ready-to-use Google Colab notebook for easy experimentation | [![Explore UniverSeg in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/NEEDTODO)<br>. 

> ⚠️ Note: Our model and evaluations are based on a specific set of cancer types. Please keep this in mind when using the model, as your pathology reports may contain cancer types that were not included in our analysis.

## Running Benchmarks

To get started, first, clone the repository:

```
git clone https://github.com/rachitsaluja/PathRep-Bench.git
```

Next, navigate into the repository directory:

```
cd PathRep-Bench
```

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