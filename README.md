# PathRep-Bench

Official implementation of [Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs](https://arxiv.org/abs/2503.01194)

This is a repository for benchmarking LLMs for -
- Cancer type Identification from Pathology Reports.
- AJCC Cancer Stage Identification from Pathology Reports (i.e Stage I, Stage II, Stage III and Stage IV). 
- Predicting from the Pathology Reports, if the patient has "good" or "bad" prognosis. 


Apart from the above benchmarks, we also provide -
- The full dataset used for the analysis, which can be found [here](https://huggingface.co/datasets/rosenthal/tcga-path-notes) 🤗.
- We provide an instruction tuned model that you can use to run inference on your own pathology reports via simple python API.
- Code for creating the instruction tuning data and training your own fine-tuned model. 
- Also provide a Google Colab implementation for users. 





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