#!/bin/bash

source activate HF-runner

#cd ./benchmarks/Task1_Disease/
cd ./benchmarks/Task2_Stage/

#python preds-llama3-8b.py
#python preds-llama3-70b.py
#python preds-mistral-large.py
#python preds-mistral-medium.py

#python preds-gpt-4o-mini.py
python preds-gpt-4o.py