#!/bin/bash

# For AIC 
module purge
module load anaconda3

source activate HF-runner

cd ./benchmarks/Task1_Disease/
#cd ./benchmarks/Task2_Stage/
#cd ./extras/
#cd ./benchmarks/Task3_Prognosis/

#python preds-llama3-8b.py
#python preds-llama3-70b.py
#python preds-mistral-large.py
#python preds-mistral-medium.py

#python preds-gpt-4o-mini-FT.py
#python preds-gpt-4o-mini.py
#python preds-gpt-4o.py
#python preds-llama3-8b.py
#python preds-llama3-70b.py
#python preds-mistral-large.py
#python preds-mistral-medium.py

#python summarization.py
#python preds-gpt-4o-mini.py
#python preds-gpt-4o.py
#python preds-llama3-8b.py
#python preds-llama3-70b.py

#python preds-mistral-medium.py
#python preds-mistral-large.py

#python preds-pathllama3.1-8b.py
python preds-llama3.1-8b.py
