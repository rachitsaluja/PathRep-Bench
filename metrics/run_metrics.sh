#!/bin/bash

# Define model names
models=("gpt-4o" "gpt-4o-mini-ft" "gpt-4o-mini" "llama3-8b" "llama3-70b" "mistral-large" "mistral-medium" "pathllama3.1-8b")

# Define file paths
test_set_loc="../data/test.csv"
test_file="../data/test.csv"
test_summarization_file="../extras/test-summarization/test-summarization-2024-09-26_20-41-24.csv"
train_file="../data/train.csv"
val_file="../data/val.csv"

# Task 1: Disease
echo "Running Task 1: Disease"
for model in "${models[@]}"; do
  prediction_dir="../benchmarks/Task1_Disease/${model}/"
  output_file="./Task1_Disease/${model}.json"
  python disease.py --prediction_dir "$prediction_dir" --test_set_loc "$test_set_loc" --output_file "$output_file"
done

# Task 2: Stage
echo "Running Task 2: Stage"
for model in "${models[@]}"; do
  prediction_dir="../benchmarks/Task2_Stage/${model}/"
  output_file="./Task2_Stage/${model}.json"
  python stage.py --prediction_dir "$prediction_dir" --test_set_loc "$test_set_loc" --output_file "$output_file"
done

# Task 3: Prognosis
echo "Running Task 3: Prognosis"
for model in "${models[@]}"; do
  prediction_dir="../benchmarks/Task3_Prognosis/${model}/"
  output_file="Task3_Prognosis/${model}.json"
  python prognosis.py --prediction_dir "$prediction_dir" --test_file "$test_file" --test_summarization_file "$test_summarization_file" --train_file "$train_file" --val_file "$val_file" --output_file "$output_file"
done

echo "All tasks completed!"