import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import pandas as pd
import os

# Configuration
os.environ["WANDB_PROJECT"]="Pathology-LLM"
max_seq_length = 4096
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
model_save_path = "/midtier/sablab/scratch/ras4037/workspace/LLM_path/fine-tuned-models/llama3.1_pathrep_v1.0"
model_save_path_merged = "/midtier/sablab/scratch/ras4037/workspace/LLM_path/fine-tuned-models/llama3.1_pathrep_v1.0-merged"
training_data_path = "/midtier/sablab/scratch/ras4037/workspace/LLM_path/SFT-training_data/"

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

# Apply LoRA settings
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

# Load training and validation data
train_disease_df = pd.read_csv(f'{training_data_path}train_disease.csv')
train_stage_df = pd.read_csv(f'{training_data_path}train_stage.csv')
train_prognosis_df = pd.read_csv(f'{training_data_path}train_prognosis.csv')

val_disease_df = pd.read_csv(f'{training_data_path}val_disease.csv')
val_stage_df = pd.read_csv(f'{training_data_path}val_stage.csv')
val_prognosis_df = pd.read_csv(f'{training_data_path}val_prognosis.csv')

# Concatenate dataframes
final_train_df = pd.concat(
    [train_disease_df, train_stage_df, train_prognosis_df])
final_val_df = pd.concat([val_disease_df, val_stage_df, val_prognosis_df])

# Shuffle the data
final_train_df_shuffled = final_train_df.sample(frac=1).reset_index(drop=True)
final_val_df_shuffled = final_val_df.sample(frac=1).reset_index(drop=True)

# Create conversation lists from dataframe rows
def create_conversations_list(row):
    return [
        {"from": "system", "value": row['system_prompt']},
        {"from": "human", "value": row['user_prompt']},
        {"from": "gpt", "value": row['assistant_output']}
    ]


final_train_df_shuffled['conversations'] = final_train_df_shuffled.apply(
    create_conversations_list, axis=1)
final_val_df_shuffled['conversations'] = final_val_df_shuffled.apply(
    create_conversations_list, axis=1)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(final_train_df_shuffled)
val_dataset = Dataset.from_pandas(final_val_df_shuffled)

# Set up the tokenizer with chat template
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value",
             "system": "system", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

# Function to apply chat template


def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}


train_dataset_chat = train_dataset.map(apply_template, batched=True)
val_dataset_chat = train_dataset.map(apply_template, batched=True)

# Prepare trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset_chat,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=6000,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=25,
        optim="adamw_8bit",
        save_total_limit=5,
        weight_decay=0.01,
        warmup_steps=10,
        output_dir=model_save_path,
        seed=0,
        group_by_length=True,
        report_to="wandb",
        run_name="llama3.1_pathrep_v1.0"
    ),
)

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
trainer.train()

# Save the trained model
model.save_pretrained_merged(
    model_save_path_merged, tokenizer, save_method="merged_16bit")