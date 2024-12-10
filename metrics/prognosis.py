import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
from glob2 import glob
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import json


def load_and_prepare_data(test_file, test_summarization_file, train_file, val_file, disease_list):
    """
    Loads and prepares data for analysis by combining test, train, and validation datasets.
    """
    test_df = pd.read_csv(test_file)
    summarization_test_df = pd.read_csv(test_summarization_file)
    test_df['text_summarized'] = summarization_test_df['preds']
    test_df = test_df[['text', 'text_summarized', 'stage_overall', 'type_name',
                       'age_at_initial_pathologic_diagnosis', 'gender', 'DSS.time']].dropna().reset_index(drop=True)
    test_df['DSS.time'] = test_df['DSS.time'].astype(float)

    train_df = pd.read_csv(train_file)
    train_df['DSS.time'] = train_df['DSS.time'].astype(float)
    val_df = pd.read_csv(val_file)
    val_df['DSS.time'] = val_df['DSS.time'].astype(float)

    full_df = pd.concat([train_df, val_df])

    # Calculate average survival times for each disease
    disease_times = [
        np.round(
            (np.mean(full_df[full_df['type_name'] == disease]['DSS.time']) / 365), 2)
        for disease in disease_list
    ]
    test_df['DSS.time'] = np.round(test_df['DSS.time'] / 365, 3)

    # Assign survival times and calculate survival_over_mean
    test_df['Survival_times'] = [
        disease_times[disease_list.index(disease)]
        for disease in test_df['type_name']
    ]
    test_df['survival_over_mean'] = (
        test_df['DSS.time'] > test_df['Survival_times']).astype(str)

    return test_df


def evaluate_predictions(pred_df, disease_name, test_df):
    """
    Evaluates predictions for a specific disease and calculates metrics.
    """
    test_subset = test_df[test_df['type_name']
                          == disease_name].reset_index(drop=True)
    test_subset['year_pred'] = pred_df['preds']
    test_subset['year_pred'] = test_subset['year_pred'].str.replace(
        '[ABCD]', '', regex=True)

    op = []
    for pred in test_subset['year_pred']:
        try:
            match = re.search(
                r'\{(.+?)\}', pred.replace('\n', '').replace('\\n', '').strip(), re.DOTALL)
            dictionary_string = match.group(0)
            extracted_dict = ast.literal_eval(dictionary_string)
            op.append(list(extracted_dict.values())[0].strip().lower())
        except Exception:
            op.append("no answer")

    true_answers = [answer.lower()
                    for answer in test_subset['survival_over_mean']]
    return true_answers, op


def calculate_metrics(gt, pred):
    """
    Calculates accuracy, precision, recall, F1-score, and ROC AUC.
    """
    gt_binary = [1 if x == "true" else 0 for x in gt]
    pred_binary = [1 if x == "true" else 0 for x in pred]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_binary, pred_binary, average='binary', zero_division=0)
    accuracy = accuracy_score(gt_binary, pred_binary)
    roc_auc = roc_auc_score(gt_binary, pred_binary)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def main(prediction_dir, test_file, test_summarization_file, train_file, val_file, output_file):
    """
    Main function to evaluate predictions and calculate metrics.
    """
    disease_list = [
        'Adrenocortical carcinoma', 'Bladder Urothelial Carcinoma', 'Breast invasive carcinoma',
        'Cholangiocarcinoma', 'Colon adenocarcinoma', 'Esophageal carcinoma',
        'Head and Neck squamous cell carcinoma', 'Kidney Chromophobe',
        'Kidney renal clear cell carcinoma', 'Kidney renal papillary cell carcinoma',
        'Liver hepatocellular carcinoma', 'Lung adenocarcinoma', 'Lung squamous cell carcinoma',
        'Mesothelioma', 'Pancreatic adenocarcinoma', 'Rectum adenocarcinoma',
        'Skin Cutaneous Melanoma', 'Stomach adenocarcinoma', 'Testicular Germ Cell Tumors',
        'Thyroid carcinoma', 'Uveal Melanoma'
    ]

    # Load and prepare data
    test_df = load_and_prepare_data(
        test_file, test_summarization_file, train_file, val_file, disease_list)

    # Evaluate predictions
    results_dict = {}
    full_metrics = []
    disease_means = {}
    disease_stds = {}

    for disease in tqdm(disease_list, desc="Evaluating Diseases"):
        disease_csvs = natsorted(
            glob(os.path.join(prediction_dir, f"{disease}*.csv")))
        disease_metrics = []

        for csv in disease_csvs:
            pred_df = pd.read_csv(csv)
            gt, pred = evaluate_predictions(pred_df, disease, test_df)
            metrics = calculate_metrics(gt, pred)
            disease_metrics.append(metrics)

        if disease_metrics:
            # Calculate mean and std for the disease
            metrics_df = pd.DataFrame(disease_metrics)
            results_dict[disease] = {
                metric: {
                    "mean": metrics_df[metric].mean(),
                    "std": metrics_df[metric].std()
                }
                for metric in metrics_df.columns
            }

            # Collect disease-level means and stds
            for metric in metrics_df.columns:
                if metric not in disease_means:
                    disease_means[metric] = []
                    disease_stds[metric] = []
                disease_means[metric].append(metrics_df[metric].mean())
                disease_stds[metric].append(metrics_df[metric].std())

            full_metrics.extend(disease_metrics)

    # Calculate overall statistics
    if disease_means:
        results_dict["Overall"] = {
            metric: {
                "mean": np.mean(disease_means[metric]),
                "mean_std": np.mean(disease_stds[metric])
            }
            for metric in disease_means
        }

    # Print results
    print(json.dumps(results_dict, indent=4))

    # Save results to a JSON file if output_file is specified
    if output_file:
        with open(output_file, "w") as json_file:
            json.dump(results_dict, json_file, indent=4)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate prognosis predictions for all diseases.")
    parser.add_argument("--prediction_dir", type=str, required=True,
                        help="Path to the directory containing prediction CSV files.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test dataset CSV file.")
    parser.add_argument("--test_summarization_file", type=str, required=True,
                        help="Path to the summarization predictions CSV file.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the train dataset CSV file.")
    parser.add_argument("--val_file", type=str, required=True,
                        help="Path to the validation dataset CSV file.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the results as a JSON file.")
    args = parser.parse_args()

    main(
        prediction_dir=args.prediction_dir,
        test_file=args.test_file,
        test_summarization_file=args.test_summarization_file,
        train_file=args.train_file,
        val_file=args.val_file,
        output_file=args.output_file
    )
