import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
from glob2 import glob
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import precision_recall_fscore_support
import json


def parse_prediction(stage_pred):
    """
    Parses the stage prediction string and extracts the stage value.
    """
    try:
        match = re.search(r'\{(.+?)\}', stage_pred.replace('\n',
                          '').replace('\\n', '').strip(), re.DOTALL)
        if match:
            dictionary_string = match.group(0)
            extracted_dict = ast.literal_eval(dictionary_string)
            stage = list(extracted_dict.values())[0].strip().lower()
            if stage.startswith('i'):
                stage = 'stage ' + stage
            return stage
    except Exception:
        pass
    return "no answer"


def evaluate_stage_predictions(pred_df, disease_name, test_set_loc):
    """
    Evaluates predictions for a specific disease type and returns metrics and comparison DataFrame.
    """
    test_df = pd.read_csv(test_set_loc)
    test_df = test_df[test_df['type_name'] ==
                      disease_name].reset_index(drop=True)
    test_df = test_df[['text', 'stage_overall']
                      ].dropna().reset_index(drop=True)

    pred_df['stage_pred'] = pred_df['preds'].str.replace(
        '[ABCD]', '', regex=True)
    parsed_preds = pred_df['stage_pred'].apply(parse_prediction)

    true_answers = test_df['stage_overall'].str.lower().tolist()

    comp_df = pd.DataFrame({
        "True_answers": true_answers,
        "Pred_answers": parsed_preds
    })
    comp_df['ComparisonResult'] = comp_df['True_answers'] == comp_df['Pred_answers']

    # Calculate metrics
    y_true = pd.Series(comp_df['True_answers']).fillna("no answer")
    y_pred = pd.Series(comp_df['Pred_answers']).fillna("no answer")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    accuracy = comp_df['ComparisonResult'].mean()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}, comp_df


def main(prediction_dir, test_set_loc, output_file):
    """
    Main function to evaluate predictions for all diseases and save results.
    """
    # Define disease list
    disease_list = [
        'Adrenocortical carcinoma', 'Bladder Urothelial Carcinoma', 'Breast invasive carcinoma',
        'Cholangiocarcinoma', 'Colon adenocarcinoma', 'Esophageal carcinoma',
        'Head and Neck squamous cell carcinoma', 'Kidney Chromophobe',
        'Kidney renal clear cell carcinoma', 'Kidney renal papillary cell carcinoma',
        'Liver hepatocellular carcinoma', 'Lung adenocarcinoma',
        'Lung squamous cell carcinoma', 'Mesothelioma', 'Pancreatic adenocarcinoma',
        'Rectum adenocarcinoma', 'Skin Cutaneous Melanoma', 'Stomach adenocarcinoma',
        'Testicular Germ Cell Tumors', 'Thyroid carcinoma', 'Uveal Melanoma'
    ]

    # Evaluate predictions for each disease
    results_dict = {}
    disease_means = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    disease_stds = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for disease in tqdm(disease_list, desc="Evaluating Diseases"):
        disease_csvs = natsorted(
            glob(os.path.join(prediction_dir, f"{disease}*.csv")))
        if not disease_csvs:
            print(f"No prediction files found for disease: {disease}")
            continue

        metrics_list = []
        for csv in disease_csvs:
            metrics, _ = evaluate_stage_predictions(
                pd.read_csv(csv), disease, test_set_loc)
            metrics_list.append(metrics)

        # Calculate mean and std for the disease
        metrics_array = pd.DataFrame(metrics_list).to_numpy()
        disease_metrics = {
            "accuracy": {"mean": np.mean(metrics_array[:, 0]), "std": np.std(metrics_array[:, 0])},
            "precision": {"mean": np.mean(metrics_array[:, 1]), "std": np.std(metrics_array[:, 1])},
            "recall": {"mean": np.mean(metrics_array[:, 2]), "std": np.std(metrics_array[:, 2])},
            "f1": {"mean": np.mean(metrics_array[:, 3]), "std": np.std(metrics_array[:, 3])},
        }

        results_dict[disease] = disease_metrics

        # Collect mean and std for each metric
        disease_means["accuracy"].append(disease_metrics["accuracy"]["mean"])
        disease_means["precision"].append(disease_metrics["precision"]["mean"])
        disease_means["recall"].append(disease_metrics["recall"]["mean"])
        disease_means["f1"].append(disease_metrics["f1"]["mean"])

        disease_stds["accuracy"].append(disease_metrics["accuracy"]["std"])
        disease_stds["precision"].append(disease_metrics["precision"]["std"])
        disease_stds["recall"].append(disease_metrics["recall"]["std"])
        disease_stds["f1"].append(disease_metrics["f1"]["std"])

    # Calculate overall metrics
    overall_metrics = {
        "accuracy": {"mean": np.mean(disease_means["accuracy"]), "mean_std": np.mean(disease_stds["accuracy"])},
        "precision": {"mean": np.mean(disease_means["precision"]), "mean_std": np.mean(disease_stds["precision"])},
        "recall": {"mean": np.mean(disease_means["recall"]), "mean_std": np.mean(disease_stds["recall"])},
        "f1": {"mean": np.mean(disease_means["f1"]), "mean_std": np.mean(disease_stds["f1"])},
    }
    results_dict["Overall"] = overall_metrics

    # Print results
    print(json.dumps(results_dict, indent=4))

    # Save results to a JSON file if output_file is specified
    if output_file:
        with open(output_file, "w") as json_file:
            json.dump(results_dict, json_file, indent=4)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stage predictions for all diseases.")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Path to the directory containing prediction CSV files."
    )
    parser.add_argument(
        "--test_set_loc",
        type=str,
        default="../test.csv",
        help="Path to the ground truth test set CSV file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the results as a JSON file. If not provided, results are not saved."
    )
    args = parser.parse_args()

    main(prediction_dir=args.prediction_dir,
         test_set_loc=args.test_set_loc, output_file=args.output_file)
