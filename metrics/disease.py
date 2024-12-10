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


def parse_prediction(prediction):
    """
    Extracts the value from the prediction string. If parsing fails, returns 'no answer'.
    """
    try:
        match = re.search(r'\{(.+?)\}', prediction.replace('\n',
                          '').replace('\\n', '').strip(), re.DOTALL)
        if match:
            dictionary_string = match.group(0)
            extracted_dict = ast.literal_eval(dictionary_string)
            return list(extracted_dict.values())[0].strip().lower()
    except Exception:
        pass
    return "no answer"


def evaluate_predictions(pred_df, test_set_loc, disease_list):
    """
    Compares predictions against ground truth and calculates accuracy, precision, recall, and F1-score for each disease type.
    """
    # Parse predictions
    pred_df['Parsed_preds'] = pred_df['preds'].apply(parse_prediction)

    # Load ground truth
    ground_truth = pd.read_csv(test_set_loc)
    ground_truth['type_name'] = ground_truth['type_name'].str.lower()

    # Create comparison DataFrame
    comp_df = pd.DataFrame({
        "True_answers": ground_truth['type_name'],
        "Pred_answers": pred_df['Parsed_preds']
    })
    comp_df['ComparisonResult'] = comp_df['True_answers'] == comp_df['Pred_answers']

    metrics = {}
    for disease in disease_list:
        disease_comp_df = comp_df[comp_df['True_answers'] == disease]
        y_true = (disease_comp_df['True_answers'] == disease).astype(int)
        y_pred = (disease_comp_df['Pred_answers'] == disease).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        accuracy = disease_comp_df['ComparisonResult'].mean()

        metrics[disease] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


def main(prediction_dir, test_set_loc, output_file):
    """
    Main function to process prediction files, evaluate them, and save results.
    """
    # Load disease list
    ground_truth_df = pd.read_csv(test_set_loc)
    disease_list = sorted(
        ground_truth_df['type_name'].str.lower().unique().tolist())

    # Locate all prediction CSVs in the provided directory
    prediction_files = natsorted(glob(os.path.join(prediction_dir, "*.csv")))
    if not prediction_files:
        print(f"No CSV files found in the directory: {prediction_dir}")
        return

    # Evaluate predictions for each CSV
    all_metrics = []
    for file in tqdm(prediction_files, desc="Evaluating Predictions"):
        metrics = evaluate_predictions(
            pred_df=pd.read_csv(file),
            test_set_loc=test_set_loc,
            disease_list=disease_list
        )
        all_metrics.append(metrics)

    # Aggregate metrics
    aggregated_metrics = {}
    for disease in disease_list:
        accuracy = [metrics[disease]['accuracy'] for metrics in all_metrics]
        precision = [metrics[disease]['precision'] for metrics in all_metrics]
        recall = [metrics[disease]['recall'] for metrics in all_metrics]
        f1 = [metrics[disease]['f1'] for metrics in all_metrics]

        aggregated_metrics[disease] = {
            "accuracy": {"mean": np.mean(accuracy), "std": np.std(accuracy)},
            "precision": {"mean": np.mean(precision), "std": np.std(precision)},
            "recall": {"mean": np.mean(recall), "std": np.std(recall)},
            "f1": {"mean": np.mean(f1), "std": np.std(f1)},
        }

    # Calculate overall metrics
    overall_metrics = {
        metric: {
            "mean": np.mean([aggregated_metrics[disease][metric]["mean"] for disease in disease_list]),
            "std": np.mean([aggregated_metrics[disease][metric]["std"] for disease in disease_list]),
        }
        for metric in ["accuracy", "precision", "recall", "f1"]
    }

    aggregated_metrics["Overall"] = overall_metrics

    # Print results dictionary with proper indentation
    print(json.dumps(aggregated_metrics, indent=4))

    # Save results to a JSON file if output_file is specified
    if output_file:
        with open(output_file, "w") as json_file:
            json.dump(aggregated_metrics, json_file, indent=4)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground truth.")
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
