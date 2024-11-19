import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
from glob2 import glob
from tqdm import tqdm
from natsort import natsorted
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
    Compares predictions against ground truth and calculates accuracy for each disease type.
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

    # Calculate per-disease accuracy
    return [
        comp_df[comp_df['True_answers'] == disease]['ComparisonResult'].mean()
        for disease in disease_list
    ]


def main(prediction_dir, test_set_loc):
    """
    Main function to process prediction files and evaluate them.
    """
    # Load disease list
    ground_truth_df = pd.read_csv(test_set_loc)
    disease_list = sorted(ground_truth_df['type_name'].str.lower().unique().tolist())

    # Locate all prediction CSVs in the provided directory
    prediction_files = natsorted(glob(os.path.join(prediction_dir, "*.csv")))
    if not prediction_files:
        print(f"No CSV files found in the directory: {prediction_dir}")
        return

    # Evaluate predictions for each CSV
    results_list = [
        evaluate_predictions(
            pred_df=pd.read_csv(file),
            test_set_loc=test_set_loc,
            disease_list=disease_list
        )
        for file in tqdm(prediction_files, desc="Evaluating Predictions")
    ]

    # Calculate mean and standard deviation for each disease type
    full_scores = np.array(results_list).T.tolist()
    means = [np.mean(scores) for scores in full_scores]
    std_devs = [np.std(scores) for scores in full_scores]

    # Build results dictionary
    results_dict = {
        disease: {"mean": round(mean, 3), "std": round(std_dev, 3)}
        for disease, mean, std_dev in zip(disease_list, means, std_devs)
    }

    # Calculate overall mean and standard deviation across all diseases
    overall_mean_score = np.mean(means)
    overall_mean_std = np.mean(std_devs)

    results_dict["Overall"] = {
        "mean": round(overall_mean_score, 3),
        "std": round(overall_mean_std, 3)
    }

    # Print results dictionary with proper indentation
    print(json.dumps(results_dict, indent=4))


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
    args = parser.parse_args()

    main(prediction_dir=args.prediction_dir, test_set_loc=args.test_set_loc)
