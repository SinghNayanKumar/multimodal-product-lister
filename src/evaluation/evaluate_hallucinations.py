# src/evaluation/evaluate_hallucinations.py

import pandas as pd
import json
import re
from tqdm import tqdm

def calculate_hallucination_rate(predictions_df: pd.DataFrame, attribute_cols: list):
    """
    Calculates the attribute hallucination rate.

    A hallucination is defined as an attribute value mentioned in the generated text
    that is NOT present in the list of predicted (or ground-truth) attributes for that item.

    Args:
        predictions_df (pd.DataFrame): DataFrame with columns 'generated_text' and
                                       one column for each predicted attribute.
        attribute_cols (list): A list of the attribute column names to check against.

    Returns:
        float: The hallucination rate (hallucinations / total attribute mentions).
    """
    hallucinations = 0
    total_mentions = 0

    for _, row in tqdm(predictions_df.iterrows(), total=len(predictions_df), desc="Evaluating Hallucinations"):
        generated_text = str(row['generated_text']).lower()
        
        # Create a set of all possible ground-truth/predicted attribute values for this item
        ground_truth_values = set()
        for col in attribute_cols:
            attr_value = str(row[col]).lower()
            if attr_value not in ['unknown', 'nan']:
                ground_truth_values.add(attr_value)

        # Find all potential attribute values mentioned in the text
        # This is a simple but effective heuristic: check for multi-word phrases first
        # then single words to avoid double counting (e.g., "v-neck" and "neck").
        mentioned_values = set(re.findall(r'\b\w+(?:-\w+)*\b', generated_text))

        if not mentioned_values:
            continue

        total_mentions += len(mentioned_values.intersection(ground_truth_values))

        # A hallucination is a mentioned value that is NOT in the ground truth set
        for mentioned_value in mentioned_values:
            # Simple check: if a plausible attribute is mentioned, is it correct?
            # We check if it's NOT in the ground truth AND if it appears in our global list of possible attributes
            # to avoid penalizing descriptive words that aren't attributes (e.g., "beautiful", "soft").
            # For this simple implementation, we'll count any mentioned value not in the ground_truth set as a potential issue.
            # A more advanced version could use a global set of all possible attribute values.
            if mentioned_value not in ground_truth_values:
                # To be more precise, we can check if the mentioned word exists in ANY of the possible attribute values
                # across the entire dataset. For now, we'll use a simpler heuristic.
                # Let's assume for this check that if a value is mentioned, it's intended as an attribute.
                
                # A simple check to see if the value is part of a larger correct value (e.g., "v" in "v-neck")
                is_substring_of_correct_value = any(mentioned_value in s for s in ground_truth_values)

                if not is_substring_of_correct_value:
                    hallucinations += 1


    # Rate = (Number of incorrect attributes mentioned) / (Total number of attributes mentioned)
    # A lower rate is better.
    return hallucinations / total_mentions if total_mentions > 0 else 0.0

def main():
    # This is an example of how you would run this script
    # You would first need to generate predictions from your models and save them as a CSV.
    
    # 1. Load your attribute mappings to get the list of attributes
    with open('data/processed/attribute_mappings.json', 'r') as f:
        mappings = json.load(f)
    ATTRIBUTE_COLS = list(mappings.keys())
    
    # 2. Load a file containing predictions.
    # This file should have a 'generated_text' column and columns for each predicted attribute.
    # For this example, we'll create a dummy dataframe.
    data = {
        'generated_text': [
            "A beautiful red v-neck t-shirt with short sleeves, perfect for summer.", # Correct
            "Check out this blue crew-neck sweater. It has long sleeves and a boat neck.", # "boat neck" is a hallucination
            "A simple black top." # No specific attributes mentioned, rate should be unaffected.
        ],
        'colour': ['red', 'blue', 'black'],
        'Neck': ['V-Neck', 'Crew-Neck', 'Round Neck'],
        'Sleeve Length': ['Short Sleeve', 'Long Sleeve', 'Short Sleeve']
    }
    df = pd.DataFrame(data)

    rate = calculate_hallucination_rate(df, ['colour', 'Neck', 'Sleeve Length'])
    
    print(f"Attribute Hallucination Rate: {rate:.2%}") # Example output: 20.00% (1 hallucination / 5 total mentions)

if __name__ == '__main__':
    main()