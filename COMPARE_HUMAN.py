# COMPARE_HUMAN.py
import pandas as pd
from scipy.stats import norm
import numpy as np

#Calculates d-prime for each target word and then averages them
def calculate_item_dprime_mean(df_group, df_total):
    d_primes = []
    
    for target_word in df_group['target'].unique():
        # Count the hits
        target_trials = df_group[df_group['target'] == target_word]
        actual_trials = len(target_trials)
        hits = sum(target_trials['pred'] == target_word)
        
        # Count the false alarms
        other_trials = df_total[df_total['target'] != target_word]
        total_other = len(other_trials)
        false_alarms = sum(other_trials['pred'] == target_word)
        
        # Macmillan & Creelman Smoothing 
        hr = (hits + 0.5) / (actual_trials + 1)
        far = (false_alarms + 0.5) / (total_other + 1)
        
        # Calculate d-prime for this specific word
        d = norm.ppf(hr) - norm.ppf(far)
        d_primes.append(d)
        
    return np.mean(d_primes)

def analyze_alignment(model_csv, constraint_csv):
    print(f"Analyzing: {model_csv}")
    
    df_model = pd.read_csv(model_csv)
    df_const = pd.read_csv(constraint_csv)
    df_const = df_const.rename(columns={'target_word': 'target'})

    df_model['pred'] = df_model['pred'].astype(str).str.lower()
    df_model['target'] = df_model['target'].astype(str).str.lower()
    df_const['target'] = df_const['target'].astype(str).str.lower()
    df_const['modal_response'] = df_const['modal_response'].astype(str).str.lower()
    df_const = df_const.drop_duplicates(subset=['target', 'cloze_group'])
    
    df = pd.merge(df_model, df_const, 
                  left_on=['target', 'group'], 
                  right_on=['target', 'cloze_group'], 
                  how='inner')

    for group in ["HC", "LC"]:
        df_group = df[df['group'] == group]
        if len(df_group) == 0:
            continue
        
        d_prime_mean = calculate_item_dprime_mean(df_group, df)
        
        # Counting similarity in incorrect guesses of humans and models
        total_guesses = len(df_group)
        matched_human_modal = sum(df_group['pred'] == df_group['modal_response'])
        
        df_wrong = df_group[df_group['pred'] != df_group['target']]
        if len(df_wrong) > 0:
            wrong_aligned = sum(df_wrong['pred'] == df_wrong['modal_response'])
            alignment_percentage = (wrong_aligned / len(df_wrong)) * 100
        else:
            alignment_percentage = 0.0

        print(f"\n[{group} Cloze]")
        print(f"  Mean d-prime score: {d_prime_mean:.3f}")
        print(f"  Overall match with human responses: {matched_human_modal}/{total_guesses} ({(matched_human_modal/total_guesses)*100:.1f}%)")
        print(f"  When model was WRONG, it guessed the same word as humans: {alignment_percentage:.1f}% of the time")

def main():
    constraint_file = "data/CaseClozed_ConstraintInfo.csv"
    analyze_alignment("FINAL_OPTIMIZED0_checkpoints/results_stage_child_best.csv", constraint_file)
    analyze_alignment("FINAL_OPTIMIZED0_checkpoints/results_stage_adult_best.csv", constraint_file)

if __name__ == "__main__":
    main()
