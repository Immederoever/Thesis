# COMPARE_HUMAN.py
import pandas as pd
from scipy.stats import norm
import numpy as np

# Calculates d-prime for each target word and then averages them
def calculate_dprime_mean(df_group, contraint_file):
    dprimes = []
    
    for target_word in df_group['target'].unique():
        # Count the hits
        target_trials = df_group[df_group['target'] == target_word]
        hits = sum(target_trials['pred'] == target_word)
        
        # Count the false alarms
        other_trials = contraint_file[contraint_file['target'] != target_word]
        false_alarms = sum(other_trials['pred'] == target_word)
        
        # Log-linear correction
        hr = (hits + 0.5) / (len(target_trials) + 1)
        far = (false_alarms + 0.5) / (len(other_trials) + 1)
        
        # Calculate d-prime for this specific word after normalizing to z-score
        d = norm.ppf(hr) - norm.ppf(far)
        dprimes.append(d)
        
    return np.mean(dprimes)

def analyze(model_csv, constraint_csv):
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
        
        dprime_mean = calculate_dprime_mean(df_group, df)
        
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
        print(f"  Mean d-prime score: {dprime_mean:.3f}")
        print(f"  Overall match with human responses: {matched_human_modal}/{total_guesses} ({(matched_human_modal/total_guesses)*100:.1f}%)")
        print(f"  When model was WRONG, it guessed the same word as humans: {alignment_percentage:.1f}% of the time")

def main():
    constraint_file = "data/CaseClozed_ConstraintInfo.csv"
    analyze("FINAL_OPTIMIZED0_checkpoints/results_stage_child_best.csv", constraint_file)
    analyze("FINAL_OPTIMIZED0_checkpoints/results_stage_adult_best.csv", constraint_file)

if __name__ == "__main__":
    main()
