# REGRESSION.py
import pandas as pd
import statsmodels.formula.api as smf
import sys

def analyze_stage(model_csv, stim_csv):
    print(f"Analyzing: {model_csv}")
    
    df_model = pd.read_csv(model_csv)
    df_stim = pd.read_csv(stim_csv)
    df_stim = df_stim.drop_duplicates(subset=['target', 'cloze_group'])
    df = pd.merge(df_model,
            df_stim,
            left_on=['target', 'group'],
            right_on=['target', 'cloze_group'],
            how='inner')
    
    print(f"Merged {len(df)} lines")

    # Using OLS regression to test whether frequency and concreteness have an effect on surprisal
    model = smf.ols("surprisal ~ freq_value + concrete_value", data=df).fit()
    print("\n--- Regression Results ---")
    print(model.summary().tables[1])
    print(f"R-squared: {model.rsquared:.3f}")

def main():
    stimulus_file = "data/CaseClozed_StimKey.csv" 
    
    analyze_stage("FINAL_OPTIMIZED0_checkpoints/results_stage_child_best.csv", stimulus_file)
    analyze_stage("FINAL_OPTIMIZED0_checkpoints/results_stage_adult_best.csv", stimulus_file)

if __name__ == "__main__":
    main()
