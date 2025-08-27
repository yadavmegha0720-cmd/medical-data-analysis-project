# Project Title: Medical Data Cleanup and Analysis
#
# This Python script demonstrates a fundamental data analytics workflow:
# 1. Loading a dataset from a publicly available URL.
# 2. Cleaning the data by handling missing values and correcting data types.
# 3. Performing basic statistical analysis and visualization.
# 4. Saving cleaned dataset and summary report.
#
# The dataset used is the 'PIMA Indians Diabetes Database' from Kaggle.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

EXPECTED_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

def load_data(url):
    """Loads data from a specified URL into a pandas DataFrame."""
    try:
        data = pd.read_csv(url)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def verify_and_rename_columns(df):
    """Verifies column count and assigns meaningful names if needed."""
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return None
    if len(df.columns) == len(EXPECTED_COLS):
        df.columns = EXPECTED_COLS
    else:
        print("Error: Unexpected number of columns in the dataset.")
        print(f"Found columns: {df.columns.tolist()}")
        return None
    return df

def clean_data(df):
    """
    Cleans the DataFrame by handling missing values and ensuring correct data types.
    Treats biologically impossible zeroes as missing (NaN), then fills with median.
    """
    if df is None or df.empty:
        print("Error: Cannot clean empty or None DataFrame.")
        return None
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_clean:
        if col not in df.columns:
            print(f"Error: Expected column '{col}' not found in DataFrame.")
            return None
    df[columns_to_clean] = df[columns_to_clean].replace(0, np.nan)
    df[columns_to_clean] = df[columns_to_clean].apply(lambda x: x.fillna(x.median()))
    print("\nCleaned data head:")
    print(df.head())
    return df

def analyze_and_summarize(df, save_summary=False):
    """Performs data analysis, generates summary, and visualizes key findings."""
    if df is None or df.empty:
        print("Error: Cannot analyze empty or None DataFrame.")
        return
    output_lines = []
    output_lines.append("Statistical Summary:\n")
    output_lines.append(df.describe().T.to_string())
    diabetes_counts = df['Outcome'].value_counts()
    output_lines.append("\nDiabetes Outcome Distribution:")
    output_lines.append(diabetes_counts.to_string())
    mean_bmi_by_outcome = df.groupby('Outcome')['BMI'].mean()
    output_lines.append("\nMean BMI by Outcome (0=Non-diabetic, 1=Diabetic):")
    output_lines.append(mean_bmi_by_outcome.to_string())
    summary_report = "\n".join(output_lines)
    print("\n" + summary_report)
    if save_summary:
        with open("summary_report.txt", "w") as f:
            f.write(summary_report)
        print("Summary report saved to summary_report.txt")
    # Bar chart for diabetes outcome
    plt.figure(figsize=(8, 6))
    diabetes_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Diabetes Outcomes')
    plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Histogram for BMI
    plt.figure(figsize=(8, 6))
    for outcome, color in zip([0, 1], ['skyblue', 'salmon']):
        df[df["Outcome"] == outcome]["BMI"].plot(kind="hist", alpha=0.5, color=color, bins=20, label=f"Outcome {outcome}")
    plt.title("BMI Distribution by Diabetes Outcome")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Histogram for Glucose
    plt.figure(figsize=(8, 6))
    for outcome, color in zip([0, 1], ['skyblue', 'salmon']):
        df[df["Outcome"] == outcome]["Glucose"].plot(kind="hist", alpha=0.5, color=color, bins=20, label=f"Outcome {outcome}")
    plt.title("Glucose Distribution by Diabetes Outcome")
    plt.xlabel("Glucose")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_cleaned_data(df, filename="cleaned_data.csv"):
    """Save cleaned dataframe to a CSV file."""
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"Cleaned data saved to {filename}")

def main():
    """Main function to run the data analysis workflow."""
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    medical_data = load_data(data_url)
    medical_data = verify_and_rename_columns(medical_data)
    if medical_data is not None:
        cleaned_data = clean_data(medical_data)
        if cleaned_data is not None:
            save_cleaned_data(cleaned_data)
            analyze_and_summarize(cleaned_data, save_summary=True)
            print("\nProject completed successfully.")
        else:
            print("Data cleaning failed.")
    else:
        print("Data loading or column verification failed.")

if __name__ == "__main__":
    main()
