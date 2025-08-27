# Project Title: Medical Data Cleanup and Analysis
#
# This Python script demonstrates a fundamental data analytics workflow:
# 1. Loading a dataset from a publicly available URL.
# 2. Cleaning the data by handling missing values and correcting data types.
# 3. Performing basic statistical analysis.
# 4. Generating a summary report of key findings.
#
# This project is designed to showcase skills highly relevant to a data analyst role,
# including Python, pandas for data manipulation, and matplotlib for visualization.
#
# The dataset used is the 'PIMA Indians Diabetes Database' from Kaggle. It's a classic
# and widely-used dataset for beginners in data science. The data is about female
# patients and whether they have diabetes, based on various health measurements.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(url):
    """
    Loads data from a specified URL into a pandas DataFrame.
    """
    try:
        data = pd.read_csv(url)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Cleans the DataFrame by handling missing values and ensuring correct data types.
    
    The dataset contains '0' values in columns like BloodPressure, BMI, etc.,
    which are biologically impossible. We'll treat these as missing values (NaN)
    and replace them with the median of their respective columns.
    """
    print("\nStarting data cleaning process...")
    # List of columns where a '0' is a missing value
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace zeros with NaN to be treated as missing values
    df[columns_to_clean] = df[columns_to_clean].replace(0, np.nan)
    
    # Fill missing values with the median of each column
    for col in columns_to_clean:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in '{col}' with median: {median_value:.2f}")

    # Display the first few rows of the cleaned data
    print("\nCleaned data head:")
    print(df.head())
    return df

def analyze_and_summarize(df):
    """
    Performs basic data analysis and prints a summary.
    
    This function calculates key statistics and generates a simple
    bar chart to visualize the distribution of diabetes outcomes.
    """
    print("\nStarting data analysis and summarization...")
    
    # Basic statistical summary of the dataset
    print("\nStatistical Summary:")
    print(df.describe().T)
    
    # Count the number of patients with and without diabetes
    diabetes_counts = df['Outcome'].value_counts()
    print("\nDiabetes Outcome Distribution:")
    print(diabetes_counts)
    
    # Calculate the mean BMI for diabetic and non-diabetic patients
    mean_bmi_by_outcome = df.groupby('Outcome')['BMI'].mean()
    print("\nMean BMI by Outcome (0=Non-diabetic, 1=Diabetic):")
    print(mean_bmi_by_outcome)
    
    # Create a simple visualization (bar chart)
    plt.figure(figsize=(8, 6))
    diabetes_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Diabetes Outcomes')
    plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the data analysis workflow.
    """
    # URL for the dataset. This dataset is public and can be accessed directly.
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    
    # Load the data
    medical_data = load_data(data_url)
    
    if medical_data is not None:
        # Assign meaningful column names for clarity
        medical_data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        # Clean the data
        cleaned_data = clean_data(medical_data)
        
        # Analyze the cleaned data and visualize key findings
        analyze_and_summarize(cleaned_data)
        
        print("\nProject completed successfully.")

if __name__ == "__main__":
    main()
