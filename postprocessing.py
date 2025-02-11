import pandas as pd
import numpy as np
import os

def process_similarity_matrix():
    # Read the similarity matrix CSV file
    df = pd.read_csv("similarity_matrix (1).csv")
    
    # Store the project names (first column)
    project_names = df.iloc[:, 0]
    
    # Get the similarity scores (all columns except the first one) and convert to numeric
    similarity_scores = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    # Find rows that have at least one value >= 0.15
    max_scores = similarity_scores.max(axis=1)
    valid_rows = max_scores >= 0.15
    
    # Filter the dataframe to keep only rows with at least one score >= 0.15
    filtered_project_names = project_names[valid_rows]
    filtered_scores = similarity_scores[valid_rows]
    
    print(f"Original number of projects: {len(df)}")
    print(f"Number of projects after filtering (at least one score >= 0.15): {len(filtered_scores)}")
    print(f"Removed {len(df) - len(filtered_scores)} projects with all scores < 0.15")
    
    # Get the program names (column headers) and remove .pdf extension
    program_names = [col.replace('.pdf', '') for col in df.columns[1:]]
    
    # For each row, find the two columns with highest values
    result_data = []
    for idx, (project, scores) in enumerate(zip(filtered_project_names, filtered_scores.iterrows())):
        # Get top 2 values and their corresponding column names
        top_2 = scores[1].nlargest(2)
        result_data.append({
            'Project': project,
            'Most Similar Program': top_2.index[0].replace('.pdf', ''),
            'Second Most Similar Program': top_2.index[1].replace('.pdf', '')
        })
    
    # Create the result dataframe
    result_df = pd.DataFrame(result_data)
    
    # If the file exists, try to remove it first
    if os.path.exists('similarity_analysis_results.xlsx'):
        os.remove('similarity_analysis_results.xlsx')
    
    # Save to Excel file
    result_df.to_excel('similarity_analysis_results.xlsx', index=False)
    print("\nProcessing complete. Results saved to 'similarity_analysis_results.xlsx'")

if __name__ == "__main__":
    process_similarity_matrix()
