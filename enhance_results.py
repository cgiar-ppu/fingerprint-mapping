import pandas as pd
import os

def enhance_results():
    # Read the similarity analysis results
    similarity_df = pd.read_excel('similarity_analysis_results.xlsx')
    
    # Read the original mapping file
    bilaterals_df = pd.read_excel(os.path.join('input', 'Bilaterals-Mapping-Processed-30May.xlsx'))
    
    # Create a mapping of project name to center
    project_center_map = bilaterals_df.groupby('Project Name')['Center'].first()
    
    # Add the Center column right after the Project column
    similarity_df.insert(1, 'Center', similarity_df['Project'].map(project_center_map))
    
    # Save to a new Excel file
    output_file = 'enhanced_similarity_results.xlsx'
    similarity_df.to_excel(output_file, index=False)
    
    print(f"\nEnhanced results saved to: {output_file}")
    print(f"Added columns:")
    print("- Center (from Bilaterals-Mapping-Processed-30May.xlsx)")
    print(f"\nTotal projects processed: {len(similarity_df)}")
    print(f"Projects with Center information: {similarity_df['Center'].notna().sum()}")
    
    # Print a sample of the enhanced data
    print("\nFirst few rows of the enhanced dataset:")
    print("=" * 100)
    print(similarity_df.head().to_string())

if __name__ == "__main__":
    enhance_results() 