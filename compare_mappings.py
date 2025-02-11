import pandas as pd
import numpy as np
import os

def standardize_program_name(name):
    # Mapping dictionary for program names
    program_mapping = {
        # New mapping names (from similarity analysis) -> Standard names
        'Better Diets and Nutrition': 'Nutrition and Diets MP',
        'Breeding For Tomorrow': 'Genetic Innovation MP',
        'Capacity Sharing': 'Capacity Sharing Accelerator',
        'Climate Action': 'Climate MP',
        'Digital Transformation': 'Digital and Data Accelerator',
        'Food Frontiers and Security': 'Frontiers MP',
        'Gender Equality and Inclusion': 'Gender Equality and Social inclusion Accelerator',
        'Genebanks': 'Genetic Innovation MP',
        'Multifunctional Landscapes': 'Landscapes MP',
        'Policy Innovations': 'Policy MP',
        'Scaling for Impact': 'Catalyzing Impact MP',
        'Sustainable Animal and Aquatic Foods': 'Animal and Aquatic-based Foods MP',
        'Sustainable Farming': 'Sustainable Farming MP',
        
        # Ensure old names map to themselves
        'Nutrition and Diets MP': 'Nutrition and Diets MP',
        'Genetic Innovation MP': 'Genetic Innovation MP',
        'Capacity Sharing Accelerator': 'Capacity Sharing Accelerator',
        'Climate MP': 'Climate MP',
        'Digital and Data Accelerator': 'Digital and Data Accelerator',
        'Frontiers MP': 'Frontiers MP',
        'Gender Equality and Social inclusion Accelerator': 'Gender Equality and Social inclusion Accelerator',
        'Landscapes MP': 'Landscapes MP',
        'Policy MP': 'Policy MP',
        'Catalyzing Impact MP': 'Catalyzing Impact MP',
        'Animal and Aquatic-based Foods MP': 'Animal and Aquatic-based Foods MP',
        'Sustainable Farming MP': 'Sustainable Farming MP'
    }
    return program_mapping.get(name, name)

def print_unique_values():
    # Read files
    old_df = pd.read_excel(os.path.join('input', 'Bilaterals-Mapping-Processed-30May.xlsx'))
    new_df = pd.read_excel("similarity_analysis_results.xlsx")
    
    # Get unique values
    old_programs = sorted(old_df['Megaprograms'].unique())
    new_programs = sorted(set(new_df['Most Similar Program'].unique()) | 
                        set(new_df['Second Most Similar Program'].unique()))
    
    print("\nUnique programs in old mapping:")
    for p in old_programs:
        print(f"- {p}")
        
    print("\nUnique programs in new mapping:")
    for p in new_programs:
        print(f"- {p}")
        
    print("\nStandardized mapping:")
    for new_name in new_programs:
        print(f"'{new_name}' -> '{standardize_program_name(new_name)}'")

def load_old_mapping():
    # Read the old mapping file from input folder
    file_path = os.path.join('input', 'Bilaterals-Mapping-Processed-30May.xlsx')
    old_df = pd.read_excel(file_path)
    
    # Remove 'Control To match 100%' and 'Other' from old mapping
    old_df = old_df[~old_df['Megaprograms'].isin(['Control To match 100%', 'Other'])]
    
    # Standardize program names
    old_df['Megaprograms'] = old_df['Megaprograms'].apply(standardize_program_name)
    
    # Group by project name and aggregate megaprograms into lists
    grouped = old_df.groupby('Project Name')['Megaprograms'].agg(list).reset_index()
    
    # Convert to dictionary for easier lookup
    old_mapping = {row['Project Name']: set(row['Megaprograms']) for _, row in grouped.iterrows()}
    return old_mapping

def load_new_mapping():
    # Read the new mapping file
    new_df = pd.read_excel("similarity_analysis_results.xlsx")
    
    # Create a dictionary with project names as keys and sets of standardized programs as values
    new_mapping = {}
    for _, row in new_df.iterrows():
        programs = {
            standardize_program_name(row['Most Similar Program']),
            standardize_program_name(row['Second Most Similar Program'])
        }
        new_mapping[row['Project']] = programs
    
    return new_mapping

def calculate_overlap():
    # First print unique values
    print_unique_values()
    
    old_mapping = load_old_mapping()
    new_mapping = load_new_mapping()
    
    # Print unique standardized values from both mappings
    print("\nUnique standardized programs in old mapping:")
    unique_old = set()
    for programs in old_mapping.values():
        unique_old.update(programs)
    for p in sorted(unique_old):
        print(f"- {p}")
        
    print("\nUnique standardized programs in new mapping:")
    unique_new = set()
    for programs in new_mapping.values():
        unique_new.update(programs)
    for p in sorted(unique_new):
        print(f"- {p}")
    
    projects_with_matches = []
    projects_without_matches = []
    total_projects = 0
    projects_with_at_least_one_match = 0
    
    # Compare each project in the new mapping
    for project, new_programs in new_mapping.items():
        if project in old_mapping:
            total_projects += 1
            old_programs = old_mapping[project]
            
            # Count matches between old and new programs
            matches = len(new_programs.intersection(old_programs))
            
            # Create detail record
            detail_record = {
                'Project': project,
                'Old Programs': ', '.join(sorted(old_programs)),
                'New Programs': ', '.join(sorted(new_programs)),
                'Number of Matches': matches,
                'Match Percentage': f"{(matches/len(new_programs))*100:.1f}%"
            }
            
            # Add to appropriate list
            if matches > 0:
                projects_with_matches.append(detail_record)
                projects_with_at_least_one_match += 1
            else:
                projects_without_matches.append(detail_record)
    
    # Calculate overall percentage of projects with at least one match
    overall_percentage = (projects_with_at_least_one_match / total_projects * 100) if total_projects > 0 else 0
    
    # Create detailed reports
    matched_df = pd.DataFrame(projects_with_matches)
    unmatched_df = pd.DataFrame(projects_without_matches)
    
    # Save reports
    output_path_matched = os.path.join('input', 'projects_with_matches.xlsx')
    output_path_unmatched = os.path.join('input', 'projects_without_matches.xlsx')
    
    matched_df.to_excel(output_path_matched, index=False)
    unmatched_df.to_excel(output_path_unmatched, index=False)
    
    print(f"\nProjects with at least one match: {projects_with_at_least_one_match}")
    print(f"Total projects compared: {total_projects}")
    print(f"Percentage of projects with at least one match: {overall_percentage:.1f}%")
    print(f"\nDetailed reports saved to:")
    print(f"- Projects with matches: '{output_path_matched}'")
    print(f"- Projects without matches: '{output_path_unmatched}'")

if __name__ == "__main__":
    calculate_overlap() 