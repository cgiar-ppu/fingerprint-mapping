import pandas as pd
import os

def analyze_center_matches():
    # Read the original mapping file
    old_mapping_df = pd.read_excel(os.path.join('input', 'Bilaterals-Mapping-Processed-30May.xlsx'))
    
    # Read the match results
    matched_df = pd.read_excel(os.path.join('input', 'projects_with_matches.xlsx'))
    unmatched_df = pd.read_excel(os.path.join('input', 'projects_without_matches.xlsx'))
    
    # Create a set of all projects that were considered (after filtering low similarity ones)
    considered_projects = set(matched_df['Project']) | set(unmatched_df['Project'])
    
    # Create a mapping of project to center, but only for considered projects
    project_center_map = {}
    for project, center in old_mapping_df.groupby('Project Name')['Center'].first().items():
        if project in considered_projects:
            project_center_map[project] = center
    
    # Initialize counters for each center
    center_stats = {}
    
    # Count matches and total projects for each center
    for project, center in project_center_map.items():
        if center not in center_stats:
            center_stats[center] = {'matched': 0, 'total': 0}
        
        center_stats[center]['total'] += 1
        if project in matched_df['Project'].values:
            center_stats[center]['matched'] += 1
    
    # Calculate percentages and create results DataFrame
    results = []
    total_considered = 0
    total_matched = 0
    
    for center, stats in center_stats.items():
        match_percentage = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
        results.append({
            'Center': center,
            'Total Projects': stats['total'],
            'Projects with Matches': stats['matched'],
            'Projects without Matches': stats['total'] - stats['matched'],
            'Match Percentage': f"{match_percentage:.1f}%"
        })
        total_considered += stats['total']
        total_matched += stats['matched']
    
    # Convert to DataFrame and sort by match percentage
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Match Percentage', ascending=False)
    
    # Print results with pandas display options set to show all rows
    pd.set_option('display.max_rows', None)
    print("\nMatch Analysis by Center:")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print(f"\nTotal Centers: {len(results_df)}")
    print(f"Total Projects Considered (after filtering low similarity): {total_considered}")
    print(f"Total Projects with Matches: {total_matched}")
    print(f"Overall Match Percentage: {(total_matched/total_considered*100):.1f}%")
    
    # Save results to Excel
    output_file = 'center_match_analysis.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    analyze_center_matches() 