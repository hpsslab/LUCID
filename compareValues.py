#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from collections import defaultdict

def parse_grid_file(filename):
    """Parse the grid file and return a nested dictionary structure."""
    grids = defaultdict(dict)
    current_grid = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("Grid:"):
                current_grid = line.split("Grid:")[1].strip()
                # Skip the next line with column headers
                next(f)
            elif current_grid and line:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if not parts:
                    continue
                
                row_key = parts[0]
                values = [float(v) if v.replace('.', '', 1).isdigit() else v for v in parts[1:]]
                grids[current_grid][row_key] = values
    
    return grids

def calculate_differences(grids1, grids2):
    """Calculate differences between corresponding values in two grid structures."""
    differences = []
    matched_pairs = 0
    total_values = 0
    zeros = 0
    all_grids = set(grids1.keys()).intersection(set(grids2.keys()))
    
    for grid in sorted(all_grids):
        grid1 = grids1[grid]
        grid2 = grids2[grid]
        
        all_rows = set(grid1.keys()).intersection(set(grid2.keys()))
        
        for row in sorted(all_rows):
            values1 = grid1[row]
            values2 = grid2[row]
            
            if len(values1) != len(values2):
                continue
            
            for v1, v2 in zip(values1, values2):
                differences.append(abs(v1 - v2))
                if abs(v1 - v2) > .5:
                    print(grid)
                    print(row)
                matched_pairs += 1
            total_values += len(values1)
    return differences, matched_pairs, total_values

def print_statistics(differences, matched_pairs, total_values, output_file=None):
    """Print statistics about the differences."""
    output = sys.stdout if output_file is None else open(output_file, 'w')
    
    if not differences:
        print("No comparable values found between the files.", file=output)
        return
    
    diffs = np.array(differences)
    
    print("\nDifference Statistics:", file=output)
    print(f"Average absolute difference: {np.mean(diffs):.4f}", file=output)
    print(f"Median absolute difference: {np.median(diffs):.4f}", file=output)
    print(f"Maximum difference: {np.max(diffs):.4f}", file=output)
    print(f"Variance of differences: {np.var(diffs):.4f}", file=output)
    print(f"Standard deviation: {np.std(diffs):.4f}", file=output)
    print(f"T-score: {np.mean(diffs)/(np.std(diffs)/np.sqrt(len(diffs))):.4f}", file=output)
    

    # Print histogram of differences
    print("\nDifference Distribution:", file=output)
    hist, bins = np.histogram(diffs, bins=10, range=(0, np.max(diffs)))
    for i in range(len(hist)):
        lower = bins[i]
        upper = bins[i+1]
        print(f"{lower:.2f}-{upper:.2f}: {hist[i]} ({hist[i]/len(diffs)*100:.1f}%)", file=output)
    
    if output_file is not None:
        output.close()

def main():
    parser = argparse.ArgumentParser(description="Compare two grid files and calculate difference statistics.")
    parser.add_argument("file1", help="First file to compare")
    parser.add_argument("file2", help="Second file to compare")
    parser.add_argument("-o", "--output", help="Output file for results (default: stdout)")
    
    args = parser.parse_args()
    
    print(f"Comparing {args.file1} and {args.file2}...", file=sys.stderr)
    
    grids1 = parse_grid_file(args.file1)
    grids2 = parse_grid_file(args.file2)
    
    differences, matched_pairs, total_values = calculate_differences(grids1, grids2)
    print_statistics(differences, matched_pairs, total_values, args.output)
    
    print("Comparison complete.", file=sys.stderr)

if __name__ == "__main__":
    main()