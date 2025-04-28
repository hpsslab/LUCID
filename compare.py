import os
import sys
from analyze import analyze 
import re
import time
import json
import random

# Path to the folder containing the files
folder_path = str(sys.argv[1])
output = str(sys.argv[2])

# Initilize catagories list
catList = ["advantages", "approach", "materials", "metrics", "problem", "processes"]

# Get a list of all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Initialize the grid as a nested dictionary
try:
    with open(sys.argv[3], "r") as json_file:
        results_grid = json.load(json_file)
except:
    results_grid = {cat:{file1:{file2: -1.1 for file2 in files} for file1 in files} for cat in catList} 



results_tracker = {}
breaking = False
# Run all comparisons
for file1 in files:
    for file2 in files:
        if file1 != file2 and (file2,file1) not in results_tracker and results_grid["metrics"][file2][file1] == -1.1:
            
            results_tracker[(file2,file1)] = 1
            try:
                result = analyze(os.path.join(folder_path, file1), os.path.join(folder_path, file2))
            except:
                breaking = True
                break
            pattern = r"(\w+):\s([0-9]+)"
            matches = re.findall(pattern, result)
            print("pap 1 is " + file1 + ". pap2 is "+ file2)
            print(result)
            print(matches)
            
            for cat, value in matches:
                if cat in catList:
                    results_grid[cat][file1][file2] = int(value)*0.01
                    results_grid[cat][file2][file1] = int(value)*0.01
            time.sleep(32)
        elif file1 == file2:
            for cat in catList:
                results_grid[cat][file2][file1] = 1.00
    if breaking: break


# Print the grid in a readable format
print("Comparison Results Grid:")
header = "\t, " + ", ".join(files)

for cat in catList:
    print(cat)
    print(header)
    for file1 in files:
        row = [str(results_grid[cat][file1][file2])+", " for file2 in files]
        print(f"{file1}, " + "".join(map(str, row)))

# Save results to a file (optional)
with open(output+".csv", "w") as f:
    for cat in catList:
        f.write(cat+", \n")
        f.write(header+", \n")
        for file1 in files:
            row = [str(results_grid[cat][file1][file2])+", " for file2 in files]
            f.write(f"{file1}, " + "".join(map(str, row)) + "\n")

#save dic incase more processing needed
with open(output+".json", "w") as json_file:
    json.dump(results_grid, json_file, indent=4)