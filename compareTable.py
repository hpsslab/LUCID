import os
import sys
from analyze import analyze, analyze2, analyze3
import re
import time
import json
import random

# Path to the folder containing the files
folder_path = str(sys.argv[1])
output = str(sys.argv[2])

# Initilize catagories list
catList = ["advantages", "approach", "materials", "metrics", "problem", "processes"]


response = ""
for i in range(len(catList)):
    tempRes= analyze3(folder_path, i)
    print(tempRes)
    response += (tempRes + "\n\n")
    time.sleep(32)



# Save results to a file (optional)
with open(output+".csv", "w") as f:
    f.write(response)