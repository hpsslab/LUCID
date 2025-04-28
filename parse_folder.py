import sys
import gemini
from schema import Schema, relations
import os
import time

# get file names
folder = sys.argv[1]
output = sys.argv[2]
if not os.path.exists(output):
    os.makedirs(output)

p1 = "Using the following answers as a reference: \""
p2 = "\" find a value for each [LABEL] which satisfies the following sentences:\n"
# add relations to the prompt
for i, relation in enumerate(relations):
    p2 += f'{i+1}. {relation}\n'
p2 += "use the label values to output a single json object."

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)

    # load answers into prompts
    with open(path) as file:
        answers = file.read()
    prompt = p1 + answers + p2    
    response = gemini.generate(prompt, "application/json", list[Schema])

    output_fold = os.path.join(output, f"{os.path.splitext(filename)[0]}_response.txt")
    with open(output_fold, 'w') as out:
        out.write(response)
    print("done one sleeping")
    time.sleep(35)
