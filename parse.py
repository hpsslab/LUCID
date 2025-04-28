import sys
import gemini
from schema import Schema, relations

# open answer file
with open(sys.argv[1]) as file:
    answers = file.read()

# build prompt
prompt = "Using the following answers as a reference: \"" + answers + "\" find a value for each [LABEL] which satisfies the following sentences:\n"

# add relations to the prompt
for i, relation in enumerate(relations):
    prompt += f'{i+1}. {relation}\n'

prompt += "use the label values to output a single json object."

response = gemini.generate(prompt, "application/json", list[Schema])

# write to file
with open(sys.argv[2], 'w') as out:
    out.write(response)