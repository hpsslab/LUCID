import sys, pymupdf
import gemini

# parse paper into text
with pymupdf.open(sys.argv[1]) as doc:
    paper = chr(10).join([page.get_text() for page in doc])

# read CQs
with open("CQ.dat") as file:
    CQs = [line.rstrip() for line in file]

# build prompt
prompt = "Answer each of the following questions for the provided paper: \n" + chr(10).join(CQs) + "\nPaper:\n" + paper
response = gemini.generate(prompt)

# write to file
with open(sys.argv[2], 'w') as out:
    out.write(response)