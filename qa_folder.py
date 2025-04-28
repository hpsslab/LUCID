import sys, pymupdf
import gemini
import os
import time


def process_paper(file_path, CQs):
    # Parse paper into text
    with pymupdf.open(file_path) as doc:
        paper = chr(10).join([page.get_text() for page in doc])

    # Build prompt
    prompt = "Answer each of the following questions for the provided paper: \n" + chr(10).join(CQs) + "\nPaper:\n" + paper
    response = gemini.generate(prompt)

    return response

# get file names
folder = sys.argv[1]
output = sys.argv[2]
if not os.path.exists(output):
    os.makedirs(output)

# read CQs
with open("CQ.dat") as file:
    CQs = [line.rstrip() for line in file]

#process papers
for filename in os.listdir(folder):
    if filename.endswith(".pdf"):  # Assuming all files are PDFs
        path = os.path.join(folder, filename)
        response = process_paper(path, CQs)

        output_fold = os.path.join(output, f"{os.path.splitext(filename)[0]}_response.txt")
        with open(output_fold, 'w') as out:
            out.write(response)
        print("done one sleeping")
        time.sleep(35)
