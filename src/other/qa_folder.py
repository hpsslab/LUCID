import sys, pymupdf
import gemini
import os
import time
import QA

if __name__ == "__main__":
    # get file names
    folder = sys.argv[1]
    output = sys.argv[2]
    if not os.path.exists(output):
        os.makedirs(output)

    # process papers
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):  # Assuming all files are PDFs
            path = os.path.join(folder, filename)
            response = process_paper(path, CQs)

            output_fold = os.path.join(output, f"{os.path.splitext(filename)[0]}_response.txt")
            with open(output_fold, 'w') as out:
                out.write(response)
            print("done one sleeping")
            time.sleep(35)
