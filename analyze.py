import sys
import gemini
import os
import re

#compares 2 papers
def analyze (paper1, paper2):
    # open object files
    with open(paper1) as file:
        obj1 = file.read()
    with open(paper2) as file:
        obj2 = file.read()

    # build prompt
    prompt = ("Given the following JSON objects:\n" + obj1 + "\n" + obj2 +
            "For each field, evaluate the similarity of the corresponding data in the two objects. " 
            "Do not compare based on language similarity, prioritize similar entities and scientific compatability. "
            "Use a number from 0 to 100 to indicate the similarity, with 100 being identical. "
            "Also include a breif explanation of the given score."
            "List the name of the object in the schema(i.e. advantages, approach, materials, metrics, problem, processes), "
            "then the number, then the explination. For example, materials: 0.000. explain here."
            "Do not compare anything else")

    response = gemini.generate(prompt)
    
    return response

#print(analyze(sys.argv[1],sys.argv[2]))

#Uses grid format for evaluation of catatgory
def analyze2(folderPath, catNum):
    #Get files
    files = sorted({f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))})

    #Get example
    with open("exampleGrid.csv", "r") as exampleFile:
        example_grid = exampleFile.read()
    
    catList = ["Advantages", "Approach", "Materials", "Metrics", "Problem", "Processes"]

    #Build prompt
    # Maybe more spesific area instead of just chemistry
    prompt = ("You are going to read several answers to previously prompted questions about chemistry research papers. "
              "For each pair of answers, evaluate the similarity on a scale of 0 to 1, 1 being identical, with 2 significant digits for the later specified category. "
              "The category will be one of the following: "+ str(catList) + ". "
              "Do not compare based on language similarity, prioritize similar entities and scientific compatibility. "
              "Provide your output in the following format and only the following format to be saved as a csv: \n") + str(example_grid) +("\n"
              "Each number in the grid represents the similarity of the 2 papers from the column and row. 0 is the base paper, the rest are citations in this paper."
              "The category goes in the top left as seen. The diagonal has 1's as these are the same papers and no need to fill out the symmetric answers below the diagonal."
              "Fill out the grid with the similarity scores for the category of "+ str(catList[catNum]) + ". Here are the papers:\n")
    
    
    # Add papers
    print( files)
    for file in files:
        with open(folderPath+"/"+file) as fileT:
            fileContents = fileT.read()
        
        prompt += ("Paper " + file + ": \n" + fileContents + "\n")
    print(prompt)
    response = gemini.generate(prompt)
    
    return response

## 2 but uses table format
def analyze3(folderPath, catNum):
    #Get files
    files = sorted({f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))})

    #Get example
    with open("exampleTable.csv", "r") as exampleFile:
        example_grid = exampleFile.read()
    
    catList = ["advantages", "approach", "materials", "metrics", "problem", "processes"]
    current_category = catList[catNum]

    # Create regex patterns for each category
    category_patterns = {
        "advantages": r"Advantages:(.*?)(?=\n\n[A-Z][a-z]+:|$)",
        "approach": r"Approach:(.*?)(?=\n\n[A-Z][a-z]+:|$)",
        "materials": r"Materials:(.*?)(?=\n\n[A-Z][a-z]+:|$)",
        "metrics": r"Metrics:(.*?)(?=\n\n[A-Z][a-z]+:|$)",
        "problem": r"Specific problem:(.*?)(?=\n\n[A-Z][a-z]+:|$)",
        "processes": r"Materials:(.*?)(?=\n\n[A-Z][a-z]+:|$)"
    }

    #Build prompt
    # Maybe more spesific area instead of just chemistry
    #prompt = ("You are going to read several answers to previously prompted questions about chemistry research papers. "
    #         "For each pair of answers, evaluate the similarity on a scale of 0 to 1, 1 being identical, with 2 significant digits for the later specified category. "
    #         "The category will be one of the following: "+ str(catList) + ". "
    #         "Do not compare based on language similarity, prioritize similar entities and scientific compatibility. "
    #         "Provide your output in the following format and only the following format to be saved as a csv: \n") + str(example_grid) +("\n"
    #         "Replace each -1.1 in the table which represents the similarity of the 2 papers from the first 2 colums of the row. Paper 0 is the base paper, the rest are citations in this paper."
    #         "The category goes in the top left as seen."
    #         "Fill out the grid with the similarity scores for the category of "+ str(catList[catNum]) + ". Here are the papers:\n")
    
    prompt = ("You are going to read several answers to previously prompted questions about chemistry research papers. For each pair of answers, evaluate the similarity specifically "
                "regarding their '"+ str(catList[catNum]) +"' on a scale of 0 to 1, where 1 represents identical "+ str(catList[catNum]) +". Provide the similarity score with 2 significant digits. \n \n "
                "Do not compare based on language similarity. Your evaluation should prioritize the similarity of the *"+ str(catList[catNum]) +"* described in each paper, focusing on the scientific "
                "compatibility and overlap of these benefits. \n \n Your output must strictly adhere to the following CSV format:\n\n "+ str(catList[catNum]) + "\n\n"+  str(example_grid) +
                "\n \n Here are the papers:\n")
    # Add papers
    print( files)
    for file in files:
        with open(folderPath+"/"+file) as fileT:
            fileContents = fileT.read()
             # Extract the relevant section using regex
            match = re.search(category_patterns[current_category], fileContents, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()
            else:
                extracted_content = "No content found for this category"
        
        prompt += ("Paper " + file + ": \n" + extracted_content + "\n")
    print(prompt)


    response = gemini.generate(prompt)
    return response
    