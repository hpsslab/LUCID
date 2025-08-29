# LUCID: Literature Understanding and Consolidation for Intelligent Discovery
## Setup
- Use `pip install -r requirements.txt` to install python dependencies (using a virtual environment is recommended).
- `python3 -m venv venv` 
- `source venv/bin/activate` 
- Set the `GEMINI_API_KEY` environment variable to your [Gemini API key](https://aistudio.google.com/app/apikey).
- Use `export GEMINI_API_KEY=[paste_key_here]` to set your key once in vm
## Configuration
### CQ.py
Contains the list of competency questions.
### config.cfg
Defines Gemini model parameters.
### schema.py
Defines the schema for the project to use. 
## Usage
### Question Answering
- Generates answers to the CQs based on the provided PDF paper(s). Does this recursively for directory paths and asynchronously.
- `python qa.py <filepath|folderpath>+`
### Answer Parsing
- Generates a JSON object based on the provided CQ answers and the ontology defined in *schema.py*. Does this recursively for directory paths and asynchronously.
- `python parse.py <filepath|folderpath>+`
### Analyze
- Compares 2 JSON objects, generating a *similarity metric* between 0 and 1 for each field. Also provides an explanation for the metric.
- `python analyze.py <object1> <object2>`

# Jackson's Additions

## Usage
### Question Answering
- Generates answers to the CQs based on the provided folder of PDF papers.
- `python qa_folder.py <input_folder> <output>`

### Answer Parsing (Not really used)
- Generates JSON objects based on the provided CQ answers and the ontology defined in *schema.py*.
- `python parse_folder.py <answer_folder> <json_object_folder>`

### Similarity Value Generation
- Takes answers from qa_folder.py and generates grid of similarity values by prompting LLM 1 pair at a time.
- `python compare.py <answers> <output_file>`

- Alternatively, make a table then reformat to a grid.
- `python compareTable.py <answers> <output_table>`
- `python tableToGrid.py <input_table> <output_grid>`

-  Note: the table comparison also isolates the answers to just the catagory in question using regular expressions. You may need to edit the output to fit the expressions in the analyze3 function. Also, I moved the "Novel fabribation techniques" to the end of the materials catagory as the questions and anwers for materials and process are combined.
-  The grid script provides every catagory answers for every question when prompting the LLM, but asks for just one catagory. 

### Compare the similarity of 2 sets of similarity scores
- Calculates the average difference and other statistics between 2 data sets.
- `python compareValues.py <Grid1> <Grid2>`

# Shaun's Additions

## Usage
### Add Files/Directories to KG
- Adds each filepath (or recursively adds everything in a directory) to the KG. Does this asynchronously. Only works on PDFs for now.
- `python3 main.py add [options] <path>+`

### Query a LLM using a KG
- Queries a LLM using the knowledge graph that has been constructed.
- `python3 main.py query [options]`
