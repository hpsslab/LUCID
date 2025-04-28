# LUCID: Literature Understanding and Consolidation for Intelligent Discovery
## Setup
- Use `pip install -r requirements.txt` to install python dependencies (using a virtual environment is recommended).
- Set the `GEMINI_API_KEY` environment variable to your [Gemini API key](https://aistudio.google.com/app/apikey).
## Configuration
### CQ.dat
Contains the list of competency questions.
### config.cfg
Defines Gemini model parameters.
### schema.py
Defines the JSON schema and LLM prompt used by *parse.py*.
## Usage
### Question Answering
- Generates answers to the CQs based on the provided PDF paper.
- `python qa.py <input_paper> <answers>`
### Answer Parsing
- Generates a JSON object based on the provided CQ answers and the ontology defined in *schema.py*.
- `python parse.py <answers> <object_file>`
### Analyze
- Compares 2 JSON objects, generating a *similarity metric* between 0 and 1 for each field. Also provides an explanation for the metric.
- `python analyze.py <object1> <object2>`