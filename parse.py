import gemini
from asyncio import sleep
from pathlib import Path
from schema import Schema, relations

async def parseCQAnswers(answerPath : Path, parsedOutputPath : Path) -> None:
    ''' Takes in a path to a file with CQ answers and parses them into a JSON object, storing the output in the specified path. '''
    with answerPath.open('r') as file:
        answers : str = file.read()
    
    ''' Include all relevant information in the prompt. '''
    prompt : str = "Using the following answers as a reference: \"" + answers + "\" find a value for each [LABEL] which satisfies the following sentences:\n"

    ''' Add relations from our schema into the prompt. '''
    for i, relation in enumerate(relations):
        prompt += f'{i+1}. {relation}\n'

    prompt += "use the label values to output a single json object."
    
    '''
    response = gemini.generate(prompt, "application/json", list[Schema])
    '''
    response : str = prompt

    print(f"Parsing answer file {answerPath.name}.")
    await sleep(10)
    print(f"Parsed answer file {answerPath.name}.")

    # write to file
    with parsedOutputPath.open('w') as out:
        out.write(response)

if __name__ == "__main__":
    pass
