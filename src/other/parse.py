import gemini
from asyncio import run, gather, sleep
from pathlib import Path
from schema import VALIDATION_SCHEMA
from exceptions.snoError import ShouldNotOccurError
from argparse import ArgumentParser, Namespace

async def parseCQAnswersInPath(answerPaths : list[Path], parsedAnswerPath : Path = Path("parsed_cq_answers")) -> None:
    ''' Takes in a list of paths to answered CQs (in .txt files) and parses them into .json files. '''

    ''' Make parsed answers directory if we don't have it yet. '''
    if not parsedAnswerPath.exists():
        parsedAnswerPath.mkdir()

    ''' Loop through paths and parse CQ answers. '''
    for answerPath in set(answerPaths):
        try:
            ''' Make sure passed answer document exists before anything. '''
            if not answerPath.exists():
                raise FileNotFoundError(f"FileNotFoundError: {answerPath.name} does not exist")

            if answerPath.is_file():
                ''' Base Case: parse the CQ answers for the file. '''
                await parseCQAnswers(answerPath)

            elif answerPath.is_dir():
                ''' Recursively get to the files if a directory path was provided. '''
                for root, _, files in answerPath.walk():
                    ''' NOTE: loop through files only. A loop through directories results in double counting due to how walk works (learned that the hard way) '''
                    await gather(*(parseCQAnswers(root.joinpath(file), parsedAnswerPath.joinpath(f"{file}_answers.json")) for file in files), return_exceptions = True)

            else:
                raise ShouldNotOccurError()

        except FileNotFoundError as e:
            ''' When the file does not exist, just let the user know, skip, and move on. '''
            print(f"{e}. Skipping {answerPath.name} and moving on.")

        except ExtensionError as e:
            ''' Certain filetypes aren't added to the KG. For now, only pdf files are supported. TODO: add other filetype functionality later? '''
            print(f"{e}. Skipping {answerPath.name} and moving on.")

        except ShouldNotOccurError as e:
            ''' Used when the function reaches a code block that it should never reach. '''
            print(f"{e}. Skipping {answerPath.name} and moving on.")

async def parseCQAnswers(answerPath : Path, parsedOutputPath : Path) -> None:
    ''' Takes in a path to a file with CQ answers and parses them into a JSON object, storing the output in the specified path. '''
    
    if not answerPath.exists():
        raise ShouldNotOccurError()

    if answerPath.suffix == ".txt":
        with answerPath.open('r') as file:
            answers : str = file.read()
    
        ''' Include all relevant information in the prompt. '''
        prompt : str = "Using the following answers as a reference: \"" + answers + "\" find a value for each [LABEL] which satisfies the following sentences. You are allowed to use each sentence multiple times:\n"

        ''' Add relations from our schema into the prompt. '''
        for i, triplet in enumerate(VALIDATION_SCHEMA):
            prompt += f'{i+1}. [{triplet[0]}] {triplet[1]} [{triplet[2]}]\n'

        prompt += "use the label values to output a single json object."
    
        '''
        response = gemini.generate(prompt, "application/json", list[Schema])
        '''
        response : str = prompt
        
        print(f"Parsing answer file {answerPath.name}.")
        await sleep(len(answerPath.stem))
        print(f"Parsed answer file {answerPath.name}.")

        ''' Write parsed json to output JSON file. '''
        with parsedOutputPath.open('w') as out:
            out.write(response)

    else:
        ''' For now, don't let program work properly if answers are not written out to a .txt file. '''
        raise ShouldNotOccurError()

if __name__ == "__main__":
    '''
    If we want to get CQs answers parsed directly from this file, we can do that.
    NOTE: this is mostly for debugging purposes and shouldn't be used in the actual program.
    '''

    parser : ArgumentParser = ArgumentParser(prog = "parse.py",
                                             usage = "python3 parse.py {filepath|folderpath}+",
                                             description = "Parses CQ answers for a list of paths containing CQ answers")
    parser.add_argument("paths",
                        type = Path,
                        nargs = "+",
                        action = "extend",
                        metavar = "path",
                        help = "The list of file or folder paths to parse CQ answers for")

    arguments : Namespace = parser.parse_args()

    run(parseCQAnswersInPath(arguments.paths))
