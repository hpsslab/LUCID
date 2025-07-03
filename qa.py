import pymupdf
import gemini
import cq
from asyncio import sleep
from exceptions.extensionError import ExtensionError
from exceptions.snoError import ShouldNotOccurError
from pathlib import Path

def answerCQsInPath(documentPaths : list[Path], dbPath : Path = Path("kgContents.txt"), questionList : list[str] = cq.COMPETENCY_QUESTIONS) -> list[str]:
    ''' Takes in a list of paths to articles that need answered CQs and returns a list containing the answers for each document. '''
    
    cqAnswers : list[str] = []
    
    ''' Loop through paths and answer CQs ''' 
    for path in documentPaths:
        try:
            if path.is_file():
                ''' Base Case: answer the CQs for the file. '''
                cqAnswer : str = answerCQs(path)
                cqAnswers.append(cqAnswer)

            elif path.is_dir():
                ''' Recursively get to the files if a directory path was provided. '''
                for root, _, files in path.walk():
                    ''' NOTE: loop through files only. A loop through directories results in double counting due to how walk works (learned that the hard way) '''
                    for file in files:
                        cqAnswer : str = answerCQs(root.joinpath(file), dbPath, questionList)
                        cqAnswers.append(cqAnswer)

            else:
                raise ShouldNotOccurError()

        except FileNotFoundError as e:
            ''' When the file does not exist, just let the user know, skip, and move on. '''
            print(f"{e}. Skipping {path.name} and moving on.")

        except ExtensionError as e:
            ''' Certain filetypes aren't added to the KG. For now, only pdf files are supported. TODO: add other filetype functionality later? '''
            print(f"{e}. Skipping {path.name} and moving on.")

        except ShouldNotOccurError as e:
            ''' Used when the function reaches a code block that it should never reach. '''
            print(f"{e}. Skipping {path.name} and moving on.")

    return cqAnswers

async def answerCQs(documentPath : Path, answerPath : Path, questionList : list[str] = cq.COMPETENCY_QUESTIONS) -> None:
    ''' 
    Takes in:
    1. a path to an article 
    2. a path to write the answers to the CQs to
    3. the list of competency questions (optionally, default is CQs in cq.py file)
    From there, answers the CQs in the question list.
    '''
    
    if not documentPath.exists():
        raise ShouldNotOccurError()

    if documentPath.suffix == ".pdf":
        ''' Read PDF into a usable string. '''
        with pymupdf.open(documentPath) as doc:
            paper : str = chr(10).join([page.get_text() for page in doc])
        
        print(f"Answering {documentPath.name} CQs.")
        await sleep(len(documentPath.name))
        print(f"Answered {documentPath.name} CQs.")

        ''' construct the prompt with the necessary details. '''
        prompt : str = "Answer each of the following questions for the provided paper: \n" + chr(10).join(questionList) + "\nPaper:\n" + paper
        
        '''
        response : str = gemini.generate(prompt)
        '''
        response : str = prompt

        # write to file
        with answerPath.open('w') as out:
            out.write(response)

    else:
        ''' For now, don't let the program work properly if we get a non pdf document passed in. '''
        raise ShouldNotOccurError()

if __name__ == "__main__":
    '''
    If we want to get CQs answered directly from this file, we can do that (mostly for debugging purposes).
    The below parser has the same logic as the subparser in main, but relocated here for convenience.
    '''

    parser : ArgumentParser = ArgumentParser(prog = "qa.py",
                                             usage = "python3 qa.py {filepath|folderpath}+",
                                             description = "Answers CQs for a list of paths containing documents")
    parser.add_argument("paths",
                        type = Path,
                        nargs = "+",
                        action = "extend",
                        metavar = "path",
                        help = "The list of file or folder paths to answer CQs for")

    arguments : Namespace = parser.parse_args()

    answerCQsInPath(arguments)
