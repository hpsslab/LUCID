from pathlib import Path
from qa import answerCQs
from exceptions.extensionError import ExtensionError
from exceptions.duplicateError import DuplicateError
from exceptions.snoError import ShouldNotOccurError
from argparse import ArgumentParser, Namespace

def addPathsToKG(pathList : list[Path]) -> None:
    '''Takes in a list of paths to articles that need to be added into the KG. '''
    for path in pathList:
        ''' Add the contents to the KG, checking for errors along the way. '''
        try:
            addPathToKG(path)
        
        except FileNotFoundError as e:
            ''' When the file does not exist, just let the user know, skip, and move on. '''
            print(f"{e}. Skipping {path.name} and moving on.")

        except ShouldNotOccurError as e:
            ''' Used when the function reaches a code block that it should never reach. '''
            print(f"{e}. Skipping {path.name} and moving on.")

def addPathToKG(path: Path) -> None:
    ''' Takes in an individual path and adds the associated article to the KG. '''
    
    if not path.exists():
        raise FileNotFoundError(f"FileNotFoundError: {path.name} does not exist")

    if path.is_file():
        ''' Add article to KG, base case. '''
        try:
            cqAnswers : str = answerCQs(path)
        
        except ExtensionError as e:
            ''' Certain filetypes aren't added to the KG. For now, only pdf files are supported. TODO: add other filetype functionality later? '''
            print(f"{e}. Skipping {path.name} and moving on.")

        except DuplicateError as e:
            ''' Don't allow files to be added into the KG multiple times. '''
            print(f"{e}. Skipping {path.name} and moving on.")
            
    elif path.is_dir():
        ''' Recursively get to the files if a directory path was provided. '''
        for root, _, files in path.walk():
            ''' NOTE: loop through files only. A loop through directories results in double counting due to how walk works (learned that the hard way). '''
            for file in files:
                addPathToKG(root.joinpath(file))
    
    else:
        raise ShouldNotOccurError() 

if __name__ == "__main__":
    '''
    If we want to add to the KG directly from this file, we can do that. 
    The below parser has the same logic as the subparser in main, but relocated here for convenience.
    '''

    parser : ArgumentParser = ArgumentParser(prog = "addToKG.py",
                                             usage = "python3 addToKG.py {filepath|folderpath}+",
                                             description = "Adds a list of paths containing documents into the knowledge graph")
    parser.add_argument("paths", 
                        type = Path, 
                        nargs = "+", 
                        action = "extend", 
                        metavar = "path", 
                        help = "The list of file or folder paths to add to the KG")

    arguments : Namespace = parser.parse_args()

    addPathsToKG(arguments.paths)
