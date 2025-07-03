from asyncio import run, gather, Lock
from pathlib import Path
from qa import answerCQs
from parse import parseCQAnswers
from xxhash import xxh128
from exceptions.extensionError import ExtensionError
from exceptions.snoError import ShouldNotOccurError
from exceptions.duplicateError import DuplicateError
from argparse import ArgumentParser, Namespace

def hashDocument(documentPath : Path) -> str:
    ''' 
    Takes in a path to a file and hashes its binary. 
    Shoutout to ChatGPT for the recommendation for a non cryptographic Python hash library and the associated code.
    '''
    hash_object : xxh3_128 = xxh128()

    with open(documentPath, 'rb') as doc:
        while chunk := doc.read(2 ** 32):
            hash_object.update(chunk)

    return hash_object.hexdigest()

def isInKG(documentPath : Path, dbPath : Path = Path("kgContents.txt")) -> bool:
    '''
    Checks our database (given by dbPath) and checks whether the contents of the documentPath file are already there.
    Will create the DB if it doesn't exist yet.
    NOTE: implementation is only with a basic txt file rn, so it isn't fancy enough for things like ACID properties or reliable in a shared env.
    TODO: expand database functionality later outside txt?
    TODO: imrpove search algorithm (currently a simple linear search on the txt file)?
    '''
    if not dbPath.exists():
        '''
        Make the DB file if it doesn't exist yet, write hash of the passed document into DB as first value.
        TODO: implement logic for constructing a DB for non txt database?
        '''
        if dbPath.suffix == ".txt":
            hash_result : str = hashDocument(documentPath)

            ''' Add the hash to our DB. '''
            with open(dbPath, 'w') as dbFile:
                dbFile.write(f"{hash_result}")

            return False

        else:
            ''' For now, don't let the program work properly if we get a non txt database passed in. '''
            raise ExtensionError(dbPath.suffix, "database")

    else:
        if dbPath.suffix == ".txt":
            ''' Hash the file and add it to our database (if necessary) to make sure we don't read the contents into the KG twice. '''
            hash_result : str = hashDocument(documentPath)
            
            with open(dbPath, 'r') as dbFile:
                for line in dbFile:
                    ''' Strip the line; there will be a newline otherwise. '''
                    if line.strip() == hash_result:
                        return True
                    else:
                        continue

            ''' Add the hash to our DB if it wasn't there. '''
            with open(dbPath, 'a') as dbFile:
                dbFile.write(f"\n{hash_result}")

            return False

        else:
            ''' For now, don't let the program work properly if we get a non txt database passed in. '''
            raise ExtensionError(dbPath.suffix, "database")

async def addPathsToKG(pathList : list[Path]) -> None:
    '''Takes in a list of paths to articles that need to be added into the KG. '''
    
    ''' Make a lock that every asynchronous process can share once we start executing. Can't have race conditions when doing DB ops. '''
    databaseLock : Lock = Lock()
    
    ''' Asynchronously execute the process of adding each path into the knowledge graph, making sure exceptions don't stop the entire execution. '''
    await gather(*(addPathToKG(path, databaseLock) for path in pathList), return_exceptions = True)

async def addPathToKG(documentPath: Path, lock : Lock, cqAnswerPath : Path = Path("cq_answers"), parsedPath : Path = Path("parsed_cq_answers")) -> None:
    ''' Takes in an individual path and adds the associated article to the KG. '''

    ''' Make sure passed document exists before anything. '''
    try:
        if not documentPath.exists():
            raise FileNotFoundError(f"FileNotFoundError: {documentPath.name} does not exist")
    
    except FileNotFoundError as e:
        print(f"{e}. Skipping {documentPath.name} and moving on.")
    
    if documentPath.is_file():
        ''' Add article to KG, base case. '''
        try:
            ''' We are only adding .pdf files into our corpus for now. TODO: add more later? '''
            if documentPath.suffix not in {".pdf"}:
                raise ExtensionError(documentPath.suffix, "knowledge graph")
           
            ''' 
            Make sure file isn't already in KG. Add to our DB that it is if we are adding to KG for the first time. 
            Make sure folder where CQ answers and parsed CQ answers will be put exist too.
            '''
            async with lock:
                inKG : bool = isInKG(documentPath)

                if not cqAnswerPath.exists():
                    cqAnswerPath.mkdir()
                
                if not parsedPath.exists():
                    parsedPath.mkdir()
            
            if inKG:
                raise DuplicateError(documentPath.name)

            else:
                ''' These are the paths where the LLM will store its answers to the CQs and parse the CQs into JSONs respectively. '''
                currentCQAnswerPath : Path = Path(cqAnswerPath.joinpath(f"{documentPath.stem}_answers.txt"))
                currentParsedAnswerPath : Path = Path(parsedPath.joinpath(f"{documentPath.stem}_answers.json"))
                
                await answerCQs(documentPath, currentCQAnswerPath)
                await parseCQAnswers(currentCQAnswerPath, currentParsedAnswerPath)
        
        except ExtensionError as e:
            ''' 
            Certain filetypes aren't allowed to be a DB/KG file (only .txt and .pdf for now respectively). 
            TODO: add other database/KG functionality later? 
            '''
            print(f"{e}. Skipping {documentPath.name} and moving on.")

        except FileNotFoundError as e:
            ''' Account for files not existing. '''
            print(f"{e}. Skipping {documentPath.name} and moving on.")

        except DuplicateError as e:
            ''' Used when we are trying to add a file to the KG that is already there. '''
            print(f"{e}. Skipping {documentPath.name} and moving on.")

        except ShouldNotOccurError as e:
            ''' Used to debug and find code execution paths that should not happen. '''
            print(f"{e}. Skipping {documentPath.name} and moving on.")

    elif documentPath.is_dir():
        ''' Recursively get to the files if a directory path was provided. '''
        for root, _, files in documentPath.walk():
            '''
            GOAL: add each file in the directory into the KG asynchronously.
            NOTE: loop through files only. Looping through directories leads to double counting (walk() recursively traverses directories). 
            '''
            await gather(*(addPathToKG(root.joinpath(file), lock) for file in files), return_exceptions = True)
    
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

    run(addPathsToKG(arguments.paths))
