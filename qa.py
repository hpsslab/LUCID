import pymupdf
import gemini
import cq
from asyncio import Lock, sleep
from xxhash import xxh128
from exceptions.extensionError import ExtensionError
from exceptions.duplicateError import DuplicateError
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

        except DuplicateError as e:
            ''' Don't allow files to be added into the KG multiple times. '''
            print(f"{e}. Skipping {path.name} and moving on.")

        except ShouldNotOccurError as e:
            ''' Used when the function reaches a code block that it should never reach. '''
            print(f"{e}. Skipping {path.name} and moving on.")


    return cqAnswers

def isInKG(documentPath : Path, dbPath : Path) -> bool:
    ''' 
    Checks our database (given by dbPath) and checks whether the contents of the documentPath file are already there. 
    Will create the DB if it doesn't exist yet.
    NOTE: implementation is only with a basic txt file rn, so it isn't fancy enough for things like ACID properties or reliable in a shared env.
    TODO: expand database functionality later outside txt?
    TODO: imrpove search algorithm (currently a simple linear search on the txt file)
    '''

    if not dbPath.exists():
        '''
        Make the DB file if it doesn't exist yet, write dummy value in just to make logic adding future values in easier.
        TODO: implement logic for constructing a DB for non txt database?
        '''
        if dbPath.suffix == ".txt":
            with open(dbPath, 'w') as file:
                file.write("0" * 32)

            return False

        else:
            ''' For now, don't let the program work properly if we get a non txt database passed in. '''
            raise ExtensionError(dbPath.suffix)

    else:
        if dbPath.suffix == ".txt":
            '''
            Hash the file and add it to our database to make sure we don't readd the contents into the KG twice.
            Shoutout to ChatGPT for the recommendation for a non cryptographic Python hash library and the associated code.
            '''
            hash_object : xxh3_128 = xxh128()
            
            with open(documentPath, 'rb') as doc:
                while chunk := doc.read(2 ** 32):
                    hash_object.update(chunk)        

            ''' Check for hash in database and return appropriate boolean value. '''
            hash_result : str = hash_object.hexdigest()
            
            with open(dbPath, 'r') as dbFile:
                for line in dbFile:
                    ''' Strip the line; there will be a newline otherwise '''
                    if line.strip() == hash_result:
                        return True
                    else:
                        continue

            return False

        else:
            ''' For now, don't let the program work properly if we get a non txt database passed in. '''
            raise ExtensionError(dbPath.suffix)

async def answerCQs(documentPath : Path, lock : Lock, dbPath : Path = Path("kgContents.txt"), questionList : list[str] = cq.COMPETENCY_QUESTIONS) -> str:
    ''' 
    Takes in:
    1. a path to an article 
    2. a path to a database containing a set of hashed files that have already been included in the KG. GOAL: don't add a file to the KG twice.
    3. the list of competency questions
    From there, answers the CQs in the question list.
    '''
    
    if not documentPath.exists():
        raise FileNotFoundError(f"FileNotFoundError: {documentPath.name} does not exist")

    if documentPath.suffix == ".pdf":
        ''' Check whether or not the document is already in the KG. '''
        ''' Need the lock to check the database and write to database. '''
        async with lock:
            if isInKG(documentPath, dbPath):
                raise DuplicateError(documentPath.name)

            else:
                ''' 
                Hash the file and add it to our database to make sure we don't readd the contents into the KG twice. 
                Shoutout to ChatGPT for the recommendation for a non cryptographic Python hash library and the associated code.
                '''
                hash_object : xxh3_128 = xxh128()            
                with open(documentPath, 'rb') as doc:
                    while chunk := doc.read(2 ** 32):
                        hash_object.update(chunk)

                ''' Add hashed file into our DB. '''                
                with open(dbPath, 'a') as dbFile:
                    dbFile.write(f"\n{hash_object.hexdigest()}")
        
        ''' Read PDF into a usable string. '''
        with pymupdf.open(documentPath) as doc:
            paper : str = chr(10).join([page.get_text() for page in doc])
        
        print(f"Adding {documentPath.name} to the KG.")
        await sleep(len(documentPath.name))
        print(f"Added {documentPath.name} to the KG.")

        '''
        # build prompt
        prompt : str = "Answer each of the following questions for the provided paper: \n" + chr(10).join(questionList) + "\nPaper:\n" + paper
        response : str = gemini.generate(prompt)

        # write to file
        with open(sys.argv[2], 'w') as out:
            out.write(response)

        return response
        '''

        return "Placeholder for future LLM output"

    else:
        ''' For now, don't let the program work properly if we get a non pdf document passed in. '''
        raise ExtensionError(documentPath.suffix)

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
