from sys import exit
from pathlib import Path
from argparse import ArgumentParser, _SubParsersAction, Namespace
from addToKG import addToKG

if __name__ == "__main__":
    parser : ArgumentParser = ArgumentParser(prog = "main.py", 
                                             usage = "python3 main.py add {filepath|folderpath}+ | python3 main.py query",
                                             description = """
                                             The main program. Functionality includes adding CQ answers to the KG and querying a LLM based on a KG.
                                             """)
    
    subparsers : _SubParsersAction = parser.add_subparsers(title = "Main subcommands",
                                                           description = "Contains all available functionality for the program",
                                                           prog = "main.py",
                                                           required = True,
                                                           dest = "option",
                                                           metavar = "add|query",
                                                           help = "The selected functionality to perform (ie: add or query)")
    """
    - Add option allows us to add some articles (or group of articles in a folder) to the KG.
    1. Calls QA.py (or folder version if adding a folder) to answer CQs
    2. Pass CQ answers to KG constructor to add documents to the KG based on the ontology/schema
    """

    add_parser : ArgumentParser = subparsers.add_parser("add")
    add_parser.add_argument("paths", 
                            type = Path, 
                            nargs = '+', 
                            action = "extend", metavar = "path", 
                            help = "The list of file or folder paths to add to the KG.")
    
    """
    - Query option allows us to query a LM using the KG as a tool
    1. Creates LangChain agent to answer question (using KG as a reference)
    2. Answers questions in a while loop
    """

    query_parser : ArgumentParser = subparsers.add_parser("query")

    arguments : Namespace = parser.parse_args()

    if arguments.option == "add":
        print("Adding")

        for path in arguments.paths:
            if not path.exists():
                print(f"{path} does not exist. Skipping {path} and moving on.")
                continue
            
            else:
                addToKG(path)
        
        exit(0)


    elif arguments.option == "query":
        print("Querying")

        while True:
            query : str = str(input("Input a query (enter 'q' to quit): "))
            
            if query == 'q':
                break

            else:
                print(f"Asking LLM {query}")

        exit(0)


    else:
        print("SNO: Invalid option provided. Exiting program.")
        exit(1)
