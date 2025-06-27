from sys import exit
from pathlib import Path
from argparse import ArgumentParser, _SubParsersAction, Namespace
from addToKG import addPathsToKG

if __name__ == "__main__":
    '''
    Make a parser to obtain arguments from command line to run program.
    Main program includes subparsers that take either add (ie: add to KG) or query (ie: query LLM using KG)
    '''

    parser : ArgumentParser = ArgumentParser(prog = "main.py", 
                                             usage = "python3 main.py add {filepath|folderpath}+ | python3 main.py query",
                                             description = "The main program. Functionality includes adding CQ answers to the KG and querying a LLM based on a KG.")

    subparsers : _SubParsersAction = parser.add_subparsers(title = "Main subcommands",
                                                           description = "Contains all available functionality for the program",
                                                           prog = "main.py",
                                                           required = True,
                                                           dest = "option",
                                                           metavar = "add|query",
                                                           help = "The selected functionality to perform (ie: add or query)")
    '''
    - Add option allows us to add some articles (or group of articles in a folder) to the KG.
    1. Calls QA.py to answer CQs
    2. Calls parse.py to format CQ answers into a JSON matching our ontology 
    3. Pass JSON object to KG constructor to add documents to the KG based on the ontology/schema
    '''

    add_parser : ArgumentParser = subparsers.add_parser("add")
    add_parser.add_argument("paths", 
                            type = Path, 
                            nargs = '+', 
                            action = "extend", metavar = "path", 
                            help = "The list of file or folder paths to add to the KG.")
    
    '''
    - Query option allows us to query a LM using the KG as a tool
    1. Creates LangChain agent to answer question (using KG as a reference)
    2. Answers questions in a while loop
    '''

    query_parser : ArgumentParser = subparsers.add_parser("query")

    arguments : Namespace = parser.parse_args()

    '''
    Perform appropriate functionality based on whether the option is add or query
    '''

    if arguments.option == "add":
        addPathsToKG(arguments.paths)
    
        exit(0)

    elif arguments.option == "query":
        while True:
            query : str = str(input("Input a query (enter 'q' to quit): "))
            
            if query == 'q':
                break

            else:
                print(f"Placeholder: asking LLM {query}")

        exit(0)


    else:
        print("SNO: Invalid option provided. Exiting program.")
        exit(1)
