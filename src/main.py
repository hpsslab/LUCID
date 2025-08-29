from asyncio import run
from sys import exit
from pathlib import Path
from argparse import ArgumentParser, _SubParsersAction, Namespace
from addToKG import addPathsToKG, initializeKG
from constructors.llm_constructor import constructLangchainLLM
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.core import PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from exceptions.snoError import ShouldNotOccurError

if __name__ == "__main__":
    '''
    Make a parser to obtain arguments from command line to run program.
    Main program includes subparsers that take either add (ie: add to KG) or query (ie: query LLM using KG)
    '''

    parser : ArgumentParser = ArgumentParser(prog = "main.py", 
                                             usage = "python3 main.py add [options] {filepath|folderpath}+ | python3 main.py query",
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

    '''
    A parser to decide which LLM to use. Not particularly useful right now since I only allow Gemini for now, but should be helpful down the line.
    '''
    add_parser.add_argument('--llm',
                            type = str,
                            nargs = '?',
                            const = None,
                            default = "gemini",
                            action = "store",
                            metavar = "llm",
                            choices = ["gemini"],
                            help = "The LLM to use when adding articles to the KG.")
    
    ''' A parser to decide which embedding model to use. '''
    add_parser.add_argument('--embed',
                        type = str,
                        nargs = '?',
                        const = None,
                        default = "gemini",
                        action = "store",
                        metavar = "embed",
                        choices = ["bert", "scibert", "gemini"],
                        help = "The LLM to use when embedding text (for the KG or for the query). This should remain the same model across program runs if the KG used is constant.")

    add_parser.add_argument("paths", 
                            type = Path, 
                            nargs = '+', 
                            action = "extend", 
                            metavar = "path", 
                            help = "The list of file or folder paths to add to the KG.")
    
    '''
    - Query option allows us to query a LM using the KG as a tool
    1. Creates LangChain agent to answer question (using KG as a reference)
    2. Answers questions in a while loop
    '''

    query_parser : ArgumentParser = subparsers.add_parser("query")

    '''
    A parser to decide which LLM to use. Not particularly useful right now since I only allow Gemini for now, but should be helpful down the line.
    '''
    query_parser.add_argument('--llm',
                            type = str,
                            nargs = '?',
                            const = None,
                            default = "gemini",
                            action = "store",
                            metavar = "llm",
                            choices = ["gemini"],
                            help = "The LLM to use when querying the KG.")

    ''' A parser to decide which embedding model to use. '''
    query_parser.add_argument('--embed',
                        type = str,
                        nargs = '?',
                        const = None,
                        default = "gemini",
                        action = "store",
                        metavar = "embed",
                        choices = ["bert", "scibert", "gemini"],
                        help = "The LLM to use when embedding text (for the KG or for the query). The model used when creating the KG and the one used to embed the query should be the same!!!")

    arguments : Namespace = parser.parse_args()

    '''
    Perform appropriate functionality based on whether the option is add or query
    '''

    if arguments.option == "add":
        run(addPathsToKG(arguments.paths, arguments.llm, arguments.embed))
    
        exit(0)

    elif arguments.option == "query":
        ''' Load the property graph into a tool that can be used by LangChain. '''
        property_graph : PropertyGraphIndex = initializeKG(arguments.llm, arguments.embed)

        property_graph_query_engine : RetrieverQueryEngine = property_graph.as_query_engine(
            include_text = True,
            show_progress = True,
            similarity_top_k = 5,
            path_depth = 1,
            num_workers = 4,
            limit = 30
        )
        
        query_engine_tool_config : IndexToolConfig = IndexToolConfig(
                query_engine = property_graph_query_engine,
                name = "knowledge_graph_query_engine",
                description = "Useful as a RAG source for answering questions in the domain of material science"
        )

        query_engine_tool = LlamaIndexTool.from_tool_config(query_engine_tool_config)

        ''' Make Langchain agent depending on selected LLM. '''
        langchain_llm : BaseChatModel = constructLangchainLLM(arguments.llm)

        langchain_agent : CompiledStateGraph = create_react_agent(
                model = langchain_llm,
                tools = [query_engine_tool]
        )

        ''' Use infinite loop for querying for now (open to improvements later). '''
        while True:
            query : str = str(input("Input a query (enter 'q' to quit): "))
            
            if query == 'q':
                break

            else:
                ''' Let the agent .stream out output (or .invoke to simply output all at once). '''
                for step in langchain_agent.stream({"messages": [query]}, stream_mode="values"):
                    step["messages"][-1].pretty_print()

        exit(0)


    else:
        raise ShouldNotOccurError()
        exit(1)
