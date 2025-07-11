from asyncio import run, gather, Lock
from pathlib import Path
from os import environ
from configparser import ConfigParser
from schema import ENTITIES, RELATIONS, VALIDATION_SCHEMA
from qa import answerCQs
from parse import parseCQAnswers
from xxhash import xxh128
from exceptions.extensionError import ExtensionError
from exceptions.snoError import ShouldNotOccurError
from exceptions.duplicateError import DuplicateError
from argparse import ArgumentParser, Namespace
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.google_genai import GoogleGenAI

def initializeKG(llmChoice : str) -> any:
    ''' Create LLM depending on argument. '''
    llm = None
    
    if llmChoice == "gemini":
        config : ConfigParser = ConfigParser()
        
        config.read('config.cfg')
        
        '''
        Use the LlamaIndex LLM constructors to creatae our LLM to construct a KG with. 
        Parameters:
            model: which Gemini model should we use?
            api_key: credentials to use the model.
            temperature: how much 'freedom' should the model have to vary its output?
            max_tokens: how many tokens is the limit for model outputs?
            context_window: the context window of the model.
            max_retries: how many times should the API retry if an API error takes place?
            vertexai_config: credentials to use if we want to use via VertexAI (provided a dictionary containing a project id and a location).
            http_options: HTTP options for the LLM requests. 
            See https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L1247 for more.
            debug_config: options that make it easier to debug (ie: use cached content so that you don't charge for every call).
            See https://github.com/googleapis/python-genai/blob/main/google/genai/client.py#L89 for more.
            generation_config: configuration instructions on how to generate the response. Not using for now, change later if model not performing.
            See https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L3766 for more.
            callback_manager: used if we want to manage the asynchronous flow of the program within LlamaIndex.
            is_function_calling_model: does the model call functions (as tools) as a RAG source?
            cached_content: a cache object if we want to cache objects that the LLM will use multiple times.
            The goal here is to provide CQ answers to the model and have the LLM create triplets for a knowledge graph (once per set of CQ answers),
            so caching won't be necessary for this particular model.
        '''
        llm : GoogleGenAI = GoogleGenAI(
            model = "gemini-2.0-flash",
            api_key = environ["GEMINI_API_KEY"],
            temperature = float(config['gemini']['Temperature']),
            max_tokens = 8192,
            context_window = 1048576,
            max_retries = 3,
            vertexai_config = None,
            http_options = None, 
            debug_config = None,
            generation_config = None,
            callback_manager = None,
            is_function_calling_model = False,
            cached_content = None 
        )

    else:
        raise ShouldNotOccurError()

    '''
    Use the schema that we defined to extract our necessary information to construct the KG.
    Ask the LLM to extract KG triples from the CQ answers.
    Parameters:
        llm: the LLM to prompt to extract the entities and relations from the CQ answers.
        extract_prompt: the prompt to feed the LLM with to extract everything. 
        I think the default prompt that the library uses will work fine, but we can add more context to the prompt later if necessary.
        possible_entities & possible_relations: the entities/relations in our schema.
        possible_entity_props & possible_relation_props: a list of properties that we would want to extract about a given entity/relation.
        Leaving as None for now unless we want specific information later (would probably need to alter our approach too).
        kg_validation_schema: rules defining which entities can pair with which relations.
        strict: must we stick to this schema or can the LLM define new attributes?
        num_workers: the amount of parallel jobs we use to do everything.
        max_triplets_per_chunk: how many KG triplets can we make from each chunk from the document?
    '''
    kgExtractor : SchemaLLMPathExtractor = SchemaLLMPathExtractor(
        llm = llm,
        extract_prompt = None,            
        possible_entities = ENTITIES,
        possible_entity_props = None,
        possible_relations = RELATIONS,
        possible_relation_props = None,
        kg_validation_schema = VALIDATION_SCHEMA,
        strict = True,
        num_workers = 4,
        max_triplets_per_chunk = 10
    )
    
    '''
    propertyGraph = PropertyGraphIndex(
        nodes =
        llm =
        kg_extractors = [kgExtractor],
        property_graph_store =
        vector_store =
        use_async = True,
        embed_model =
        embed_kg_nodes = True,
        callback_manager =
        transformations =
        storage_context =
        show_progress =
    )

    print(type(propertyGraph))

    return propertyGraph
    '''

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

async def addPathsToKG(pathList : list[Path], llmChoice : str) -> None:
    '''Takes in a list of paths to articles that need to be added into the KG. '''
    
    ''' Make a lock that every asynchronous process can share once we start executing. Can't have race conditions when doing DB ops. '''
    databaseLock : Lock = Lock()
    
    ''' Make our knowledge graph. '''
    async with databaseLock:
        try:
            knowledgeGraph = initializeKG(llmChoice)
        
        except KeyError as e:
            print(f"KeyError: {e} not set.")

    ''' Asynchronously execute the process of adding each path into the knowledge graph, making sure exceptions don't stop the entire execution. '''
    await gather(*(addPathToKG(path, databaseLock) for path in pathList), return_exceptions = True)

async def addPathToKG(documentPath: Path, lock : Lock, cqAnswerPath : Path = Path("cq_answers"), parsedPath : Path = Path("parsed_cq_answers")) -> None:
    ''' Takes in an individual path and adds the associated article to the KG. '''

    ''' Make sure passed document exists before anything. Code is definitely ugly, but it'll do for now. '''
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
    '''
    A parser to decide which LLM to use. Not particularly useful right now since I only allow Gemini for now, but should be helpful down the line.
    '''
    parser.add_argument('--llm',
                        type = str,
                        nargs = '?',
                        const = None,
                        default = "gemini",
                        action = "store",
                        metavar = "llm",
                        choices = ["gemini"],
                        help = "The LLM to use when adding articles to the KG.")

    arguments : Namespace = parser.parse_args()

    run(addPathsToKG(arguments.paths, arguments.llm))
