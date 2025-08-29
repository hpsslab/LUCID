from asyncio import run, gather, Lock
from pathlib import Path
from schema import ENTITIES, RELATIONS, VALIDATION_SCHEMA
from qa import answerCQs
from xxhash import xxh128
from constructors.llm_constructor import constructLLM
from constructors.embedding_model_constructor import constructEmbeddingModel
from exceptions.extensionError import ExtensionError
from exceptions.snoError import ShouldNotOccurError
from exceptions.duplicateError import DuplicateError
from argparse import ArgumentParser, Namespace
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import PropertyGraphIndex, Settings, StorageContext, Document, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

''' CONSTANTS '''
KG_STORAGE_PATH : str = "./kg_storage"
ACCEPTABLE_DOCUMENT_EXTENSIONS : set([str]) = {".pdf"}

def initializeKG(llmChoice : str, embedChoice : str) -> any:
    ''' Create LLM depending on argument. '''
    llm : BaseLLM = constructLLM(llmChoice) 

    ''' Create embedding model based on argument. '''
    embeddingModel : BaseEmbedding = constructEmbeddingModel(embedChoice)

    '''
    How we will split up CQ answers into parsable chunks (aka nodes) to process chunks into KG triplets for the future KG.
    Parameters:
        buffer_size: the number of sentences to group together when evaluating semantic similarity. Consider each sentence individually for now.
        embed_model: the embedding model to use.
        sentence_splitter: how to split text into sentences. Uses default method provided by library.
        include_metadata: whether to include the metadata of the text in nodes.
        include_prev_next_rel: whether to include prev/next relationships between nodes.
        breakpoint_percentile_threshold: the percent dissimilarity required between sentences to form different nodes.
    '''
    transformation : SemanticSplitterNodeParser = SemanticSplitterNodeParser(
        buffer_size = 1,
        embed_model = embeddingModel,
        include_metadata = False,
        include_prev_next_rel = True,
        breakpoint_percentile_threshold = 95
    )
    
    ''' Set the LlamaIndex embedding model, LLM, and CQ parser to the ones we just created. '''
    Settings.embed_model = embeddingModel
    Settings.llm = llm
    Settings.node_parser = transformation

    '''
    Use the schema that we defined to extract our necessary information to construct the KG.
    Ask the LLM to extract KG triples from the CQ answers.
    Parameters:
        llm: the LLM to prompt to extract the entities and relations from the CQ answers.
        extract_prompt: the prompt to feed the LLM with to extract everything. Using default for now.
        I think the default prompt that the library uses will work fine, but we can add more context to the prompt later if necessary.
        possible_entities & possible_relations: the entities/relations in our schema.
        possible_entity_props & possible_relation_props: a list of properties that we would want to extract about a given entity/relation.
        Leaving as None for now unless we want specific information later (would probably need to alter our approach too).
        kg_schema_cls: a class that represents our schema. Letting LlamaIndex create the class for now but might look into this later.
        kg_validation_schema: rules defining which entities can pair with which relations.
        strict: must we stick to this schema or can the LLM define new attributes?
        NOTE: this current version does not stick to the schema very well (TODO: fix later), so strict = true will filter out a lot of triplets.
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
        kg_validation_schema = {"relationships": VALIDATION_SCHEMA},
        strict = False,
        num_workers = 4,
        max_triplets_per_chunk = 10
    )

    '''
    Now, we can create the knowledge graph object that the LLM agent will eventually query from.
    NOTE: check for existence (and load from memory instead) before we create a new object.
    Parameters:
        nodes: a list of nodes to add to the KG from the beginning. We want to add as we answer CQs, so we set to empty list.
        llm: the LLM that the KG will utilize to extract triplets that are relevant to the query.
        kg_extractors: the method in which triplets will be acquired (set to the Schema based extractor defined above).
        property_graph_store: the LlamaIndex object storing our knowledge graph.
        vector_store: the LlamaIndex object storing our vector embeddings of the KG nodes.
        use_async: whether or not to handle these tasks asynchronously.
        embed_model: the model that the KG will use to embed nodes and queries. 
        embed_kg_nodes: whether or not to embed KG nodes.
        callback_manager: used if we want to manage the asynchronous flow of the program within LlamaIndex.
        transformations: how to divide the source content (ie: the CQ answers) into pieces for the LLM to process aka the earlier SemanticSplitter.
        storage_context: the context in which the KG will be stored (used so recreation is easier upon multiple runs).
        show_progress: whether or not to show a progress bar when adding to the KG.
    '''

    if not Path(KG_STORAGE_PATH).exists():
        ''' 
        Set vector and graph databases to default memory based implementations for now and create storage context for easy reloads later.
        TODO: add credentials for a different type of database to improve functionality?
        '''
        property_graph_store : SimplePropertyGraphStore = SimplePropertyGraphStore()
        vector_store : SimpleVectorStore = SimpleVectorStore()

        storage_context : StorageContext = StorageContext.from_defaults(
            property_graph_store = property_graph_store,
            vector_store = vector_store
        )
        
        propertyGraph : PropertyGraphIndex = PropertyGraphIndex(
            nodes = [],
            llm = llm,
            kg_extractors = [kgExtractor],
            property_graph_store = property_graph_store,
            vector_store = vector_store,
            use_async = True,
            embed_model = embeddingModel,
            embed_kg_nodes = True,
            callback_manager = None,
            transformations = [transformation],
            storage_context = storage_context,
            show_progress = True
        )
    
    else:
        ''' We can just load everything in if we have already run the program before. '''
        print(f"Loading KG from {KG_STORAGE_PATH}.")
        storage_context : StorageContext = StorageContext.from_defaults(persist_dir = KG_STORAGE_PATH)
        propertyGraph : PropertyGraphIndex = load_index_from_storage(storage_context,
            llm = llm,
            kg_extractors = [kgExtractor],
            use_async = True,
            embed_model = embeddingModel,
            embed_kg_nodes = True,
            callback_manager = None,
            transformations = [transformation],
            show_progress = True
        )

    return propertyGraph

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

def isInKG(documentPath : Path, dbPath : Path) -> bool:
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

async def addPathsToKG(pathList : list[Path], llmChoice : str, embedChoice : str) -> None:
    '''Takes in a list of paths to articles that need to be added into the KG. '''
    
    ''' Make a lock that every asynchronous process can share once we start executing. Can't have race conditions when doing DB ops. '''
    databaseLock : Lock = Lock()
    
    print("Constructing KG.")
    
    ''' Make our knowledge graph. '''
    try:
        knowledgeGraph : PropertyGraphIndex = initializeKG(llmChoice, embedChoice)
        
    except KeyError as e:
        print(f"KeyError: {e} not set.")

    except ShouldNotOccurError as e:
        print(f"SNO Error: {e}.")

    print("Finished KG construction.")
    
    ''' Asynchronously execute the process of adding each path into the knowledge graph, making sure exceptions don't stop the entire execution. '''
    await gather(*(addPathToKG(path, databaseLock, knowledgeGraph, llmChoice) for path in pathList), return_exceptions = True)

async def addPathToKG(documentPath: Path, lock : Lock, propertyGraph : PropertyGraphIndex, llmChoice : str, databasePath : Path = Path("databases"), cqAnswerPath : Path = Path("cq_answers"), parsedPath : Path = Path("parsed_cq_answers")) -> None:
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
            if documentPath.suffix not in ACCEPTABLE_DOCUMENT_EXTENSIONS:
                raise ExtensionError(documentPath.suffix, "knowledge graph")
           
            ''' 
            Make sure file isn't already in KG. Add to our DB that it is if we are adding to KG for the first time. 
            Make sure folder where CQ answers will be put exists too.
            '''
            async with lock:
                if not cqAnswerPath.exists():
                    cqAnswerPath.mkdir()

                if not databasePath.exists():
                    databasePath.mkdir()
                
                inKG : bool = isInKG(documentPath, Path(databasePath.joinpath("kgContents.txt")))

            if inKG:
                raise DuplicateError(documentPath.name)

            else:
                ''' This is the path where the LLM will store its answers to the CQs. '''
                currentCQAnswerPath : Path = Path(cqAnswerPath.joinpath(f"{documentPath.stem}_answers.txt"))
                
                ''' Answer CQs and pass those answers into the method that creates triples and inserts them into the KG. '''
                 
                print(f"Answering CQs for document {documentPath.stem}.")
                await answerCQs(documentPath, currentCQAnswerPath, llmChoice)
                print(f"CQs have been answered for document {documentPath.stem}.")

                with open(currentCQAnswerPath, "r") as cqFile:
                    cqAnswers : str = cqFile.read()
                
                print(f"Adding document {documentPath} to the KG.")

                cqAnswerDocument : Document = Document(
                    text = cqAnswers,
                    metadata={"filename": documentPath.stem, "doc_id": hashDocument(documentPath)}
                )

                await propertyGraph.ainsert(cqAnswerDocument)

                print(f"Added document {documentPath} to the KG.")
                
                ''' Make sure to save our changes. '''
                propertyGraph.storage_context.persist(KG_STORAGE_PATH)
            
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
            await gather(*(addPathToKG(root.joinpath(file), lock, propertyGraph, llmChoice) for file in files), return_exceptions = True)
    
    else:
        raise ShouldNotOccurError() 

if __name__ == "__main__":
    '''
    If we want to add to the KG directly from this file, we can do that. 
    The below parser has the same logic as the subparser in main, but relocated here for convenience.
    '''

    parser : ArgumentParser = ArgumentParser(prog = "addToKG.py",
                                             usage = "python3 addToKG.py [options] {filepath|folderpath}+",
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

    ''' A parser to decide which embedding model to use. '''
    parser.add_argument('--embed',
                        type = str,
                        nargs = '?',
                        const = None,
                        default = "gemini",
                        action = "store",
                        metavar = "embed",
                        choices = ["bert", "scibert", "gemini"],
                        help = "The LLM to use when embedding text (for the KG or for the query).")

    arguments : Namespace = parser.parse_args()

    run(addPathsToKG(arguments.paths, arguments.llm, arguments.embed))
