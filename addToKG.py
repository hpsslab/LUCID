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
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import PropertyGraphIndex, Settings, StorageContext, Document, load_index_from_storage
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.google_genai import GoogleGenAI

def initializeKG(llmChoice : str, embedChoice : str) -> any:
    ''' Create LLM depending on argument. '''
    llm : Any = None
    
    if llmChoice == "gemini":
        config : ConfigParser = ConfigParser()
        
        config.read('./gemini/config.cfg')
        
        '''
        Use the LlamaIndex LLM constructors to create our LLM to construct a KG with. 
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
            debug_config: options that make it easier to debug (ie: use cached content so that you don't get charged for every LLM call).
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
    Create embedding model based on arguments. 
    NOTE: embedding model should be a SentenceTransformer when filtering on HuggingFace so the model gets along with LlamaIndex.
    '''
    embeddingModel : Any = None
    tokenizer : Any = None
    queryInstruction : str = "Represent the query for KG retrieval: " 
    cqInstruction : str = "Represent the document for KG construction: "
    embeddingBatchSize : int = 8
    
    if embedChoice == "bert":
        ''' 
        Initialize all relevant paths into variables. 
        Construct model/tokenizer from uncased version (ie: "science" is the same as "Science") and base version (ie: not the larger model). 
        '''
        bertTokenizerPath : str = "./bert_tokenizer"
        bertModelPath : str = "./bert_model"
        bertHuggingFacePath : str = "inokufu/bert-base-uncased-xnli-sts-finetuned-education"

        if not Path(bertTokenizerPath).exists():
            print(f"Loading BERT tokenizer and saving to {bertTokenizerPath}.")
        
        tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast = AutoTokenizer.from_pretrained(bertHuggingFacePath, cache_dir = bertTokenizerPath)

        if not Path(bertModelPath).exists():
            print(f"Loading BERT model and saving to {bertModelPath}.")
        
        '''
        Construct the embedding model that we will use to extract relevant feature information from text samples (ie: query or CQ answers).
        Parameters:
            model_name: the name of the model to use from HuggingFace.
            max_length: the maximum length input that should be embedded. Sticking with default value here.
            query_instruction: the prompt to apply to the query before embedding it.
            text_instruction: the prompt to apply to CQ answers before embedding them.
            normalize: whether or not to normalize the embedding vectors. Set to true since magnitude of vectors won't matter in cosine similarity.
            embed_batch_size: the size of the batch to use when calculating embeddings.
            cache_folder: a folder to cache the retrieved model files to so we don't need to reload multiple times from HuggingFace.
            trust_remote_code: whether or not to trust public HuggingFace code to run.
            device: the devices to use for the computation (ie: "cpu", "cuda", "npu", etc.). Setting to None so it can dynamically figure this out.
            callback_manager: used if we want to manage the asynchronous flow of the program within LlamaIndex.
            parallel_process: whether or not to execute the embedding process in parallel.
            target_devices: the actual devices to use. None means it uses device parameter to actually determine which cores to target.
            show_progress_bar: whether or not to show a progress bar upon making the model.
        '''
        embeddingModel : HuggingFaceEmbedding = HuggingFaceEmbedding(
            model_name = bertHuggingFacePath,
            max_length = None,
            query_instruction = queryInstruction,
            text_instruction = cqInstruction,
            normalize = True,
            embed_batch_size = embeddingBatchSize,
            cache_folder = bertModelPath,
            trust_remote_code = False,
            device = None,
            callback_manager = None,
            parallel_process = True, 
            target_devices = None,
            show_progress_bar = False
        )

    elif embedChoice == "scibert":
        '''
        Initialize all relevant paths into variables.
        Construct model/tokenizer from uncased version (ie: "science" is the same as "Science").
        '''
        scibertTokenizerPath : str = "./scibert_tokenizer"
        scibertModelPath : str = "./scibert_model"
        scibertHuggingFacePath : str = "jordyvl/scibert_scivocab_uncased_sentence_transformer"

        if not Path(scibertTokenizerPath).exists():
            print(f"Loading SciBERT tokenizer and saving to {scibertTokenizerPath}.")

        tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast = AutoTokenizer.from_pretrained(scibertHuggingFacePath, cache_dir = scibertTokenizerPath)

        if not Path(scibertModelPath).exists():
            print(f"Loading SciBERT model and saving to {scibertModelPath}.")

        '''
        Construct the embedding model that we will use to extract relevant feature information from text samples (ie: query or CQ answers).
        Parameters:
            model_name: the name of the model to use from HuggingFace.
            max_length: the maximum length input that should be embedded. Sticking with default value here.
            query_instruction: the prompt to apply to the query before embedding it.
            text_instruction: the prompt to apply to CQ answers before embedding them.
            normalize: whether or not to normalize the embedding vectors. Set to true since magnitude of vectors won't matter in cosine similarity.
            embed_batch_size: the size of the batch to use when calculating embeddings.
            cache_folder: a folder to cache the retrieved model files to so we don't need to reload multiple times from HuggingFace.
            trust_remote_code: whether or not to trust public HuggingFace code to run.
            device: the devices to use for the computation (ie: "cpu", "cuda", "npu", etc.). Setting to None so it can dynamically figure this out.
            callback_manager: used if we want to manage the asynchronous flow of the program within LlamaIndex.
            parallel_process: whether or not to execute the embedding process in parallel.
            target_devices: the actual devices to use. None means it uses device parameter to actually determine which cores to target.
            show_progress_bar: whether or not to show a progress bar upon making the model.
        '''
        embeddingModel : HuggingFaceEmbedding = HuggingFaceEmbedding(
            model_name = scibertHuggingFacePath,
            max_length = None,
            query_instruction = queryInstruction,
            text_instruction = cqInstruction,
            normalize = True,
            embed_batch_size = embeddingBatchSize,
            cache_folder = scibertModelPath,
            trust_remote_code = False,
            device = None,
            callback_manager = None,
            parallel_process = True,
            target_devices = None,
            show_progress_bar = False
        )
        
    else:
        raise ShouldNotOccurError()

    '''
    How we will split up CQ answers into parsable chunks (aka nodes) to process chunks into KG triplets for the future KG.
    Parameters:
        chunk_size: how many tokens should be stored in each chunk that is processed?
        Not making too large since most CQ answers are only a sentence or two long.
        chunk_overlap: how many tokens should overlap from chunk to chunk?
        tokenizer: the tokenizer to use to tokenize the CQ answers.
        Tokenizer should match the embedding model.
        callback_manager: used if we want to manage the asynchronous flow of the program within LlamaIndex.
        separator: the primary character that divides words from one another.
        backup_separators: a list of other characters that could be used to divide words from one another.
        keep_whitespaces: whether or not to keep whitespace when chunking the CQ answers.
        include_metadata: whether or not to consider metadata when chunking CQ answers.
        include_prev_next_rel: whether or not to include how one chunk is related to the previous/next chunks.
        id_func: a function that generates chunk ids from a static variable.
    '''
    transformation : TokenTextSplitter = TokenTextSplitter(
        chunk_size = 128,
        chunk_overlap = 16,
        tokenizer = tokenizer,
        callback_manager = None,
        separator = " ",
        backup_separators = ["\n"],
        keep_whitespaces = False,
        include_metadata = False,
        include_prev_next_rel = True,
        id_func = None
    )
    
    ''' Set the LlamaIndex embedding model, LLM, tokenizer, and CQ parser to the ones we just created. '''
    Settings.embed_model = embeddingModel
    Settings.llm = llm
    Settings.tokenizer = tokenizer
    Settings.node_parser = transformation

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
        transformations: how to divide the source content (ie: the CQ answers) into pieces for the LLM to process.
        I chose the TokenTextSplitter since we already logically split up ideas using the CQ approach and defined a tokenizer earlier.
        storage_context: the context in which the KG will be stored (used so recreation is easier upon multiple runs).
        show_progress: whether or not to show a progress bar when adding to the KG.
    '''

    kgStoragePath : str = "./kg_storage"

    if not Path(kgStoragePath).exists():
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
            show_progress = False
        )
        
        print(f"Storing KG information in {kgStoragePath}.")
        propertyGraph.storage_context.persist(kgStoragePath)
    
    else:
        ''' We can just load everything in if we have already run the program before. '''
        print(f"Loading KG from {kgStoragePath}.")
        storage_context : StorageContext = StorageContext.from_defaults(persist_dir = kgStoragePath)
        propertyGraph : PropertyGraphIndex = load_index_from_storage(storage_context)
    
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
    async with databaseLock:
        try:
            knowledgeGraph : PropertyGraphIndex = initializeKG(llmChoice, embedChoice)
        
        except KeyError as e:
            print(f"KeyError: {e} not set.")
    print("Finished KG construction.")
    
    ''' Asynchronously execute the process of adding each path into the knowledge graph, making sure exceptions don't stop the entire execution. '''
    await gather(*(addPathToKG(path, databaseLock, knowledgeGraph) for path in pathList), return_exceptions = True)

async def addPathToKG(documentPath: Path, lock : Lock, propertyGraph : PropertyGraphIndex, databasePath : Path = Path("databases"), cqAnswerPath : Path = Path("cq_answers"), parsedPath : Path = Path("parsed_cq_answers")) -> None:
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
                '''
                print(f"Answering CQs for document {documentPath.stem}.")
                await answerCQs(documentPath, currentCQAnswerPath)
                print(f"CQs have been answered for document {documentPath.stem}.")
                '''

                with open(currentCQAnswerPath, "r") as cqFile:
                    cqAnswers : str = cqFile.read()
                
                print(f"Adding document {documentPath.stem} to the KG.")
                
                cqAnswerDocument : Document = Document(
                    text = cqAnswers,
                    metadata={"filename": documentPath.stem, "doc_id": hashDocument(documentPath)}
                )
                print(f"Document for {documentPath.stem} made.")
                propertyGraph.insert(cqAnswerDocument)

                print(f"Added document {documentPath.stem} to the KG.")
                
                ''' Make sure to save our changes. '''
                propertyGraph.storage_context.persist("./kgStorage")
            
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
            await gather(*(addPathToKG(root.joinpath(file), lock, propertyGraph) for file in files), return_exceptions = True)
    
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
                        default = "bert",
                        action = "store",
                        metavar = "embed",
                        choices = ["bert", "scibert"],
                        help = "The LLM to use when embedding text (for the KG or for the query).")

    arguments : Namespace = parser.parse_args()

    run(addPathsToKG(arguments.paths, arguments.llm, arguments.embed))
