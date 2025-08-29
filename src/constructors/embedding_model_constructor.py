from transformers import AutoTokenizer
from exceptions.snoError import ShouldNotOccurError
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from os import environ

''' CONSTANTS '''
QUERY_INSTRUCTION : str = "Represent the query for KG retrieval: "
CQ_INSTRUCTION : str = "Represent the document for KG construction: "
EMBEDDING_BATCH_SIZE : int = 8
BERT_TOKENIZER_PATH : str = "./bert_tokenizer"
BERT_MODEL_PATH : str = "./bert_model"
BERT_HF_PATH : str = "inokufu/bert-base-uncased-xnli-sts-finetuned-education"
SCIBERT_TOKENIZER_PATH : str = "./scibert_tokenizer"
SCIBERT_MODEL_PATH : str = "./scibert_model"
SCIBERT_HF_PATH : str = "jordyvl/scibert_scivocab_uncased_sentence_transformer"

def constructBERTModel() -> HuggingFaceEmbedding:    
    ''' 
    Initialize all relevant paths into variables. 
    Construct model/tokenizer from uncased version (ie: "science" is the same as "Science") and base version (ie: not the larger model). 
    '''
    if not Path(BERT_TOKENIZER_PATH).exists():
        print(f"Loading BERT tokenizer and saving to {BERT_TOKENIZER_PATH}.")
        
    tokenizer : BertTokenizerFast = AutoTokenizer.from_pretrained(BERT_HF_PATH, cache_dir = BERT_TOKENIZER_PATH)

    Settings.tokenizer = tokenizer

    if not Path(BERT_MODEL_PATH).exists():
        print(f"Loading BERT model and saving to {BERT_MODEL_PATH}.")
        
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
    return HuggingFaceEmbedding(
        model_name = BERT_HF_PATH,
        max_length = None,
        query_instruction = QUERY_INSTRUCTION,
        text_instruction = CQ_INSTRUCTION,
        normalize = True,
        embed_batch_size = EMBEDDING_BATCH_SIZE,
        cache_folder = BERT_MODEL_PATH,
        trust_remote_code = False,
        device = None,
        callback_manager = None,
        parallel_process = True, 
        target_devices = None,
        show_progress_bar = True
    )

def constructSciBERTModel() -> HuggingFaceEmbedding:
    '''
    Initialize all relevant paths into variables.
    Construct model/tokenizer from uncased version (ie: "science" is the same as "Science").
    '''
    if not Path(SCIBERT_TOKENIZER_PATH).exists():
        print(f"Loading SciBERT tokenizer and saving to {SCIBERT_TOKENIZER_PATH}.")

    tokenizer : BertTokenizerFast = AutoTokenizer.from_pretrained(SCIBERT_HF_PATH, cache_dir = SCIBERT_TOKENIZER_PATH)

    Settings.tokenizer = tokenizer

    if not Path(SCIBERT_MODEL_PATH).exists():
        print(f"Loading SciBERT model and saving to {SCIBERT_MODEL_PATH}.")

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
    return HuggingFaceEmbedding(
        model_name = SCIBERT_HF_PATH,
        max_length = None,
        query_instruction = QUERY_INSTRUCTION,
        text_instruction = CQ_INSTRUCTION,
        normalize = True,
        embed_batch_size = EMBEDDING_BATCH_SIZE,
        cache_folder = SCIBERT_MODEL_PATH,
        trust_remote_code = False,
        device = None,
        callback_manager = None,
        parallel_process = True,
        target_devices = None,
        show_progress_bar = False
    )

def constructGeminiModel() -> GoogleGenAIEmbedding:
    '''
    Construct the embedding model that we will use to extract relevant feature information from text samples (ie: query or CQ answers).
    Parameters:
        model_name: which Gemini embedding model should we use?
        api_key: credentials to use the model.
        embedding_config: parameters to give to the embedding model to improve performance (ie: task, dimensionality, etc.).
        vertexai_config: credentials to use if we want to use via VertexAI (provided a dictionary containing a project id and a location).
        http_options: HTTP options for the LLM requests.
        See https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L1247 for more.
        debug_config: options that make it easier to debug (ie: use cached content so that you don't get charged for every LLM call).
        See https://github.com/googleapis/python-genai/blob/main/google/genai/client.py#L89 for more.
        embed_batch_size (int): Batch size for embedding. Defaults to 100.
        callback_manager: used if we want to manage the asynchronous flow of the program within LlamaIndex.
        retries: how many times should the API retry if an API error takes place?
        timeout: how long should it be (in seconds) before API calls timeout?
        retry_min_seconds: minimum wait time between API call retries.
        retry_max_seconds: maximum wait time between API call retries.
        retry_exponential_base: exponential base to multiply retry wait time by before retrying.
    '''
    return GoogleGenAIEmbedding(
        model_name = "text-embedding-004",
        api_key = environ["GEMINI_API_KEY"],
        embedding_config = None,
        vertexai_config = None,
        http_options = None,
        debug_config = None,
        embed_batch_size = EMBEDDING_BATCH_SIZE,
        callback_manager = None,
        retries = 3,
        timeout = 10,
        retry_min_seconds = 1,
        retry_max_seconds = 10,
        retry_exponential_base = 2
    )
        
def constructEmbeddingModel(embeddingChoice : str) -> BaseEmbedding:
    if embeddingChoice == "bert":
        return constructBERTModel()
    elif embeddingChoice == "scibert":
        return constructSciBERTModel()
    elif embeddingChoice == "gemini":
        return constructGeminiModel()
    else:
        raise ShouldNotOccurError()
