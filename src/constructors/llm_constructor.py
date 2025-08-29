from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.base.llms.base import BaseLLM
from os import environ
from configparser import ConfigParser
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from exceptions.snoError import ShouldNotOccurError

''' CONSTANTS '''
GEMINI_MODEL = "gemini-2.0-flash"

def constructGeminiLLM() -> GoogleGenAI:
    config : ConfigParser = ConfigParser()

    config.read('./src/llms/gemini/config.cfg')

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
        The goal here is to provide CQ answers to the model and have the LLM create triplets for a knowledge graph (once per set of CQ answers),            so caching won't be necessary for this particular model.
    '''
    return GoogleGenAI(
        model = GEMINI_MODEL,
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

def constructLLM(llmChoice : str) -> BaseLLM: 
    if llmChoice == "gemini":
        return constructGeminiLLM()
    else:
        raise ShouldNotOccurError()

def constructLangchainLLM(llmChoice : str) -> BaseChatModel:
    if llmChoice == "gemini":
        return ChatGoogleGenerativeAI(model = GEMINI_MODEL, google_api_key = environ["GEMINI_API_KEY"])
    else:
        raise ShouldNotOccurError()
