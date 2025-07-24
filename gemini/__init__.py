from google.generativeai import configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from os import environ
from pathlib import Path
from configparser import ConfigParser
import asyncio

''' Load config file. '''
config : ConfigParser = ConfigParser()

''' Read in the configuration of the model. '''
try:
    config.read('./gemini/config.cfg')
except FileNotFoundError as e:
    print(f"{e}.")

temp : float = float(config['gemini']['Temperature'])
sysInstruction : str = config['gemini']["SystemInstruction"]

''' Read in Gemini Key from our environment. '''
try:
    configure(api_key = environ["GEMINI_API_KEY"])
except KeyError as e:
    print(f"KeyError: {e} not set.")

model : GenerativeModel = GenerativeModel(model_name="gemini-2.0-flash", system_instruction=sysInstruction)

# send a request to the model and return the text response
def generate(input : str, answerPath : Path, mime = None, schema = None) -> None:
    res : GenerateContentResponse = model.generate_content(
        input,
        generation_config = GenerationConfig(
            temperature=temp,
            response_mime_type=mime, 
            response_schema=schema
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
        stream = False
    )
    
    with open(answerPath, 'w') as outputFile:
        outputFile.write(res.text)

    return

# send a request to the model and return the text response (asynchronous version)
async def agenerate(input : str, answerPath : Path, mime=None, schema=None):
    res : AsyncGenerateContentResponse = await model.generate_content_async(
        input,
        generation_config = GenerationConfig(
            temperature=temp,
            response_mime_type=mime,
            response_schema=schema
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
        stream = False
    )

    with open(answerPath, 'w') as outputFile:
        outputFile.write(res.text)

    return
