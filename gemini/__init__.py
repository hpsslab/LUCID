import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import configparser
import asyncio

# load parameters from config file
config = configparser.ConfigParser()
config.read('config.cfg')
temp = float(config['gemini']['Temperature'])
sysInstruction = config['gemini']["SystemInstruction"]

# initialize Gemini 1.5 Pro
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction=sysInstruction)

# send a request to the model and return the text response
def generate(input, mime=None, schema=None):
    res = model.generate_content(
        input,
        generation_config = genai.GenerationConfig(
            temperature=temp,
            response_mime_type=mime, 
            response_schema=schema
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
        stream = False
    )
    return res.text

# send a request to the model and return the text response (asynchronous version)
async def agenerate(input, mime=None, schema=None):
    res = await model.generate_content_async(
        input,
        generation_config = genai.GenerationConfig(
            temperature=temp,
            response_mime_type=mime,
            response_schema=schema
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
        stream = False
    )
    return res.text
