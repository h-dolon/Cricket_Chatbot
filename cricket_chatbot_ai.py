import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from getpass import getpass
from docx import Document

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text_content = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(text_content)

docx_path = "Cricket.docx"
text_content = extract_text_from_docx(docx_path)

# print(text_content)

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN']= HUGGINGFACEHUB_API_TOKEN

model_id = "mistralai/Mistral-7B-v0.1"
conv_model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.6, "max_new_tokens":150})

template = """You are a Cricket game teacher and AI assistant that gives suggestions based on the query received as input

{query}

"""

@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=['query', 'text_content'])
    conv_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True)

    cl.user_session.set("llm_chain", conv_chain)
    cl.user_session.set("text_content", text_content)

@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    text_content = cl.user_session.get("text_content")

    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler(text_content)])

    await cl.Message(content=res["text"]).send()
