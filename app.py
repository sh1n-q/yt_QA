import gradio as gr
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
import re

# Environment variables for API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")
if ACTIVELOOP_TOKEN is None:
    raise ValueError("ACTIVELOOP_TOKEN is not set in the environment variables")


llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
db = DeepLake(dataset_path='hub://dash/youtube_dyson', embedding=embeddings, read_only=True)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

# LLM prompt template and chain
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points. 
Avoid generic details. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullet points:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

def strip_markdown(text):
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    return text

def query_fn(query):
    result = qa.run(query)
    return strip_markdown(result)

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# Influencer Video Query App")
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Query influencers videos:")
        with gr.Column():
            output = gr.Textbox(label="Query Results")
    query.submit(query_fn, query, output)


interface.launch()
