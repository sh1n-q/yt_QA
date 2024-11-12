import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import OpenAIEmbeddings
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

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
db = DeepLake(dataset_path='hub://dash/youtube_dyson', embedding=embeddings, read_only=True)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

# Summary generation
summary_prompt_template = """
Generate a personalized summary for the video transcript below. Focus on the major benefits and drawbacks, highlight key insights, and mention any unique features or common user concerns. Make sure to cover aspects like ease of use, value for money, and any unexpected pros or cons. The summary should help someone decide whether the product is worth their time and money without watching the entire video. Be concise and focus on actionable insights, avoiding generic details.

{context}

Personalized Summary:
"""
SUMMARY_PROMPT = PromptTemplate(template=summary_prompt_template, input_variables=["context"])
summary_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": SUMMARY_PROMPT})

# Answering specific questions
qa_prompt_template = """
Use the following pieces of transcripts from a video to answer the question in bullet points. 
Avoid generic details. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullet points:
"""
QA_PROMPT = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": QA_PROMPT})

def strip_markdown(text):
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    return text

# Function to generate personalized summary
def generate_summary():
    result = summary_chain.run("Generate a summary for the transcript.")
    return strip_markdown(result)

# Function to query detailed information
def query_fn(query):
    result = qa.run(query)
    return strip_markdown(result)

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# Influencer Video Query App")
    with gr.Row():
        with gr.Column():
            summary_btn = gr.Button("Generate Summary")
            summary_output = gr.Textbox(label="Summary Results")
            query = gr.Textbox(label="Query influencers videos:")
        with gr.Column():
            output = gr.Textbox(label="Query Results")
    
    # Button actions
    summary_btn.click(generate_summary, outputs=summary_output)
    query.submit(query_fn, query, output)

interface.launch()
