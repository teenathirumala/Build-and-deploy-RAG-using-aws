# get the data -> split it into chunks -> turn them into embeddings ->store in vector store
import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from QAsystem.ingestion import data_ingestion
from QAsystem.ingestion import get_vector_store
from QAsystem.retrievalandGeneration import get_llama3_llm,get_response_llm
bedrock=boto3.client("bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)
def main():
    st.set_page_config(" QA with doc")
    st.header("QA with doc using langchain and AWSbedrock")
    user_question=st.text_input("ask any question from the pdf")
    with st.sidebar:
        st.title("update or create vectorstore")
        if st.button("vector update"):
            with st.spinner("processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("done")
        if st.button("llama model"):
            with st.spinner("processing..."):
                faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
                llm=get_llama3_llm()

                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("done")
if __name__=='__main__':
    main()
          