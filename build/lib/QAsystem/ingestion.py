# get the data -> split it into chunks -> turn them into embeddings ->store in vector store

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import boto3
import sys
import os
import json

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS

bedrock=boto3.client(service_name="bedrock-runtime",region_name="ap-south-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def data_ingestion():
    loader=PyPDFDirectoryLoader("./data")
    documents=loader.load()

    # make an object of RecursivecharacterTextSplitter class
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs=text_splitter.split_documents(documents)
    return docs
def get_vector_store(doc):
    vector_store_faiss=FAISS.from_documents(doc,bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss


if __name__=='__main__':
    docs=data_ingestion()
    get_vector_store(docs)