# get the data -> split it into chunks -> turn them into embeddings ->store in vector store
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from QAsystem.ingestion import get_vector_store
import boto3

bedrock=boto3.client("bedrock-runtime")

#get prompt template
prompt_template="""
Use the following piece of context to provide a concise answer to the question at the end but use atleast summarize with 250 words with detailed explainations. if you dont know, dont try to makeup an answer.
<context>
{context}
</context>
Question:{question}
Assistant: 
"""

PROMPT=PromptTemplate(
    template=prompt_template, input_variables=["context","question"]
)

# get llm
def get_llama3_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock, model_kwargs={"maxTokens":512})
    return llm

# get retriever
def get_response_llm(LLm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        LLm=LLm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        ),
        return_surce_parameters=True,
        chain_type_kwargs={"prompt":PROMPT}
        
    )
    answer=qa({"query":query})
    return answer["result"]


if __name__=='__main__':
    query="what is RAG token?"
    vector_faiss=get_vector_store()
    llm=get_llama3_llm()
    get_response_llm(llm,vector_faiss,query)
