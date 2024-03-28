import os, sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from data_loaders import load_pdfs, load_docx_files, load_text_files
import logging 


def load_and_embed(docs_path):

    # load the documents
    documents = []
    word_docs = load_docx_files(f"{docs_path}/docx")
    pdfs = load_pdfs(f"{docs_path}/pdfs")
    documents = pdfs + word_docs
    print("total pages found: ", len(documents))
    # print(documents[0])

    # split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 150
    )

    # store the splits to vector db
    splits = text_splitter.split_documents(documents)

    # we create our vectorDB inside the ./data directory
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory='./data/chromadb'
    )

    # save to local storage
    vectordb.persist()

    # create retriver based on vectordb
    retriever = vectordb.as_retriever(search_kwargs={'k': 5})

    # return the retriever
    return retriever


def load_data():


    return
