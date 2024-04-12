import os, sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from data_loaders import load_pdfs, load_docx_files, load_text_files, load_json_file
import logging 


def load_and_embed(docs_path):

    # we create our vectorDB inside the ./data directory
    embedding = OpenAIEmbeddings()

    db_path = f"{docs_path}/chromadb"
    if os.path.exists(db_path):
        print("loading index from disk: ", db_path)
        vectordb = Chroma(persist_directory=db_path, 
                          embedding_function=embedding)
    
    else:
        # load the documents
        documents = []
        word_docs = load_docx_files(f"{docs_path}/docx")
        pdfs = load_pdfs(f"{docs_path}/pdfs")
        # json_docs = load_json_file(f"{docs_path}/json/jira-conversations-faqs.json",
        #                         jq_schema='.[].faq[]',
        #                         text_content=False)

        documents = pdfs + word_docs # + json_docs
        print("total pages found: ", len(documents))
        # print(documents[0])

        # split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 400,
            chunk_overlap = 100
        )

        # store the splits to vector db
        splits = text_splitter.split_documents(documents)
        # vetcordb
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=db_path
        )
        # save to local storage
        vectordb.persist()

    # create retriver based on vectordb
    retriever = vectordb.as_retriever(search_kwargs={'k': 5})

    # return the retriever
    return retriever