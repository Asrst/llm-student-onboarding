import os, sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from .data_loaders import load_pdfs, load_docx_files, load_text_files, load_json_file
import logging, time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY="56d7ec45-b638-411b-bda5-bec35a1ae85a"
PINECONE_INDEX_NAME="bull-buddy-index"


def load_and_embed(docs_path):

    # we create our vectorDB inside the ./data directory
    embedding = OpenAIEmbeddings()
    index_name = "bull-buddy-index"

    # configure client  
    pc = Pinecone(api_key=PINECONE_API_KEY)  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
    
    # check for and delete index if already exists  
    if index_name in pc.list_indexes().names():
        print("loading existing pincone index....")
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            print("Waiting...Index Not Ready...")
            time.sleep(1)
        # initialize the vector-db
        vectordb = PineconeVectorStore(index_name=index_name, embedding=embedding)    
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

        # create a new index  
        pc.create_index(  
            index_name,  
            dimension=1536,  # dimensionality of text-embedding-ada-002  
            metric='dotproduct',  
            spec=spec  
        )

        # wait for index to be initialized  
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            print("Waiting...Index Not Ready...")
            time.sleep(1)

        # vector store
        vectordb = PineconeVectorStore.from_documents(splits, embedding, 
                                                      index_name=index_name)

    # print index info
    index = pc.Index(index_name) 
    print(index.describe_index_stats())

    # create retriver based on vectordb
    # search_type="mmr"
    retriever = vectordb.as_retriever(search_kwargs={'k': 5})

    # return the retriever
    return retriever


# def load_and_embed(docs_path):

#     # we create our vectorDB inside the ./data directory
#     embedding = OpenAIEmbeddings()

#     db_path = f"{docs_path}/chromadb"
#     if os.path.exists(db_path):
#         print("loading index from disk: ", db_path)
#         vectordb = Chroma(persist_directory=db_path, 
#                           embedding_function=embedding)
    
#     else:
#         # load the documents
#         documents = []
#         word_docs = load_docx_files(f"{docs_path}/docx")
#         pdfs = load_pdfs(f"{docs_path}/pdfs")
#         json_docs = load_json_file(f"{docs_path}/json/jira-conversations-faqs.json",
#                                 jq_schema='.[].faq[]',
#                                 text_content=False)

#         documents = pdfs + word_docs + json_docs
#         print("total pages found: ", len(documents))
#         # print(documents[0])

#         # split the data into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size = 400,
#             chunk_overlap = 100
#         )

#         # store the splits to vector db
#         splits = text_splitter.split_documents(documents)
#         # vetcordb
#         vectordb = Chroma.from_documents(
#             documents=splits,
#             embedding=embedding,
#             persist_directory=db_path
#         )
#         # save to local storage
#         vectordb.persist()

#     # create retriver based on vectordb
#     retriever = vectordb.as_retriever(search_kwargs={'k': 5})

#     # return the retriever
#     return retriever