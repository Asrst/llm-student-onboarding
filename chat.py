import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import OpenAIEmbeddings



def load_pdfs(path, chunksize=1000, overlap=100):
    """Recursively loads all the pdfs in a given directory path 
    and return a list containing pages of all documents. 

    Args:
        chunksize (int, optional): Defaults to 1000.
        overlap (int, optional): Defaults to 100.

    Returns:
        type: List of Langchain Docment Class
    """    

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import PyPDFLoader


    # load all pdfs in the directory
    loader = DirectoryLoader(
        path, 
        use_multithreading=True,
        loader_cls=PyPDFLoader,
        show_progress=True,
        recursive=True
    )

    # returns a list of pages as Document types
    pdf_docs = loader.load() 
    return pdf_docs



load_dotenv('.env')

# load the document as before
documents = load_pdfs(r"data/pdfs")
print("total pages found: ", len(documents))
# print(documents[0])


# we split the data into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(documents)

# we create our vectorDB inside the ./data directory
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory='./data'
)

vectordb.persist()

# Create the RetrievalQA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True
)

# we can now exectute queries againse our Q&A chain
# question = "What is MS BAIS program and Explain its circulum"

question = input("What do u want to know about MS BAIS or AIBA program?")
result = qa_chain.invoke({'query': f'{question}'})
print(result['result'])