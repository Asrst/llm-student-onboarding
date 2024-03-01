import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import OpenAIEmbeddings


load_dotenv('.env')

# load the document as before
loader = PyPDFLoader('Satya_Resume_Data.pdf')
documents = loader.load()

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
result = qa_chain.invoke({'query': 'Who is the CV about?'})
print(result['result'])