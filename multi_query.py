import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from langchain.output_parsers import PydanticToolsParser
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA

from data_loaders import load_pdfs, load_docx_files, load_text_files
import logging 

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


load_dotenv('.env')

# load the documents
documents = []
word_docs = load_docx_files(r"data/docx")
pdfs = load_pdfs(r"data/pdfs")
documents = pdfs + word_docs

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
    persist_directory='./data/chromadb'
)

vectordb.persist()
# create retriver based on vectordb
retriever = vectordb.as_retriever(search_kwargs={'k': 5})

# Create a retrieval chain
# streaming=True, callbacks=[StreamingStdOutCallbackHandler()]
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000,
                                                  return_messages=True, 
                                                  output_key="answer", 
                                                  input_key="question")



system = """You are an expert at assisting students regarding University of South Florida's Masters in Business Analytics and 
Information System program. Answer the user question as best you can, addressing the user's concerns.\

Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Return at least 3 versions of the question."""

QUERY_AUG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# answer prompt

_template = """You are an chat assistant for supporting usf students with their queries. 
If applicable use the context provided to better answer the question in the English. 
If the question cannot be the answered from the context provided, just say that you don't know. 
Context: {context} 
Question: {question} 
Answer:
"""
ANSWER_PROMPT  = ChatPromptTemplate.from_template(_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, sep="\n\n"
                       ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return sep.join(doc_strings)


# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)


multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, 
                                               llm=llm)


# Now we retrieve the documents
multi_retrieval_docs = {"question": RunnablePassthrough(), 
                       "docs": multi_retriever}


# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | llm,
    "docs": itemgetter("docs"),
}

# And now we put it all together!
rag_chain = loaded_memory | multi_retrieval_docs | answer

while True:
    question_input = input("\nUser: ")
    if question_input == "exit":
        break
    inputs = {"question": question_input}
    result = rag_chain.invoke(inputs)
    answer = result["answer"].content
    print(f"Agent: {answer}")
    memory.save_context(inputs, {"answer": answer})
    # print("\n")
    # print(memory.load_memory_variables({}))