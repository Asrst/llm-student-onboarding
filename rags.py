import os, sys
from dotenv import load_dotenv
from operator import itemgetter
import logging 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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

from prompts import ANSWER_PROMPT, QUERY_AUG_PROMPT, HYDE_PROMPT, CONDENSE_QUESTION_PROMPT
from prompts import _combine_documents

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv('.env')

# Create a retrieval chain
# streaming=True, callbacks=[StreamingStdOutCallbackHandler()]

# intialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)



def base_rag(memory, retriever):

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

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
    rag_chain = loaded_memory | standalone_question | retrieved_documents | answer


    return rag_chain


def rag_with_hyde(memory, retriever):


    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    # Now we get a hypothethical document embedding
    hyde_doc = {
        "hyde_document": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | HYDE_PROMPT
        | llm
        | StrOutputParser(),
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("hyde_document") | retriever,
        "question": lambda x: x["hyde_document"],
    }

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
    rag_chain = loaded_memory | hyde_doc | retrieved_documents | answer

    # return the chain
    return rag_chain


def rag_with_query_aug(memory, retriever):
    
    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    # intilaize multi query retriever
    multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

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

    # return the chain
    return rag_chain