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

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA

from data_loaders import load_pdfs, load_docx_files, load_text_files



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



_template = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in English.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


_template = """You are an chat assistant for supporting usf students with their queries. 
If applicable use the context provided to better answer the question in the English. 
If the question cannot be the answered from the context provided, just say that you don't know. 
Context: {context} 
Question: {question} 
Answer:
"""
ANSWER_PROMPT  = ChatPromptTemplate.from_template(_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


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
