import os, sys

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document


# basic template for page content
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(docs, sep="\n\n"):
    doc_strings = [format_document(doc, DEFAULT_DOCUMENT_PROMPT) for doc in docs]
    return sep.join(doc_strings)


# final answer/ouptut prompt
_template = """You are a friendly chatbot named Bull Buddy.
You are here to help USF students with their admissions and onboarding.
If applicable use the context provided to better answer the question in the English. 
If the question cannot be the answered from the context provided, just say that you don't know. 

Context: {context} 
Question: {question} 
Answer:
"""
ANSWER_PROMPT  = ChatPromptTemplate.from_template(_template)


# standard alone
_template = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in English.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)



# query augumentation prmompt
system = """You are an expert at assisting students regarding University of South Florida's Masters in Business Analytics and 
Information System program.\

Perform Query Expansion/Augumentation. If there are multiple common ways of phrasing a user question \
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

# HYDE Prompt
system = """You are an expert at assisting students regarding University of South Florida's Masters in Business Analytics and 
Information System program.\

Answer the user question as best you can, addressing the user's concerns.
"""

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

