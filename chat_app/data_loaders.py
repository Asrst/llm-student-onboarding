import os, sys
import json
import pandas as pd


def load_pdfs(path):
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


def load_text_files(path):
    """ 
    Load all text files from a given directory

    Args:
        path (str): This is the subfolder path - be sure that this is relative to where you start the program
        chunksize (int, optional): Size of text chunks. Defaults to 1000.
        overlap (int, optional): Overlap between text chunks. Defaults to 100.

    Returns:
        list(Document): Returns a list of LangChain Document types
    """

    text_loader_kwargs = {'autodetect_encoding': True}

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader

    loader =  DirectoryLoader(path,  glob="**/*.txt", 
                                loader_cls=TextLoader, 
                                loader_kwargs=text_loader_kwargs)
    txt_docs = loader.load()
    
    # return a list of text
    return txt_docs 




def load_docx_files(path):
    """Recursively loads all the pdfs in a given directory path 
    and return a list containing pages of all documents. 

    Args:
        chunksize (int, optional): Defaults to 1000.
        overlap (int, optional): Defaults to 100.

    Returns:
        type: List of Langchain Docment Class
    """    

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    # load all pdfs in the directory
    loader = DirectoryLoader(
        path, glob="./*.docx",
        use_multithreading=True,
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True,
        recursive=True,
        silent_errors = False
    )

    # returns a list of pages as Document types
    word_docs = loader.load() 
    return word_docs


def load_json_file(path, **kwargs):
    """Recursively loads all the pdfs in a given directory path 
    and return a list containing pages of all documents. 

    Args:
        chunksize (int, optional): Defaults to 1000.
        overlap (int, optional): Defaults to 100.

    Returns:
        type: List of Langchain Docment Class
    """    

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import JSONLoader


    # load all pdfs in the directory
    loader = JSONLoader(path, **kwargs)

    # returns a list of pages as Document types
    json_docs = loader.load() 
    return json_docs


def load_html_files(path):
    """Recursively loads all the pdfs in a given directory path 
    and return a list containing pages of all documents. 

    Args:
        chunksize (int, optional): Defaults to 1000.
        overlap (int, optional): Defaults to 100.

    Returns:
        type: List of Langchain Docment Class
    """    

    from langchain_community.document_loaders import DirectoryLoader

    # load all pdfs in the directory
    loader = DirectoryLoader(
        path)

    # returns a list of pages as Document types
    htmls = loader.load() 
    return htmls

