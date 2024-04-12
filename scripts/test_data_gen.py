import os, sys
from dotenv import load_dotenv
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import load_docx_files, load_pdfs



load_dotenv('.env')


documents = []
docs_path = "data"
word_docs = load_docx_files(f"{docs_path}/docx")
pdfs = load_pdfs(f"{docs_path}/pdfs")
documents = pdfs + word_docs


for document in documents:
    # print(document.metadata)
    document.metadata['filename'] = document.metadata['source']

    # if document.metadata["source"].endswith("docx"):
    #     print(document)



# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=100, 
                                                 distributions={simple: 0.5, reasoning: 0.25, 
                                                                multi_context: 0.25})


test_df = testset.to_pandas()
test_df.to_csv("test_data.csv", index=False)