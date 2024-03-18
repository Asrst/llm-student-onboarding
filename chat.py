import os, sys
import logging
from dotenv import load_dotenv
from operator import itemgetter

from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter


# configure logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# load open_ai key
load_dotenv('.env')

# setup a llm to use
llm = OpenAI(model="gpt-3.5-turbo-16k", temperature=0, max_tokens=512)

# model for embedding
# embed_model = "local:BAAI/bge-base-en" 
embed_model = OpenAIEmbedding(embed_batch_size=64)

# node parser or document splitter
# node_parser = SentenceSplitter.from_defaults(
#         chunk_size=NODE_PARSER_CHUNK_SIZE,
#         chunk_overlap=NODE_PARSER_CHUNK_OVERLAP,
#         callback_manager=callback_manager,
#     )

# setup a global service context
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data/pdfs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as support"
        " students at USF regarding MS BAIS program."
    ),

# Either way we can now query the index
chat_engine = index.as_chat_engine(system_prompt=(
        "You are a student support chatbot, able to have normal interactions and assist"
        " students on their regarding MS BAIS program."
    ),
)

while True:
    text_input = input("\nUser: ")
    if text_input == "exit":
        break
    # response = chat_engine.chat(text_input)
    # print(f"Agent: {response}")

    streaming_response = chat_engine.stream_chat(text_input)
    for token in streaming_response.response_gen:
        print(token, end="")

    print("\n", chat_engine.chat_history)