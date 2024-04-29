import reflex as rx
import os, sys, openai
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory

from .rags import base_rag, rag_with_hyde, rag_with_query_aug, rag_with_react
from .utils import load_and_embed, get_pinecone_db
print()

# load env variables
# load_dotenv('.env')
# get openapi api from env
# openai.api_key = os.environ["OPENAI_API_KEY"]
# print(openai.api_key)
# client = openai.OpenAI()

# use rag or directly use openai_api
use_rag_mode = True
# all rag methods
rag_methods = {"base": base_rag, "hyde": rag_with_hyde, 
               "query_aug":rag_with_query_aug, "react": rag_with_react}

# load data and embed
# webui dir
dir_path = os.getcwd() # f"{os.path.abspath(__file__)}"

# to load and embed documents into vector db
# embedding_model = OpenAIEmbeddings()
# vector_db = load_and_embed(f"{dir_path}/data", embedding_model)


class State(rx.State):

    # The current question being asked.
    question: str
    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]] = []
    # openai key
    openai_api_key: str
    # to track rag initialization
    rag_init = False
    # possible rag values
    rag_values: list[str] = ["base", "query_aug", "hyde", "react"]
    # rag type
    rag_type: str = rag_values[0]
    # track rag_val changes
    current_rag_val : str = rag_values[0]


    def initialize_rag(self, rag_type):

        print(f"Initialzing llm & {rag_type} rag...")
        # get open api key if provided

        openai.api_key = self.openai_api_key

        # intialize the LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, 
                              api_key=self.openai_api_key)
        embedding_model = OpenAIEmbeddings(api_key=self.openai_api_key)

        # initialize retriever client
        vector_db = get_pinecone_db(embedding_model)
        # create retriver based on vectordb
        # search_type="mmr"
        self.retriever = vector_db.as_retriever(search_kwargs={'k': 5})

        # Intialize memory
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=2000,
                                                        return_messages=True, 
                                                        output_key="answer", 
                                                        input_key="question")
        # get the rag_chain
        # self.rag_chain = rag_with_query_aug(self.memory, retriever)
        self.rag_chain = rag_methods[rag_type](self.llm, self.memory, self.retriever)


    def change_rag_type(self):
        """Change the select value var."""
        self.rag_type = self.current_rag_val
        self.initialize_rag(self.rag_type)
        self.chat_history = []


    def answer(self):
        #
        # Our chatbot brain!
        if (use_rag_mode) and (not self.rag_init):
            # intialize the rag for first time
            self.initialize_rag(self.rag_type)

            # change the variable
            self.rag_init = True

        # define a empty response
        session = {}
        system_msg = [
                    {"role": "system", 
                     "content": ("You are a friendly chatbot named Bull Buddy."
                                 "You are here to help USF students with their admissions and onboarding.") }                 
                ]
        
        # Check if the question is empty
        if self.question == "":
            return

        if len(self.question):
            if use_rag_mode:
                inputs = {"question": self.question}
                session = self.rag_chain.stream(inputs)
            else:
                print("directly using chatgpt api...")
                msg = [{"role": "user", "content": self.question}]
                session = openai.OpenAI(
                    api_key=self.openai_api_key).chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=system_msg + msg,
                    stop=None,
                    temperature=0.9,
                    stream=True)
        
        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        if use_rag_mode:
            # save the memory for rag
            self.memory.save_context(inputs, {"answer": answer})

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        if use_rag_mode:
            for chunk in session:
                # print(chunk)
                for key in chunk:
                    if key == "answer":
                        answer += chunk[key].content
                        self.chat_history[-1] = (
                        self.chat_history[-1][0],
                        answer,)
                        yield
                    # print(f"{chunk[key].content}", end="", flush=True)
        
        else:
            for item in session:
                if item.choices[0].delta.content:
                    answer += item.choices[0].delta.content
                    self.chat_history[-1] = (
                        self.chat_history[-1][0],
                        answer,
                    )
                    yield