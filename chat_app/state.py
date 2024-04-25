import reflex as rx
import os, sys
import openai
from dotenv import load_dotenv

from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAI
from langchain.memory import ConversationSummaryBufferMemory

from .rags import base_rag, rag_with_hyde, rag_with_query_aug, rag_with_react
from .utils import load_and_embed
print()

load_dotenv('.env')
openai.api_key = os.environ["OPENAI_API_KEY"]
# print(openai.api_key)
client = openai.OpenAI()

# load data and embed
# webui dir
dir_path = os.getcwd() # f"{os.path.abspath(__file__)}"
retriever = load_and_embed(f"{dir_path}/data")
# intialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
# Intialize memory
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000,
                                                return_messages=True, 
                                                output_key="answer", 
                                                input_key="question")
# get the rag_chain
rag_chain = rag_with_query_aug(memory, retriever)


class State(rx.State):

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    # client = openai.OpenAI()

    def answer(self):
        # Our chatbot has some brains now!

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
            msg = [{"role": "user", "content": self.question}]
            # session = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=system_msg + msg,
            #     stop=None,
            #     temperature=0.9,
            #     stream=True,
            # )

            inputs = {"question": self.question}
            session =  rag_chain.stream(inputs)

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))
        # save the memory for rag
        memory.save_context(inputs, {"answer": answer})

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        # for item in session:
        #     if item.choices[0].delta.content:
        #         answer += item.choices[0].delta.content
        #         self.chat_history[-1] = (
        #             self.chat_history[-1][0],
        #             answer,
        #         )
        #         yield

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

