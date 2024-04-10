import reflex as rx
import os, sys
import openai
from dotenv import load_dotenv

# load_dotenv('../.env')
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.OpenAI()

class State(rx.State):

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

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
            session = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=system_msg + msg,
                stop=None,
                temperature=0.9,
                stream=True,
            )
        

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield


        for item in session:
            if item.choices[0].delta.content:
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield
