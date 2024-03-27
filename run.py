import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAI
from langchain.memory import ConversationSummaryBufferMemory

from utils import load_and_embed
from rags import base_rag, rag_with_hyde, rag_with_query_aug


if __name__ == "__main__":
    rag_type = "base"

    # intialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Intialize memory
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000,
                                                    return_messages=True, 
                                                    output_key="answer", 
                                                    input_key="question")
    
    # load data and embed
    retriever = load_and_embed("data")

    # get the rag chain
    if rag_type == "base":
        rag_chain = base_rag(memory, retriever)

    if rag_type == "data_aug":
        rag_chain = rag_with_query_aug(memory, retriever)

    if rag_type == "hyde":
        rag_chain = rag_with_hyde(memory, retriever)

    # if rag_type == "react":
    #     rag_with_react()
        

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
