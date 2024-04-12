import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAI
from langchain.memory import ConversationSummaryBufferMemory

from utils import load_and_embed
from rags import base_rag, rag_with_hyde, rag_with_query_aug, rag_with_react


if __name__ == "__main__":

    from simple_term_menu import TerminalMenu
    
    # list all rag types
    options = ["base", "query_aug", "hyde", "ReAct"]
    terminal_menu = TerminalMenu(options)
    idx = terminal_menu.show()
    print(f"Building {options[idx]} RAG...")

    # get the rag type
    rag_type = options[idx]

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

    if rag_type == "query_aug":
        rag_chain = rag_with_query_aug(memory, retriever)

    if rag_type == "hyde":
        rag_chain = rag_with_hyde(memory, retriever)

    if rag_type.lower() == "react":
        rag_chain = rag_with_react(memory, retriever)
    

    # while True:
    #     question_input = input("\nUser: ")
    #     if question_input == "exit":
    #         break
    #     inputs = {"question": question_input}
    #     result = rag_chain.invoke(inputs)
    #     # print(result.keys())
    #     answer = result["answer"].content
    #     print(f"Agent: {answer}")
    #     memory.save_context(inputs, {"answer": answer})
    #     print("\n")
    #     print(memory.load_memory_variables({}))
    #     print(result["docs"])


    #     rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain_from_docs)

    
    while True:
        question_input = input("\n\nUser: ")
        if question_input == "exit":
            break

        inputs = {"question": question_input}
        output = {}
        curr_key = None
        print("\nanswer:")
        for chunk in rag_chain.stream(inputs):
            # print(chunk)
            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]
                if key == "answer":
                    print(f"{chunk[key].content}", end="", flush=True)
                # else:
                #     print(chunk[key], end="", flush=True)
                # curr_key = key