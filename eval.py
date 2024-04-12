import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory

from utils import load_and_embed
from rags import base_rag, rag_with_hyde, rag_with_query_aug, rag_with_react

from datasets import Dataset
import pandas as pd
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    context_recall,
    context_precision,
    context_relevancy, 
)

# gen syntetic questions
# test_df = testset.to_pandas()

if __name__ == "__main__":

    # intialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    openai_embed = OpenAIEmbeddings()

    # Intialize memory
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000,
                                                    return_messages=True, 
                                                    output_key="answer", 
                                                    input_key="question")
    
    # load data and embed
    retriever = load_and_embed("data")

    # load test data
    test_df = pd.read_csv("scripts/test_data.csv")
    print("test data: ", test_df.shape)
    test_questions = test_df["question"].values.tolist()
    test_groundtruths = test_df["ground_truth"].values.tolist()

    # metrics for evaluation
    metrics = [
            answer_relevancy,
            answer_correctness,
            answer_similarity,
            context_recall,
            context_precision,
            context_relevancy, 
        ]


    options = ["base", "query_aug", "hyde", "ReAct"]

    all_results = []
    for rag_type in options[:]:
            # get the rag chain
        if rag_type == "base":
            rag_chain = base_rag(memory, retriever)

        if rag_type == "query_aug":
            rag_chain = rag_with_query_aug(memory, retriever)

        if rag_type == "hyde":
            rag_chain = rag_with_hyde(memory, retriever)

        if rag_type.lower() == "react":
            rag_chain = rag_with_react(memory, retriever)

        answers = []
        contexts = []

        print(f"\n\nEvaluating {rag_type} RAG on the test data...")
        for question in test_questions:
            response = rag_chain.invoke({"question" : question})
            answers.append(response["answer"].content)
            contexts.append([context.page_content for context in response['docs']])

        # create hf dataset
        response_dataset = Dataset.from_dict({
            "question" : test_questions,
            "answer" : answers,
            "contexts" : contexts,
            "ground_truth" : test_groundtruths
        })

        print("sample response: ")
        print(response_dataset[0])

        # run_config = RunConfig(
        #                 timeout=10,
        #                 max_retries=3,
        #                 max_wait=10,
        #                 max_workers=10,
        #             )

        results = evaluate(response_dataset, metrics, 
                           llm=llm, embeddings=openai_embed,
                           raise_exceptions=False,
                           # run_config=run_config,
                           )
        print(results)
        results["rag_type"] = rag_type
        all_results.append(results.scores)

        # save results to dataframe. overwrite df for each rag.
        res_df = pd.DataFrame(all_results)
        res_df.to_csv("eval_results.csv", index=False)

        print("-"*50)

