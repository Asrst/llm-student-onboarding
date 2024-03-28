import json, ast
import os, sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import StrOutputParser

from utils import load_and_embed, str_to_json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# get api key
load_dotenv('.env')

def evaluate_with_llm(model, query, generated_text):

    """
    Uses a Large Language Model (LLM) to evaluate generated text.

    :param model: An instance of the LLM, ready to generate responses.
    :param query: The original query given to the system.
    :param generated_text: The text generated by the SELF-RAG system.
    :return: A dictionary containing critique scores or assessments.
    """

    evaluations = {}

    # Template for creating evaluation queries
    def evaluate_query(template, **kwargs):
        query = ChatPromptTemplate.from_template(template)
        chain = query | model
        return float(chain.invoke(kwargs).content)

    # Evaluate Relevance
    relevance_template = "Given the context provided by the following prompt: '{prompt}', please evaluate on a scale from 0 to 1, where 1 is highly relevant and 0 is not relevant at all, how relevant is this generated response: '{generated_text}'? Provide a numerical score only."
    evaluations['relevance'] = evaluate_query(relevance_template, prompt=query, generated_text=generated_text)

    # Evaluate Clarity
    clarity_template = "How clear and easily understandable is this text: '{generated_text}'? Rate its clarity on a scale from 0 to 1, where 1 indicates that the text is very clear and 0 indicates that the text is very unclear. Provide a numerical score only."
    evaluations['clarity'] = evaluate_query(clarity_template, prompt=query, generated_text=generated_text)

    # Evaluate Coherence
    coherence_template = "On a scale from 0 to 1, with 1 being highly coherent and 0 being not coherent at all, how well do the ideas in this generated text: '{generated_text}' flow together? Consider if the text makes logical sense as a whole. Provide a numerical score only."
    evaluations['coherence'] = evaluate_query(coherence_template, prompt=query, generated_text=generated_text)

    # Evaluate Detail and Exhaustiveness
    detail_template = "Assessing the detail and exhaustiveness relative to the prompt '{prompt}', how thoroughly does this generated text: '{generated_text}' cover the topic? Rate on a scale from 0 to 1, where 1 is very detailed and exhaustive, and 0 is not detailed at all. Provide a numerical score only."
    evaluations['details'] = evaluate_query(detail_template, prompt=query, generated_text=generated_text)

    # Evaluate Suitability as an Answer
    suitability_template = "Evaluate the suitability of this generated text: '{generated_text}' as an answer to the original prompt '{prompt}'. On a scale from 0 to 1, where 1 is a perfect answer and 0 is completely unsuitable, provide a numerical score only."
    evaluations['suitability'] = evaluate_query(suitability_template, prompt=query, generated_text=generated_text)

    return evaluations
    
def critique_with_llm(model, query, generated_text):

    evaluation_weights = {
        'relevance': 3,
        'clarity': 1,
        'coherence': 0.5,
        'details': 1.5,
        'suitability': 2
    }

    evaluations = evaluate_with_llm(model, query, generated_text)
    print("Evaluations:", evaluations)

    # Calculate the weighted sum of the evaluations
    weighted_sum = sum(evaluations[aspect] * evaluation_weights.get(aspect, 1) for aspect in evaluations)

    # Calculate the sum of weights for the aspects evaluated
    total_weight = sum(evaluation_weights.get(aspect, 1) for aspect in evaluations)

    # Calculate the weighted average of the evaluations
    weighted_average = weighted_sum / total_weight if total_weight > 0 else 0

    return [weighted_average, evaluations]


class QueryDetail:

    def __init__(self, query, retriever):
        self.query = query
        self.retriever = retriever
        self.content: List[str] = []
        self.critique_score: float = 0.0
        self.critique_details: Dict[str, Any] = {}
        self.retrieval_needed: bool = False
        self.search_needed: bool = False

    def add_response(self, model, search) -> None:

        """Process the query to add response, handle retrieval and critique."""

        if self.is_retrieval_needed(model, self.query):
            contexts = self.retriever.get_relevant_documents(self.query)
            # print(contexts)
            response = "\n".join([c.page_content for c in contexts])
            self.retrieval_needed = True
        else:
            response = "Some generated answer"
            self.retrieval_needed = False

        self.content.append(response)

        critique_score, critique_details = critique_with_llm(model, self.query, response)
        self.critique_score = critique_score
        self.critique_details = critique_details
        self.search_needed = critique_score < 0.5

        if self.search_needed:
            self.search_and_add_results(search)

    @staticmethod
    def is_retrieval_needed(model, prompt):
        is_retrieval_needed_prompt = ChatPromptTemplate.from_template("Given the prompt: '{prompt}', is retrieval from an external source necessary to answer the question? Reply with only True or False")
        is_retrieval_needed_chain = is_retrieval_needed_prompt | model
        return is_retrieval_needed_chain.invoke({"prompt": prompt}).content
    

    def search_and_add_results(self, search) -> None:
        """Perform a search and process the results if critique score is low."""
        search_result_raw = search.run(self.query)
        print(search_result_raw)
        search_result = str_to_json(search_result_raw) or []
        self.content.extend(search_result)


class QueryProcessor:

    def __init__(self, model, retriver, search, queries: List[str]):
        self.model = model
        self.search = search
        self.queries = [QueryDetail(query=q, retriever=retriver) for q in queries]

    def process_queries(self) -> List[QueryDetail]:
        """Process each query in the list."""
        for query_detail in self.queries:
            query_detail.add_response(self.model, self.search)
            if query_detail.search_needed:
                consolidated_response = consolidate(self.model, query_detail.content)
                query_detail.content = [consolidated_response]
                critique_score, critique_details = critique_with_llm(self.model, query_detail.query, consolidated_response)
                query_detail.critique_score = critique_score
                query_detail.critique_details = critique_details
        return self.queries
    

def consolidate(model, text):
    consolidate_prompt = ChatPromptTemplate.from_template("Given the following set of texts, please consolidate them: '{text}'")
    consolidate_chain = consolidate_prompt | model
    return consolidate_chain.invoke({"text": text}).content


def generate_queries(model,prompt, num_queries):
  query_generation_prompt = ChatPromptTemplate.from_template("Given the question: '{prompt}', generate {num_queries} questions that are better articulated. Return the output in the form of a json with key questions")
  query_generation_chain = query_generation_prompt | model
  output = query_generation_chain.invoke({"prompt": prompt, "num_queries": num_queries})
  return str_to_json(output.content)["questions"]


def rag_runner(model, retriver, query, num_queries=3):

    search = DuckDuckGoSearchResults()
    initial_queries = generate_queries(model, query, num_queries)
    print(initial_queries)
    query_processor = QueryProcessor(model, retriver, search, initial_queries[:num_queries])
    processed_queries = query_processor.process_queries()
    combined_content = " ".join(content for result in processed_queries for content in result.content)
    rag_results = consolidate(model, combined_content)

    return rag_results



if __name__ == "__main__":
    # intialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # # Intialize memory
    # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000,
    #                                                 return_messages=True, 
    #                                                 output_key="answer", 
    #                                                 input_key="question")
    
    # load data and embed
    retriever = load_and_embed("data")


    while True:
        question_input = input("\nUser: ")
        if question_input == "exit":
            break
        result = rag_runner(llm, retriever, question_input)
        answer = result
        print(f"Agent: {answer}")
        # memory.save_context(inputs, {"answer": answer})
        # print("\n")
        # print(memory.load_memory_variables({}))

