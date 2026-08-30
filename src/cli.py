import json
import os
from tqdm import tqdm  # type: ignore[import-untyped]

from src.models import (
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion)

from .retrieval import BM25Retrieval
from .indexer import Indexer
from .llm_model import LLModel
from .evaluator import Evaluator


class CLI:
    def __init__(self):
        self.chunks = []
        self.evaluator = Evaluator()
        self.llm = LLModel()
        self.index(force=False, max_chunk_size=2000)

        self.bm25 = BM25Retrieval(self.chunks)

    def index(self, force=False, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(max_chunk_size=max_chunk_size)
        chunks = indexer.run(force)
        self.chunks = chunks

    def search(self,
               query: str | UnansweredQuestion,
               k: int, p: bool = True) -> MinimalSearchResults:

        if isinstance(query, UnansweredQuestion):
            question = query
        else:
            question = UnansweredQuestion(question=query)
        results = self.bm25.retrieve(question.question, k)
        ms_results = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=results)

        ss_results = StudentSearchResults(
            search_results=[ms_results],
            k=k)

        output = ss_results.model_dump_json(indent=4)
        if p:
            print(output)
        return ms_results

    def search_dataset(
            self, dataset_path: str,
            k: int, save_directory: str):

        with open(dataset_path, "r") as f:
            questions = json.load(f)

        questions = questions.get("rag_questions", [])
        unanswered_questions = [
            UnansweredQuestion(**question) for question in questions]
        ms_results = []

        progress_bar = tqdm(unanswered_questions,
                            desc="Processing questions",
                            unit="question",
                            total=len(unanswered_questions))
        for question in unanswered_questions:
            progress_bar.update(1)
            ms_result = self.search(question, k, p=False)
            ms_results.append(ms_result)
        progress_bar.close()

        ss_results = StudentSearchResults(search_results=ms_results, k=k)
        result_dict = ss_results.model_dump()
        # for result in result_dict["search_results"]:
        #     del result["retrieved_sources"][0]["content"]
        #     del result["retrieved_sources"][0]["file_type"]
        #     del result["retrieved_sources"][0]["score"]
        #     del result["retrieved_sources"][0]["metadata"]

        output_path = f"{save_directory}/{dataset_path.split('/')[-1]}"
        os.makedirs(save_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=4)

    def answer(self,
               query: str | UnansweredQuestion,
               k: int, p: bool = True) -> StudentSearchResultsAndAnswer:
        sources = self.search(query, k, p=False)
        messages = self.llm.generate_prompt(query, sources.retrieved_sources)
        answer = self.llm.generate(messages)
        min_answer = MinimalAnswer(
            question_id=sources.question_id,
            question=sources.question,
            retrieved_sources=sources.retrieved_sources,
            answer=answer
        )
        ss_results_and_answers = StudentSearchResultsAndAnswer(
            search_results=[min_answer], k=k)
        output = ss_results_and_answers.model_dump_json(indent=4)
        if p:
            print(output)

        return ss_results_and_answers

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str):
        with open(student_search_results_path, "r") as f:
            questions = json.load(f)

        questions = questions.get("rag_questions", [])
        unanswered_questions = [
            UnansweredQuestion(**question) for question in questions]
        min_answers = []

        progress_bar = tqdm(unanswered_questions,
                            desc="Processing questions",
                            unit="question",
                            total=len(unanswered_questions))
        for question in unanswered_questions:
            progress_bar.update(1)
            sources = self.search(question, k=10, p=False)
            messages = self.llm.generate_prompt(
                question.question, sources.retrieved_sources)
            answer = self.llm.generate(messages)
            min_answer = MinimalAnswer(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources.retrieved_sources,
                answer=answer
            )
            min_answers.append(min_answer)
        progress_bar.close()

        ss_results_and_answers = StudentSearchResultsAndAnswer(
            search_results=min_answers, k=10)

        output_path = (
            f"{save_directory}/"
            f"{student_search_results_path.split('/')[-1]}"
            )

        os.makedirs(save_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ss_results_and_answers.model_dump(), f, indent=4)

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str):
        # recall = self.evaluator.evaluate(
        #     student_search_results_path,
        #     dataset_path
        # )
        return
