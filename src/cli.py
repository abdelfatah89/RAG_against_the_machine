import json
import os
from typing import List
from tqdm import tqdm

from src.models import (
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion)

from .retrieval import BM25Retrieval, HybridRetrieval
from .indexer import Indexer
from .llm_model import LLModel
from .evaluator import Evaluator
from .chunker import Chunk
from .tools import _safe


class CLI:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.evaluator = Evaluator()
        self.llm = LLModel()
        self.hybrid: HybridRetrieval | None = None
        self.index(force=False, embed=False, max_chunk_size=2000)

        self.bm25 = BM25Retrieval(self.chunks)

    def _hybrid(self) -> HybridRetrieval:
        if self.hybrid is None:
            self.hybrid = HybridRetrieval(self.chunks)
        return self.hybrid

    @_safe
    def index(self,
              data_dir: str = "data/raw",
              force: bool = False,
              embed: bool = False,
              max_chunk_size: int = 2000) -> None:
        """Build or load the document index."""
        indexer = Indexer(data_dir=data_dir, max_chunk_size=max_chunk_size)
        chunks = indexer.run(force, embed)
        self.chunks = chunks

    @_safe
    def search(self,
               query: str | UnansweredQuestion,
               k: int, p: bool = True) -> MinimalSearchResults:
        """Retrieve the top-k sources for one query."""

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

    @_safe
    def search_dataset(
            self, dataset_path: str,
            k: int, save_directory: str) -> None:
        """Run retrieval for every question in a dataset and save JSON."""

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

        output_path = f"{save_directory}/{dataset_path.split('/')[-1]}"
        os.makedirs(save_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=4)

    @_safe
    def answer(self,
               query: str | UnansweredQuestion,
               k: int, p: bool = True) -> StudentSearchResultsAndAnswer | None:
        """Answer one query using retrieved context."""
        sources = self.search(query, k, p=False)
        source_texts = [source.content for source in sources.retrieved_sources]
        messages = self.llm.generate_prompt(sources.question, source_texts)
        answer = self.llm.generate(messages)
        min_answer = MinimalAnswer(
            question_id=sources.question_id,
            question=sources.question,
            retrieved_sources=sources.retrieved_sources,
            answer=answer
        )
        ss_results_and_answers = StudentSearchResultsAndAnswer(
            search_results=[min_answer], k=k)
        output_dict = ss_results_and_answers.model_dump()
        for mr in output_dict["search_results"]:
            for source in mr["retrieved_sources"]:
                del source["content"]
                del source["metadata"]
                del source["file_type"]
                del source["score"]
        output = json.dumps(output_dict, indent=4)
        if p:
            print(output)
            return None

        return ss_results_and_answers

    @_safe
    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a saved search-results dataset."""
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
            source_texts = [
                source.content for source in sources.retrieved_sources]
            messages = self.llm.generate_prompt(
                question.question, source_texts)
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

    @_safe
    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall for student search results against ground truth."""
        try:
            recall = self.evaluator.evaluate(
                student_search_results_path,
                dataset_path
            )
        except ValueError as exc:
            print(f"Evaluation failed: {exc}")
            return

        print(f"Recall: {recall:.3f} ({recall * 100:.1f}%)")

    @_safe
    def hybrid_search(self,
                      query: str | UnansweredQuestion,
                      k: int = 10,
                      bm25_factor: float = 0.3,
                      embedding_factor: float = 0.7,
                      p: bool = True) -> MinimalSearchResults | None:
        """Retrieve top-k sources with combined BM25 and embeddings."""
        if isinstance(query, UnansweredQuestion):
            question = query
        else:
            question = UnansweredQuestion(question=query)

        self.hybrid = self._hybrid()
        results = self.hybrid.retrieve(
            question.question,
            k,
            bm25_factor=bm25_factor,
            embedding_factor=embedding_factor)
        ms_results = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=results)

        ss_results = StudentSearchResults(
            search_results=[ms_results],
            k=k)
        output_dict = ss_results.model_dump()
        for mr in output_dict["search_results"]:
            for source in mr["retrieved_sources"]:
                del source["content"]
                del source["metadata"]
                del source["file_type"]
        output = json.dumps(output_dict, indent=4)
        if p:
            print(output)
            return None
        return ms_results

    @_safe
    def hybrid_answer(
            self,
            query: str | UnansweredQuestion,
            k: int = 10,
            bm25_factor: float = 0.3,
            embedding_factor: float = 0.7,
            p: bool = True) -> StudentSearchResultsAndAnswer | None:
        """Answer one query using hybrid retrieved context."""
        sources = self.hybrid_search(
            query,
            k=k,
            bm25_factor=bm25_factor,
            embedding_factor=embedding_factor,
            p=False)
        if sources is None:
            return None

        source_texts = [source.content for source in sources.retrieved_sources]
        messages = self.llm.generate_prompt(sources.question, source_texts)
        answer = self.llm.generate(messages)
        min_answer = MinimalAnswer(
            question_id=sources.question_id,
            question=sources.question,
            retrieved_sources=sources.retrieved_sources,
            answer=answer
        )
        ss_results_and_answers = StudentSearchResultsAndAnswer(
            search_results=[min_answer], k=k)
        output_dict = ss_results_and_answers.model_dump()
        for mr in output_dict["search_results"]:
            for source in mr["retrieved_sources"]:
                del source["content"]
                del source["metadata"]
                del source["file_type"]
        output = json.dumps(output_dict, indent=4)
        if p:
            print(output)
            return None

        return ss_results_and_answers
