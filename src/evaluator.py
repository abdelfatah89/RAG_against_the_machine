"""Evaluator for the RAG retrieval pipeline.

Computes recall@k by comparing student-produced search results against
a ground-truth dataset of RAG questions, using an Intersection-over-Union
(IoU) overlap of character ranges to decide whether a retrieved source
matches a correct source.

Character ranges use an exclusive end index, i.e. ``last_character_index``
behaves like Python slicing (``text[first:last]``), matching the reference
moulinette's behaviour as verified by black-box testing.
"""

import json
from typing import Any, Dict, List, Optional

IOU_THRESHOLD = 0.05


class Evaluator:
    """Computes recall@k for a set of retrieved sources.

    A retrieved source is counted as a match for a correct source when
    both refer to the same file and the IoU of their character ranges
    is at least ``IOU_THRESHOLD``.
    """

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str) -> float:
        """Compute the average recall over every question in the dataset.

        Args:
            student_search_results_path: Path to a JSON file following
                the ``StudentSearchResults`` pydantic model.
            dataset_path: Path to a JSON file following the
                ``RagDataset`` model (ground truth, with ``sources``).

        Returns:
            The average recall across all evaluated questions, as a
            float in [0.0, 1.0]. Returns 0.0 if there are no questions
            with correct sources to evaluate against.

        Raises:
            ValueError: If a file is missing, is not valid JSON, or a
                retrieved/correct source is missing character indices.
        """
        student_search_results = self._load_json(
            student_search_results_path)
        dataset = self._load_json(dataset_path)

        questions = dataset.get("rag_questions", [])
        search_results = student_search_results.get("search_results", [])

        recalls: List[float] = []
        for question in questions:
            correct_sources = question.get("sources", [])
            if not correct_sources:
                continue

            retrieved_sources = self._get_retrieved_sources(
                question.get("question_id"),
                search_results)

            matched = 0
            for correct_source in correct_sources:
                if self._has_match(correct_source, retrieved_sources):
                    matched += 1

            recalls.append(matched / len(correct_sources))

        return sum(recalls) / len(recalls) if recalls else 0.0

    def _has_match(
        self,
        correct_source: Dict[str, Any],
        retrieved_sources: List[Dict[str, Any]],
    ) -> bool:
        """Check whether any retrieved source matches a correct source."""
        for retrieved_source in retrieved_sources:
            if (correct_source.get("file_path") !=
                    retrieved_source.get("file_path")):
                continue

            retrieved_start = retrieved_source.get("first_character_index")
            retrieved_end = retrieved_source.get("last_character_index")
            if (not isinstance(retrieved_start, int) or
                    not isinstance(retrieved_end, int)):
                raise ValueError(
                    "Retrieved source is missing character indices.")

            iou = self._calculate_iou(
                correct_source.get("first_character_index"),
                correct_source.get("last_character_index"),
                retrieved_start,
                retrieved_end,
            )
            if iou > IOU_THRESHOLD:
                return True

        return False

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        """Load and parse a JSON file, raising a clear error on failure.

        Args:
            path: Path to the JSON file to load.

        Returns:
            The parsed JSON content as a dictionary.

        Raises:
            ValueError: If the file does not exist or is not valid JSON.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content: Dict[str, Any] = json.load(handle)
                return content
        except FileNotFoundError as exc:
            raise ValueError(f"File not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    def _calculate_iou(
        self,
        first_start: Optional[int],
        first_end: Optional[int],
        second_start: int,
        second_end: int,
    ) -> float:
        """Compute the Intersection-over-Union of two integer ranges.

        Ranges use an exclusive end index, i.e. the same convention as
        Python slicing (``text[start:end]``). This matches the reference
        moulinette's behaviour, confirmed by black-box testing: e.g. a
        correct range of ``[100, 199]`` against a retrieved range of
        ``[90, 105]`` is *not* a match, which only holds under the
        exclusive-end convention (IoU ~0.046) and not the inclusive one
        (IoU ~0.055, which would incorrectly count as a match).

        Args:
            first_start: Start index of the first (correct) range.
            first_end: Exclusive end index of the first (correct) range.
            second_start: Start index of the second (retrieved) range.
            second_end: Exclusive end index of the second (retrieved)
                range.

        Returns:
            The IoU as a float in [0.0, 1.0].

        Raises:
            ValueError: If either range is malformed (start > end) or
                the correct source's indices are missing.
        """
        if first_start is None or first_end is None:
            raise ValueError("Correct source is missing character indices.")
        if first_start > first_end:
            raise ValueError("first_start cannot be greater than first_end")
        if second_start > second_end:
            raise ValueError(
                "second_start cannot be greater than second_end")

        intersection_start = max(first_start, second_start)
        intersection_end = min(first_end, second_end)
        if intersection_start >= intersection_end:
            return 0.0

        intersection = intersection_end - intersection_start
        first_length = first_end - first_start
        second_length = second_end - second_start
        union = first_length + second_length - intersection

        return intersection / union if union > 0 else 0.0

    def _get_retrieved_sources(
        self,
        question_id: Optional[str],
        search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return the retrieved sources for a given question id.

        Args:
            question_id: The id of the question to look up.
            search_results: The list of ``MinimalSearchResults``-like
                dictionaries from the student's output.

        Returns:
            The list of retrieved source dictionaries for the question,
            or an empty list if the question id was not found.
        """
        for result in search_results:
            if result.get("question_id") == question_id:
                sources: List[Dict[str, Any]] = result.get(
                    "retrieved_sources", [])
                return sources
        return []
