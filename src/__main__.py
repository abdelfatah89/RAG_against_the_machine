from fire import Fire  # type: ignore[import-untyped]

from .cli import CLI
from .indexer import Indexer


def main() -> None:
    cli = CLI()
    Fire(cli)

    # indexer = Indexer(data_dir="data/raw", max_chunk_size=2000)
    # _ = indexer.run()

    # bm25 = cli.bm25
    # k = 3
    # with open(dataset_path, "r") as f:
    #     questions = json.load(f)
    # questions = questions.get("rag_questions", [])
    # unanswered_questions = [
    #     UnansweredQuestion(**question) for question in questions]
    # answered_questions = []
    # for question in unanswered_questions:
    #     results = bm25.retrieve(question.question, k)
    #     answered_question = AnsweredQuestion(
    #         question_id=question.question_id,
    #         question=question.question,
    #         sources=results,
    #         answer="This is a placeholder answer."
    #     )
    #     answered_questions.append(answered_question)

    # with open(f"{save_directory}/{dataset_path.split('/')[-1]}", "w") as f:
    #     json.dump(
    #         {"rag_questions": [q.model_dump() for q in answered_questions]},
    #         f,
    #         indent=4)
    # output = answered_questions.model_dump_json(indent=4)
    # print(output)


if __name__ == "__main__":
    main()
