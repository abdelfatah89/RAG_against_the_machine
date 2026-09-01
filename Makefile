run:
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb src

clean:
	rm -rf */*.pyc */__pycache__/ .mypy_cache/

fclean: clean
	rm -rf data/output/search_results/*
	rm -rf data/processed/*

index:
	@uv run python -m src index --max_chunk_size=2000 --force True

search-private:
	@uv run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_code_private.json \
	--k 10 --save_directory data/output/search_results/
	@uv run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_docs_private.json \
	--k 10 --save_directory data/output/search_results/

search-public:
	@uv run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
	--k 10 --save_directory data/output/search_results/
	@uv run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
	--k 10 --save_directory data/output/search_results/

moulinette-code-private:
	@./moulinette evaluate_student_search_results \
	data/output/search_results/dataset_code_private.json \
	data/datasets/AnsweredQuestions/dataset_code_private.json \
	--k 10

moulinette-code-public:
	@./moulinette evaluate_student_search_results \
	data/output/search_results/dataset_code_public.json \
	data/datasets/AnsweredQuestions/dataset_code_public.json \
	--k 10

moulinette-docs-private:
	@./moulinette evaluate_student_search_results \
	data/output/search_results/dataset_docs_private.json \
	data/datasets/AnsweredQuestions/dataset_docs_private.json \
	--k 10
moulinette-docs-public:
	@./moulinette evaluate_student_search_results \
	data/output/search_results/dataset_docs_public.json \
	data/datasets/AnsweredQuestions/dataset_docs_public.json \
	--k 10

start-local-api:
	@uv run python -m src.local_api

lint:
lint-strict: