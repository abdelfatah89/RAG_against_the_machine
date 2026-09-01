*This project has been created as part of the 42 curriculum by <alaktaou>.*

# RAG against the machine

## Description

**RAG against the machine** is a Retrieval-Augmented Generation (RAG) system designed to answer questions about a codebase.

The project implements a complete RAG pipeline:

1. **Indexing** — ingest the provided vLLM codebase, split files into chunks, and build a searchable index.
2. **Retrieval** — retrieve the most relevant code or documentation snippets for a user question.
3. **Augmentation** — provide the retrieved snippets as context to the language model.
4. **Generation** — use `Qwen/Qwen3-0.6B` to generate an answer grounded in the retrieved context.

The system also provides retrieval evaluation using **Recall@k**.

The main goal is not only to generate an answer, but to retrieve the correct source locations that contain the information needed to answer the question.

---

## Features

* Python 3.10+ implementation
* Python Fire command-line interface
* Python type hints and Pydantic data models
* Unified sliding-window chunking strategy applied to:

  * Python source code
  * Markdown/text documents
* Configurable chunk size
* Lexical retrieval using **BM25**
* Optional semantic retrieval using **all-MiniLM-L6-v2** embeddings and ChromaDB
* Optional hybrid retrieval combining BM25 and embedding scores
* Single-query retrieval
* Dataset-based retrieval
* Local answer generation using `Qwen/Qwen3-0.6B`
* Dataset-based answer generation
* Recall@k evaluation for development and testing
* `tqdm` progress bars for long-running operations
* Makefile for common development tasks
* Optional bonus features:

  * Semantic embeddings
  * Hybrid retrieval
  * Incremental indexing
  * Caching
  * Local HTTP API

---

# System Architecture

The system is organized as a pipeline where each stage produces data consumed by the next stage.

```text
                         ┌──────────────────┐
                         │   data/raw/      │
                         │    vLLM codebase │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Indexer      │
                         └────────┬─────────┘
                                  │
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Sliding Window  │
                         │     Chunker      │
                         │ (py / md / txt)  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Search Index    │
                         │ data/processed/  │
                         └────────┬─────────┘
                                  │
                         User question
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Retriever     │
                         │      BM25        │
                         └────────┬─────────┘
                                  │
                             Top-k sources
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Augmenter     │
                         │ Retrieved context│
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Qwen/Qwen3-0.6B  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Answer      │
                         └──────────────────┘
```

### Pipeline stages

#### 1. Indexing

The indexer reads the useful files from `data/raw/`, applies the appropriate chunking strategy, and persists the resulting index under `data/processed/`.

The chunk size is configurable through `--max_chunk_size` and must not exceed 2000 characters.

#### 2. Retrieval

A question is passed to the retrieval system.

The retriever ranks indexed chunks according to their relevance and returns the top `k` source locations.

Every result contains:

* `file_path`
* `first_character_index`
* `last_character_index`

The file path must correspond exactly to the path used in the indexed corpus.

#### 3. Augmentation

The retrieved chunks are assembled into the context given to the language model.

Only retrieved information is provided as evidence for the generated answer.

#### 4. Generation

The system uses `Qwen/Qwen3-0.6B` to generate a natural-language answer from the retrieved context.

---

# Chunking Strategy

All supported file types — Python source, Markdown, and plain text — are chunked using the same **sliding window** strategy, so there is a single chunking code path for the whole corpus.

## Sliding Window Chunking

Each file's raw content is read as a single string, and a fixed-size window is slid across it from the start of the file to the end:

* the window is at most `max_chunk_size` characters wide;
* consecutive windows overlap by a small fraction of `max_chunk_size` (roughly 10%), so content that sits right on a window boundary still appears in full in at least one chunk;
* when a window would end mid-line, the cut is nudged back to the nearest preceding newline (as long as that doesn't shrink the window by more than half), so chunks tend to end on a line boundary rather than mid-token;
* every chunk is an exact substring of the source file — `content == source[first_character_index:last_character_index]` always holds.

This strategy is intentionally structure-agnostic: it does not parse Python ASTs or Markdown headings to find "meaningful" units. The trade-off is that a chunk can occasionally cut through the middle of a function or a section, but it keeps the chunker simple, fast, and uniform across file types, and it guarantees the character-offset exactness the evaluator relies on.

The resulting chunks never exceed the configured maximum chunk size.

### Maximum chunk size

The default maximum chunk size is:

```text
2000 characters
```

This limit is important because the evaluation system rejects retrieved sources longer than 2000 characters.

The value can be changed through:

```bash
--max_chunk_size
```

For example:

```bash
uv run python -m src index --max_chunk_size 2000
```

---

# Retrieval Method

The mandatory lexical retrieval method implemented by this project is:

**BM25**

## BM25

The retriever builds a `BM25Okapi` index over the processed chunks saved in `data/processed/processed_chunks.json`. At query time, the user question is tokenized with the same tokenizer as the indexed chunks, scored against every chunk, and the top `k` source locations are returned.

The BM25 tokenizer is designed for both documentation and source code:

* text is Unicode-normalized and lowercased;
* word tokens are extracted with a regex;
* code identifiers are split into searchable words, so names like `openai_chat_completion` and `HTTPServerConfig` become smaller matching terms;
* common stopwords and very short low-signal tokens are removed;
* when NLTK is available, English stemming is applied so related forms like `configure`, `configured`, and `configuration` are easier to match.

BM25 assigns a relevance score to each indexed chunk based on term frequency, document frequency, and document length. The implementation also applies a small ranking boost to Markdown files under `/docs/`, because documentation chunks are often the best answer sources for docs-oriented questions even when examples contain the same API names.

The chunks are ranked by their retrieval score, and the highest-ranked `k` results are returned.

## Embedding Retrieval

The project also supports semantic retrieval as an optional bonus path. When indexing is run with embeddings enabled, each chunk is embedded with `all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection under `chromadb/`.

At query time, the question is embedded with the same model and ChromaDB returns the nearest chunks by vector similarity. This helps with questions that use different wording from the source text, where exact lexical matching alone may miss useful context.

The semantic result still returns the same source fields required by the subject:

* `file_path`
* `first_character_index`
* `last_character_index`

The stored metadata also keeps the chunk content so answer generation can use the retrieved text as context.

## Hybrid Retrieval

Hybrid retrieval combines the BM25 and embedding retrievers into one ranking. The implementation runs both retrievers, normalizes their scores separately, then computes a weighted score for each unique source chunk:

```text
hybrid_score = bm25_factor * normalized_bm25_score
             + embedding_factor * normalized_embedding_score
```

The default weights favor semantic retrieval while still keeping lexical matches useful:

```text
bm25_factor = 0.3
embedding_factor = 0.7
```

This approach keeps exact identifier matches from BM25, while also allowing semantically related chunks from the embedding index to appear in the final top `k` results.

### Ranking

For a query `q`, the retriever:

1. Tokenizes the query.
2. Searches the lexical index.
3. Calculates a relevance score for indexed chunks.
4. Sorts chunks by descending score.
5. Returns the top `k` source locations.

---

# Data Models

The project uses Pydantic models to validate data exchanged between pipeline stages.

The main models are:

### `MinimalSource`

Represents a retrieved source location.

```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
```

### `UnansweredQuestion`

Represents a question before answer generation.

```python
class UnansweredQuestion(BaseModel):
    question_id: str
    question: str
```

### `AnsweredQuestion`

Represents a question together with its answer and reference sources.

```python
class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str
```

### Search results

Search operations return the equivalent of:

```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]
```

The complete dataset-level search output contains:

```python
class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
```

Answer generation extends this structure with the generated answer.

---

# Project Structure

The repository follows the structure required by the project:

```text
.
├── .flake8
├── .gitignore
├── Makefile
├── README.md
├── mypy.ini
├── pyproject.toml
├── uv.lock
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── bm25_tokenizer.py
│   ├── chunker.py
│   ├── cli.py
│   ├── embedder.py
│   ├── evaluator.py
│   ├── file_manager.py
│   ├── index.html
│   ├── indexer.py
│   ├── llm_model.py
│   ├── local_api.py
│   ├── models.py
│   ├── retrieval.py
│   ├── tools.py
│   └── vectordb.py
└── data/
    ├── raw/
    │   └── .gitkeep
    ├── processed/
    │   └── .gitkeep
    ├── datasets/
    │   └── .gitkeep
    └── output/
        └── .gitkeep
```

The evaluator or the user provides the actual corpus and datasets at runtime:

```text
data/raw/vllm-0.10.1/
data/datasets/UnansweredQuestions/
data/datasets/AnsweredQuestions/
```

The commands generate runtime artifacts such as:

```text
data/processed/processed_chunks.json
data/processed/file_hashes.json
data/output/search_results/<DatasetScope>/
data/output/search_results_and_answer/<DatasetScope>/
chromadb/
```

Generated indexes, model weights, large datasets, vector databases, and generated outputs should not be committed to the repository.

---

# Installation

This project uses **uv** as its project and package manager.

Install the project dependencies with:

```bash
make install
```

Or directly with:

```bash
uv sync
```

It is recommended to use a virtual environment provided by `uv` for dependency isolation.

---

# Instructions

## Makefile

The project provides these Makefile commands:

```bash
make install
make run
make debug
make clean
make fclean
make lint
make lint-strict
make start-local-api
make index
make search-public
make search-private
make moulinette-docs-public
make moulinette-docs-private
make moulinette-code-public
make moulinette-code-private
```

`make install` runs `uv sync`.

`make run` starts the Python Fire CLI with `uv run python -m src`.

`make debug` starts the module under `pdb` with `uv run python -m pdb src`.

`make clean` removes Python bytecode and `.mypy_cache/`.

`make fclean` also removes generated files under `data/output/search_results/` and `data/processed/`.

`make start-local-api` runs the local FastAPI server entry point with `uv run python -m src.local_api`.

The `lint` target runs:

```bash
uv run flake8 .
uv run mypy --warn-return-any --warn-unused-ignores --disallow-untyped-defs --check-untyped-defs
```

`make lint-strict` runs:

```bash
uv run flake8 .
uv run mypy . --strict
```

The dataset helper targets run the current public/private search commands and write into `data/output/search_results/`. The moulinette helper targets evaluate those files against the matching datasets under `data/datasets/AnsweredQuestions/`.

---

# CLI

All commands are executed through:

```bash
uv run python -m src <command>
```

The CLI is implemented with Python Fire in `src/cli.py`.

## Index the codebase

Build the index from `data/raw/`:

```bash
uv run python -m src index --max_chunk_size 2000
```

Available options:

```text
--data_dir <path>          default: data/raw
--max_chunk_size <int>     default: 2000
--force <bool>             rebuild the full index when true
--embed <bool>             also build/update the ChromaDB embedding index
```

The generated index is stored under:

```text
data/processed/
```

When `--embed True` is used, semantic vectors are also stored under:

```text
chromadb/
```

---

## Search a single question

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 10
```

The command uses BM25 retrieval and returns a `StudentSearchResults` JSON object containing one result.

Options:

```text
<query>        question string
--k <int>      number of sources to return
```

---

## Search a dataset

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

The resulting JSON file follows the required search-results structure.
The output filename is copied from the input dataset filename and written inside `--save_directory`.

---

## Generate an answer

To answer a single question:

```bash
uv run python -m src answer \
    "How to configure the OpenAI server?" \
    --k 10
```

The system retrieves relevant context and passes it to `Qwen/Qwen3-0.6B`.

Options:

```text
<query>        question string
--k <int>      number of sources to retrieve before generation
```

---

## Generate answers for a dataset

```bash
uv run python -m src answer_dataset \
    --student_search_results_path \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory \
    data/output/search_results_and_answer/UnansweredQuestions
```

This command expects a `StudentSearchResults` file produced by `search_dataset`. It generates a `StudentSearchResultsAndAnswer` JSON file with the same filename in `--save_directory`.

---

## Evaluate retrieval

For local development and iteration:

```bash
uv run python -m src evaluate \
    --student_search_results_path \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path \
    data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The official evaluation is performed by the provided evaluation executable during the defense. The project itself does not import or call the evaluation executable.

---

## Hybrid search

```bash
uv run python -m src hybrid_search \
    "How to configure the OpenAI server?" \
    --k 10 \
    --bm25_factor 0.3 \
    --embedding_factor 0.7
```

This command combines BM25 and ChromaDB embedding retrieval. It requires the embedding index to have been built first with `index --embed True`.

---

## Hybrid answer

```bash
uv run python -m src hybrid_answer \
    "How to configure the OpenAI server?" \
    --k 10 \
    --bm25_factor 0.3 \
    --embedding_factor 0.7
```

This command retrieves context with hybrid retrieval, then sends that context to `Qwen/Qwen3-0.6B` for answer generation.

---

# End-to-End Example

The mandatory BM25 workflow is:

### 1. Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

The equivalent Makefile helper is:

```bash
make index
```

### 2. Search the dataset

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

The current Makefile helper commands use `data/output/search_results/` directly:

```bash
make search-public
make search-private
```

### 3. Evaluate the retrieval results

```bash
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --max_context_length 2000
```

The Makefile also provides dataset-specific moulinette helpers:

```bash
make moulinette-docs-public
make moulinette-docs-private
make moulinette-code-public
make moulinette-code-private
```

### 4. Generate answers

```bash
uv run python -m src answer_dataset \
    --student_search_results_path \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory \
    data/output/search_results_and_answer/UnansweredQuestions
```

### Optional hybrid workflow

Build the lexical index and the embedding index:

```bash
uv run python -m src index --max_chunk_size 2000 --embed True
```

Run hybrid retrieval for one query:

```bash
uv run python -m src hybrid_search \
    "How to configure the OpenAI server?" \
    --k 10
```

Run hybrid retrieval plus answer generation:

```bash
uv run python -m src hybrid_answer \
    "How to configure the OpenAI server?" \
    --k 10
```

---

# Example Usage

Install dependencies:

```bash
make install
```

Build the BM25 index from `data/raw/`:

```bash
uv run python -m src index --max_chunk_size 2000
```

Search one question:

```bash
uv run python -m src search "How can I start the OpenAI-compatible server?" --k 10
```

Search a full dataset and save the results:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

Evaluate the saved retrieval results with the moulinette:

```bash
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --max_context_length 2000
```

Generate answers from saved search results:

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Optional hybrid retrieval example:

```bash
uv run python -m src index --max_chunk_size 2000 --embed True
uv run python -m src hybrid_search "How can I start the OpenAI-compatible server?" --k 10
```

---

# Evaluation

Retrieval quality is measured using **Recall@k**.

A retrieved source is considered correct when:

* it belongs to the correct file, and
* its character range overlaps the reference range sufficiently according to the evaluation metric.

The file path must match the corpus path exactly.

## Required performance

The project requirements specify:

| Metric                     |  Requirement |
| -------------------------- | -----------: |
| Indexing time              |  ≤ 5 minutes |
| Retrieval of 200 questions | ≤ 90 seconds |
| Docs Recall@5              |        ≥ 80% |
| Code Recall@5              |        ≥ 50% |

Recall@k is calculated as the share of correct sources retrieved within the top `k` results.

## Results

Measured with the provided `moulinette evaluate_student_search_results` command using the generated BM25 search results in `data/output/search_results/` and `--k 10 --max_context_length 2000`.

| Dataset                | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| ---------------------- | --------: | -------: | -------: | -------: | --------: |
| Docs public            |       100 |    62.0% |    78.0% |  **85.0%** |     90.0% |
| Code public            |        99 |    36.4% |    61.6% |  **64.6%** |     71.7% |

The measured results pass the required Recall@5 thresholds: documentation is above 80%, and code is above 50%.

### Performance measurements

The moulinette recall command does not report indexing or retrieval timing. Timing should be measured separately when needed.

---

# Design Decisions

## Why a single sliding-window chunker for every file type?

Rather than maintaining separate AST-aware and heading-aware chunkers, the project uses one sliding-window strategy for Python, Markdown, and text files alike.

This keeps the chunking code path simple and consistent, makes character-offset bookkeeping straightforward (every chunk is a direct substring slice), and avoids edge cases around malformed/unparseable source files. The overlap between consecutive windows mitigates the main downside — content near a chunk boundary getting split — by ensuring it still appears whole in at least one chunk.

## Why use a configurable chunk size?

The project requires the chunk size to be configurable through the CLI.

The default value is 2000 characters, while smaller chunks can be used to experiment with retrieval quality.

Changing the chunk size can affect Recall@k because it changes the amount and granularity of information contained in each retrieved source.

## Why BM25?

> Keep this explanation only if BM25 is your actual implementation.

BM25 is a lexical retrieval method that is particularly useful for source-code search because exact identifiers, function names, class names, configuration names, and technical terms can be highly informative.

---

# Challenges Faced

## Challenge 1 — Chunking source code

> Describe the actual problem you encountered.

For example:

* preserving meaningful Python structures
* keeping character offsets correct
* preventing chunks from exceeding 2000 characters

### Solution

> Describe your actual implementation and solution here.

---

## Challenge 2 — Character offsets

The evaluator compares the retrieved character ranges with the reference ranges.

Therefore, the index must preserve the original source text positions.

### Solution

> Describe how your implementation calculates and stores:
>
> * `first_character_index`
> * `last_character_index`

---

## Challenge 3 — Retrieval quality

> Describe the actual retrieval problems you encountered.

Possible topics to document if they genuinely occurred:

* questions using different terminology from the source
* exact identifiers vs natural-language questions
* chunk size effects
* tokenizer choices
* ranking quality

### Solution

> Explain the experiments you actually performed and their measured impact on Recall@k.

---

# Error Handling

The CLI handles invalid and degenerate inputs without producing an unhandled traceback.

Examples include:

* empty queries
* nonsensical queries
* `k=0`
* missing files
* malformed JSON
* invalid CLI arguments

Long-running operations provide progress information using `tqdm`.

---

# Bonus Features

## Semantic Embeddings

A semantic vector index is available alongside the lexical BM25 index. When `index --embed True` is used, chunks are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB under `chromadb/`.

**Status:** Implemented.

Main files: `src/embedder.py`, `src/vectordb.py`, `src/indexer.py`, `src/retrieval.py`.

Usage:

```bash
uv run python -m src index --max_chunk_size 2000 --embed True
```

## Hybrid Retrieval

Hybrid retrieval combines BM25 and ChromaDB embedding retrieval into a single ranked list. Scores from both retrievers are normalized, then merged with configurable weights.

**Status:** Implemented.

Main file: `src/retrieval.py`.

Usage:

```bash
uv run python -m src hybrid_search "How to configure the OpenAI server?" --k 10
uv run python -m src hybrid_answer "How to configure the OpenAI server?" --k 10
```

## Incremental Indexing

Incremental indexing tracks file hashes in `data/processed/file_hashes.json`. If processed chunks already exist, unchanged files are reused, modified files are re-chunked, and deleted files are removed from the saved chunk list. When embedding indexing is enabled, stale vector entries are also deleted and modified chunks are added back.

**Status:** Implemented.

Main files: `src/file_manager.py`, `src/indexer.py`.

## Caching

The project persists processed chunks in `data/processed/processed_chunks.json` and persists vector data in ChromaDB, so repeated runs can reuse existing indexed data instead of rebuilding everything. Query-result caching is not implemented as a separate cache layer.

**Status:** Partially implemented through persistent indexes.

Main files: `src/indexer.py`, `src/vectordb.py`.

## Local HTTP API

The local API exposes indexing, search, and answer generation over FastAPI. Search and answer endpoints can use either BM25 or hybrid retrieval through the `hybrid` parameter.

**Status:** Implemented.

Main file: `src/local_api.py`.

Usage:

```bash
make start-local-api
```

---

# Resources

The following resources were used to understand the technologies involved in this project:

* [YouTube | RAG Crash Course for Beginners](https://youtu.be/swvzKSOEluc?si=bwZ4MoxUNeKdUhPN)
* [YouTube | What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M)
* [YouTube | RAG Explained For Beginners](https://www.youtube.com/watch?v=_HQ2H_0Ayy0&t=418s)
* [YouTube | Word Embeddings: TF-IDF](https://www.youtube.com/watch?v=x1u5TotQ0G0&t=101s)
* [YouTube | How does a Vector Database work?](https://www.youtube.com/watch?v=VVNYQKDLY5s)
* [YouTube | What is a Vector Database? Powering Semantic Search & AI Applications](https://www.youtube.com/watch?v=gl1r1XV0SLw)
* [YouTube | What are Word Embeddings?](https://youtu.be/wgfSDrqYMJ4?si=QGQFWL9d7fyfZ-ZT)

## AI Usage

AI tools were used as development and learning assistants.

Examples of tasks for which AI assistance was used:

* understanding the concepts behind Retrieval-Augmented Generation
* understanding BM25 and lexical retrieval
* exploring chunking strategies
* debugging implementation issues
* understanding Python libraries and APIs
* reviewing code and identifying possible problems
* helping design the project architecture
* generating ideas for tests and edge cases

AI-generated material was reviewed, tested, and adapted before being integrated into the project.

The project author remains responsible for understanding and validating the final implementation.
