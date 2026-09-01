from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .models import StudentSearchResults, StudentSearchResultsAndAnswer
from .file_manager import FileManager
from .cli import CLI


DATA_DIR = "data/raw"
CHUNKS_CACHE_FILE = "data/processed/processed_chunks.json"


class LocalAPI:
    def __init__(self) -> None:
        self.app = FastAPI()
        self.file_manager = FileManager(DATA_DIR)
        self.cli: CLI | None = None
        self._setup_cors()
        self._setup_routes()

    def run(self) -> None:
        uvicorn.run(self.app)

    def _cli(self) -> CLI:
        if self.cli is None:
            self.cli = CLI()
        return self.cli

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            )

    def _setup_routes(self) -> None:
        @self.app.get("/")
        def health() -> Dict[str, str]:
            if self.file_manager.modified_files_exist():
                return {"message": "Files have been modified."}
            if not Path(CHUNKS_CACHE_FILE).is_file():
                return {"message": "No processed chunks found. "
                                   "Please run the index command."}
            return {"message": "Chunks are ready and up-to-date."}

        @self.app.get("/index")
        def index(data_dir: str = DATA_DIR,
                  max_chunk_size: int = 2000,
                  force: bool = True, embed: bool = False
                  ) -> Dict[str, str]:
            if not self.file_manager.get_files():
                return {"message": "No files found to index."}

            self.cli = self._cli()
            self.cli.index(data_dir=data_dir,
                           max_chunk_size=max_chunk_size,
                           force=force,
                           embed=embed)
            return {"message": "Indexing completed successfully."}

        @self.app.post("/search")
        def search(query: str, k: int = 10,
                   hybrid: bool = False) -> Dict[str, Any]:
            self.cli = self._cli()
            if hybrid:
                results = self.cli.hybrid_search(query=query, k=k, p=False)
            else:
                results = self.cli.search(query=query, k=k, p=False)

            if results is None:
                return {"message": "No search results found."}
            ss_results = StudentSearchResults(search_results=[results], k=k)
            dict_results = self._cleanup(ss_results)
            return dict_results

        @self.app.post("/answer")
        def answer(query: str, k: int = 10,
                   hybrid: bool = False) -> Dict[str, Any]:
            self.cli = self._cli()
            if hybrid:
                results = self.cli.hybrid_answer(query=query, k=k, p=False)
            else:
                results = self.cli.answer(query=query, k=k, p=False)
            if results is None:
                return {"message": "No answer found."}
            dict_results = self._cleanup(results)
            return dict_results

    def _cleanup(self,
                 results: StudentSearchResults | StudentSearchResultsAndAnswer
                 ) -> Dict[str, Any]:
        dict_results = results.model_dump()
        for result in dict_results["search_results"]:
            for source in result["retrieved_sources"]:
                del source["content"]
                del source["score"]
                del source["metadata"]
                del source["file_type"]

        return dict_results


if __name__ == "__main__":
    api = LocalAPI()
    api.run()
