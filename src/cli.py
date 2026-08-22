class CLI:
    def index(
        self,
        max_chunk_size: int
            ):
        pass

    def search(
        self,
        query: str, k: int
            ):
        pass

    def search_dataset(
        self,
        dataset_path: str, k: int,
        save_directory: str
            ):
        pass

    def answer(
        self,
        query: str, k: int
            ):
        pass

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
            ):
        pass

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
            ):
        pass
