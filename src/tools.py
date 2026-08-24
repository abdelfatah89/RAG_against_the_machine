def get_keywords_args(file_path: str,
                      first_character_index: int,
                      last_character_index: int,
                      content: str = "",
                      file_type: str = "",
                      metadata: dict = {},
                      score: float = 0.0
                      ) -> dict:
    return {
        "file_path": file_path,
        "first_character_index": first_character_index,
        "last_character_index": last_character_index,
        "content": content,
        "file_type": file_type,
        "metadata": metadata,
        "score": score
    }
