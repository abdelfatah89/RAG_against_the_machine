from typing import List
import json
from .models import MinimalSource


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


def save_processed_data(data: List[MinimalSource], file_path: str) -> None:
    output_data = [item.model_dump() for item in data]
    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=4)
