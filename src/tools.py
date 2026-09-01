from functools import wraps
from typing import Any, Callable, Dict, List, TypeVar, cast
import json
from .models import MinimalSource


F = TypeVar("F", bound=Callable[..., Any])


def _safe(func: F) -> F:
    """Print clean CLI errors instead of exposing Python tracebacks.

    Internal calls that pass ``p=False`` are allowed to raise normally so the
    outer user-facing command can handle and report the original failure.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if kwargs.get("p") is False:
                raise
            print(f"Error: {exc}")
            return None

    return cast(F, wrapper)


def get_keywords_args(file_path: str,
                      first_character_index: int,
                      last_character_index: int,
                      content: str = "",
                      file_type: str = "",
                      metadata: Dict[str, Any] = {},
                      score: float = 0.0
                      ) -> Dict[str, Any]:
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
