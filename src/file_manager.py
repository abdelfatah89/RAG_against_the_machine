from pathlib import Path
import hashlib
import json
from typing import Dict, List


SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}
HASHES_PATH = Path("data/processed/file_hashes.json")


class FileManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def get_files(self) -> List[str]:
        root = Path(self.data_dir)
        files = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(path))
        return files

    def get_current_hashes(self) -> Dict[str, str]:
        """Hash every currently-present supported file. Does NOT persist."""
        current_hashes: Dict[str, str] = {}
        for path in self.get_files():
            file_path = Path(path)
            current_hashes[str(file_path)] = self.get_file_hash(file_path)
        return current_hashes

    def save_hashes(self, hashes: Dict[str, str]) -> None:
        """Persist the given hash map as the new baseline for next run."""
        HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HASHES_PATH, "w") as f:
            json.dump(hashes, f, indent=4)

    def get_old_hashes(self) -> Dict[str, str]:
        if HASHES_PATH.is_file():
            with open(HASHES_PATH, "r") as f:
                return json.load(f)
        return {}

    def get_modified_files(self) -> List[str]:
        """Files that are new or whose content changed since the last
        persisted baseline. Does not include deleted files -- see
        get_deleted_files()."""
        old_hashes = self.get_old_hashes()
        current_hashes = self.get_current_hashes()

        if not old_hashes:
            return list(current_hashes.keys())

        return [
            path for path, current_hash in current_hashes.items()
            if old_hashes.get(path) != current_hash
        ]

    def get_deleted_files(self) -> List[str]:
        """Files present in the last baseline but no longer on disk."""
        old_hashes = self.get_old_hashes()
        current_files = set(self.get_files())
        return [path for path in old_hashes if path not in current_files]

    def modified_files_exist(self) -> bool:
        if not self.get_old_hashes():
            return True
        return bool(self.get_modified_files() or self.get_deleted_files())
