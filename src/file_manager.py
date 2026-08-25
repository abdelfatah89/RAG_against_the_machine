from pathlib import Path
import hashlib
import json
from typing import List


SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


class FileManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.generate_file_hashes()

    def get_file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def get_files(self) -> List[str]:
        data_dir = self.data_dir
        root = Path(data_dir)
        files = []
        for path in root.rglob("*"):
            if (path.is_file() and
                    path.suffix.lower() not in SUPPORTED_EXTENSIONS):
                continue
            if (path.is_file() and
                (path.suffix.lower() == ".py" or
                 path.suffix.lower() == ".md" or
                 path.suffix.lower() == ".txt")):
                files.append(str(path))
        return files

    def generate_file_hashes(self) -> None:
        files = self.get_files()
        file_hashes = {}
        for path in files:
            file_path = Path(path)
            if file_path.is_file():
                file_hashes[str(file_path)] = self.get_file_hash(file_path)
        with open("data/processed/file_hashes.json", "w") as f:
            json.dump(file_hashes, f, indent=4)

    def get_old_hashes(self) -> dict[str, str]:
        old_hashes_path = Path("data/processed/file_hashes.json")
        if old_hashes_path.is_file():
            with open(old_hashes_path, "r") as f:
                return json.load(f)
        return {}

    def get_modified_files(self) -> List[str]:
        modified = []
        old_hashes = self.get_old_hashes()
        if not old_hashes:
            return self.get_files()
        files = self.get_files()

        for path in files:
            file_path = Path(path)
            if not file_path.is_file():
                continue

            key = str(file_path)
            current_hash = self.get_file_hash(file_path)

            # New file OR content changed
            if old_hashes.get(key) != current_hash:
                modified.append(str(file_path))

        return modified
