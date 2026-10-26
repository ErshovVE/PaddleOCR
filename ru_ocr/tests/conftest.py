import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_tool(name, folder="tools"):
    """Import ru_ocr/<folder>/<name>.py as a module (the folders are not packages)."""
    path = REPO_ROOT / "ru_ocr" / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
