"""Shared loader for chapter tests that must not read the developer's .env."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_lesson(workdir: Path, lesson_path: Path):
    fake_dotenv = ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda override=True: None

    previous_dotenv = sys.modules.get("dotenv")
    previous_cwd = Path.cwd()
    previous_model = os.environ.get("MODEL_ID")
    module_name = f"lesson_{lesson_path.parent.name}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, lesson_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    sys.modules["dotenv"] = fake_dotenv
    sys.modules[module_name] = module
    try:
        os.chdir(workdir)
        os.environ["MODEL_ID"] = "test-model"
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model
        if previous_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = previous_dotenv
        sys.modules.pop(module_name, None)
