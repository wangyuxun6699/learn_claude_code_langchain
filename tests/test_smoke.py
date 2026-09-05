"""逐章可运行性冒烟测试：全部 Python 源文件必须通过 py_compile。"""
import pathlib
import py_compile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDE = {
    "venv", ".git", "__pycache__", ".pytest_cache",
    ".runtime", ".tasks", ".memory", ".task_outputs", ".ref-repo",
}


def _source_files():
    for p in ROOT.rglob("*.py"):
        if EXCLUDE & set(p.parts):
            continue
        yield p


def test_all_source_files_compile():
    files = list(_source_files())
    assert files, "no source files found"
    for p in files:
        py_compile.compile(str(p), doraise=True)
