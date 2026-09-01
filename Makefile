.PHONY: test lint compile all

compile:  ## py_compile 全部章节源码
	python -m compileall -q -x 'venv|__pycache__|\.git|\.runtime|\.tasks|\.memory' .

lint:  ## 仅 lint harness 与 tests（教学章节保留各自风格）
	ruff check harness tests

test:  ## 运行冒烟 + 单元测试
	pytest -q

all: lint test compile
