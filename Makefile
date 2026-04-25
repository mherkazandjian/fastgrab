# Help system: any target prefixed with a line starting with `#@help: ...`
# is picked up by `make help` and shown next to its name.
#
# Default: targets run on the host (regular `pip`, `poetry`, `pytest`).
# Pass `docker=1` to route the same targets through the project's
# docker compose dev container instead. Examples:
#
#   make test              # runs pytest on the host
#   make test docker=1     # runs the test suite inside docker
#
# Contributors usually want `docker=1`; a regular user installing
# fastgrab does not need docker at all.

.DEFAULT_GOAL := help

ifeq ($(docker),1)
DC := docker compose run --rm --entrypoint bash test -c
BUILD_CMD := $(DC) 'poetry build'
TEST_CMD := docker compose run --rm test
INSTALL_CMD := $(DC) 'pip install .'
BENCH_CMD := docker compose run --rm benchmark
LOCK_CMD := $(DC) 'poetry lock'
else
BUILD_CMD := poetry build
TEST_CMD := pytest tests
INSTALL_CMD := pip install .
BENCH_CMD := python examples/benchmark.py
LOCK_CMD := poetry lock
endif

#@help: build the wheel + sdist into dist/
build:
	$(BUILD_CMD)

#@help: run the unit + integration test suite
test:
	$(TEST_CMD)

#@help: install fastgrab into the active python environment
install:
	$(INSTALL_CMD)

#@help: bring up the dev VNC desktop on host port 5901 (requires docker=1)
dev:
ifeq ($(docker),1)
	docker compose up -d dev
	@echo "VNC ready on localhost:5901"
else
	@echo "dev requires docker=1: make dev docker=1"
	@exit 1
endif

#@help: run the capture-fps benchmark across resolutions
benchmark:
	$(BENCH_CMD)

#@help: regenerate poetry.lock from pyproject.toml
lock:
	$(LOCK_CMD)

#@help: remove build artifacts (also docker compose volumes when docker=1)
clean:
ifeq ($(docker),1)
	-docker run --rm -v "$(CURDIR)":/work -w /work alpine sh -c 'rm -rf dist build *.egg-info'
	-docker compose down -v
else
	rm -rf dist build *.egg-info
endif

define AWK_SCRIPT
BEGIN {
    help = "";
}
/^#@help:/ {
    sub(/^#@help:[[:space:]]*/, "", $$0);
    if (help != "") {
        help = help "\n                                 " $$0;
    } else {
        help = $$0;
    }
}
/^[a-zA-Z0-9_.@%\/\-]+:/ {
    if ($$0 ~ /^\./) next;
    target = $$0;
    sub(/:.*$$/, "", target);
    printf "  %-30s %s\n", target, help;
    help = "";
}
endef
export AWK_SCRIPT

#@help: show this help message (pass docker=1 to route targets through docker)
help:
	@echo "Available targets:"
	@awk "$$AWK_SCRIPT" $(MAKEFILE_LIST)
	@echo
	@echo "Pass docker=1 to run targets in the dev container, e.g.:"
	@echo "  make test docker=1"

.PHONY: help build test install dev benchmark lock clean
