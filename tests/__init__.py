import sys
import pytest


def run() -> None:
    sys.exit(pytest.main(["tests/"]))


def coverage() -> None:
    sys.exit(pytest.main(["tests/", "--cov=utils", "--cov=scripts", "--cov-report=term-missing"]))
