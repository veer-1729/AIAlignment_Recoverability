"""Every script must import.

Added after a broken import reached a push: patching an import line spliced
`utility` onto the wrong module, and the 107-test suite passed anyway because no
test imports the scripts. The scripts are what runs on the GPU box, hours from
here, and an ImportError there costs a round trip and whatever the run was
holding.
"""

import glob
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")))


def test_there_are_scripts_to_check():
    assert SCRIPTS, "glob found nothing -- the test would pass vacuously"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: os.path.basename(p))
def test_script_imports(path):
    spec = importlib.util.spec_from_file_location(
        "script_" + os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass                      # argparse on import is fine
    except ImportError as exc:
        # Optional heavy deps (torch, transformers, sklearn) are not installed
        # everywhere; a missing third-party module is not this test's business.
        if any(k in str(exc) for k in ("torch", "transformers", "sklearn", "matplotlib",
                                       "alfworld", "textworld")):
            pytest.skip("optional dependency: {}".format(exc))
        raise
