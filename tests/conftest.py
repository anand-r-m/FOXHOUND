import sys
from pathlib import Path

import pytest

# Package root: FOXHOUND/FOXHOUND (models.py, env.py, server/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _clear_global_env():
    """server.app keeps a module-level env; isolate tests."""
    import server.app as app_module

    app_module.env = None
    yield
    app_module.env = None
