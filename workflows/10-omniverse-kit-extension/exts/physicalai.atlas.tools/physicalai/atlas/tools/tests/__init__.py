"""Kit-side tests, run by Kit's test runner (``./repo.sh test`` or the Extension Manager's test button), not pytest.

The CPU tests for this repo live in ``10-omniverse-kit-extension/tests/``.
"""

import contextlib

with contextlib.suppress(ImportError):  # outside Kit, omni.kit.test is unavailable
    from .test_window import *  # noqa: F403
