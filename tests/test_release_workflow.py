"""The trusted publisher is registered against exact names; a drift here breaks the publish silently."""
import re
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


@pytest.fixture(autouse=True)
def _workflow_present():
    if not WF.is_file():
        pytest.skip(f"{WF} is absent (an sdist ships no .github/)")


def test_release_workflow_matches_the_trusted_publisher_contract():
    text = WF.read_text()
    assert re.search(r"tags:\s*\n\s*- ['\"]?v\*", text)                       # trigger: tag v*
    assert "environment: pypi" in text and "pypa/gh-action-pypi-publish" in text
    assert "uv build" in text and "uv version --short" in text and "${GITHUB_REF_NAME#v}" in text
    publish = text.split("publish:", 1)[1]
    build = text.split("publish:", 1)[0]
    assert "id-token: write" in publish and "contents: read" in publish    # the OIDC grant is on the publish job
    assert "id-token" not in build                                         # and nowhere near the build job
    assert "needs: build" in publish and "actions/checkout" not in publish   # no checkout in the publish job
    assert "actions/download-artifact" in publish and "actions/upload-artifact" in text.split("publish:", 1)[0]


def test_the_tag_publish_runs_the_bias_set_release_gate_before_building():
    """A tag pushed while BUILTIN_SET_REVISION is still the placeholder must not ship (the bias-set fetch)."""
    build = WF.read_text().split("publish:", 1)[0]
    assert "python -m bench.release_check" in build
    assert build.index("uv version --short") < build.index("bench.release_check") < build.index("run: uv build")
