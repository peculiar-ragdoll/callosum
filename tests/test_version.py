from importlib.metadata import version

import callo


def test_dunder_version_matches_the_installed_metadata():
    assert callo.__version__ == version("callo") == "0.3.1"
