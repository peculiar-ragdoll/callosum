"""brand/tests run explicitly (they are outside the package suite):

    uv run --with fonttools --with uharfbuzz --with pillow pytest brand/tests -q

They import the figure modules from brand/ and callo from src/."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BRAND = os.path.dirname(HERE)
for p in (HERE, BRAND, os.path.join(os.path.dirname(BRAND), "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
