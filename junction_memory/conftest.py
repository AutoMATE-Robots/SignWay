"""Makes `junction_memory/` importable for tests living in tests/.

pytest's default (prepend) import mode inserts each test file's basedir on
sys.path -- that is tests/, not the package root -- so `import memory_runtime`
would fail. A conftest.py at the package root gets its own directory inserted,
which fixes it regardless of the directory pytest is invoked from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
