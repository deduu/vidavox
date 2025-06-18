# retrieval_system/__init__.py
"""Top-level package exports for the modular retrieval system."""

from importlib import import_module

# Expose the public Engine class at package root for convenience.
# The second argument to import_module should be the *current* package,
# which allows the relative import to be resolved correctly from its context.
RetrievalEngine = import_module(".core.engine", __package__).RetrievalEngine

__all__ = ["RetrieralEngine"]