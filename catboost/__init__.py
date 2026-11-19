"""Coursework CatBoost experiments alongside the installed CatBoost library.

The folder name is required by the coursework layout. Extending the package
search path makes the installed library's ``catboost.core`` available while
``catboost.pipeline`` and ``catboost.learning_curve`` stay in this repository.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .core import CatBoostClassifier  # noqa: E402
from .version import VERSION as __version__  # noqa: E402

__all__ = ["CatBoostClassifier"]
