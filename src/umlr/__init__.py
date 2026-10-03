"""Umbilic Multinomial Logistic Regression (UMLR).

Hyperplane (H) and sphere (S) class coordinates with first-order or
signed-square links, plus the prototype and angular-margin comparators used
in the paper.
"""

from .classifier import BoundaryClassifier
from .heads import AngularMarginHead, HyperplaneHead, PrototypeDistanceHead
from .umlr_head import MixedHyperplaneSphereHead, SphereUMLRHead

__all__ = [
    "AngularMarginHead",
    "BoundaryClassifier",
    "HyperplaneHead",
    "MixedHyperplaneSphereHead",
    "PrototypeDistanceHead",
    "SphereUMLRHead",
]
__version__ = "1.0.0"
