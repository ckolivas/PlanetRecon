"""Compatibility import for the qualified application implementation."""
import sys
from planetrecon.pipeline import cfa_interpolation as _implementation

sys.modules[__name__] = _implementation
