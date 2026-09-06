"""PlanetRecon Gate-1 simulator, estimators, Q2 MFBD, and capture reconstruction."""

from planetrecon import constants as C

__version__ = "0.1.0"
try:
    from planetrecon._build_version import VERSION as __version__
except ImportError:
    pass
__roadmap_revision__ = C.ROADMAP_REVISION
__estimator_operator_version__ = C.ESTIMATOR_OPERATOR_VERSION
__simulator_operator_version__ = C.SIMULATOR_OPERATOR_VERSION
