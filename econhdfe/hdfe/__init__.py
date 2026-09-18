"""High-dimensional fixed-effect compilation, absorption, rank, and solvers."""
from .absorber import HDFEAbsorber, HDFEConvergenceError
from .two_way import TwoWayFEAbsorber, TwoWaySolveInfo
from .group_individual import GroupIndividualAbsorber, GroupIndividualInfo
from .plan import FEPlan
from .dof import DofInfo
from .specs import FixedEffect, Interaction, fe, interaction
from .rank import (
    CategoricalRankInfo,
    RankBackend,
    available_rank_backends,
    categorical_rank,
    categorical_prefix_ranks,
)

from .numerical_core import NumericalCorePlan, build_numerical_core
