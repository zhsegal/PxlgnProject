from torch.distributions import NegativeBinomial, Normal, Beta
from enum import Enum, auto


class D(Enum):
    '''Supported distributions'''
    Normal = auto()
    Beta = auto()

    def to_torch(self):
        if self == D.Normal:
            return Normal
        elif self == D.Beta:
            return Beta

    def __str__(self):
        if self == D.Normal:
            return 'Normal'
        elif self == D.Beta:
            return 'Beta'


class AggMethod(Enum):
    '''
    Chosen method for aggregating modalities
    '''
    SHARED_ENCODER = auto()             # Shared encoder
    AOE_FIXED_WEIGHTS = auto()          # Average of experts, fixed weights
    AOE_GLOBAL_WEIGHTS = auto()         # Average of experts, learned global weights
    AOE_PER_CELL_WEIGHTS = auto()           # Average of experts, learned per-cell weights

    def __str__(self):
        if self == AggMethod.SHARED_ENCODER:
            return 'shared_encoder'
        elif self == AggMethod.AOE_FIXED_WEIGHTS:
            return 'aoe_fixed_weights'
        elif self == AggMethod.AOE_GLOBAL_WEIGHTS:
            return 'aoe_global_weights'
        elif self == AggMethod.AOE_PER_CELL_WEIGHTS:
            return 'aoe_per_cell_weights'