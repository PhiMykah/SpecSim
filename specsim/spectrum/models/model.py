
from .exponential import sim_exponential_1D
from .gaussian import sim_gaussian_1D
from .composite import sim_composite_1D
from enum import Enum

class SimulationModel(Enum):
    EXP = 0, sim_exponential_1D, 'exp'
    GAUSS = 1, sim_gaussian_1D, 'gauss'
    COMP = 2, sim_composite_1D, 'comp'

def get_simulation_model(model : str) -> SimulationModel:
    match model:
        case 'exp':
            return SimulationModel.EXP
        case 'gauss':
            return SimulationModel.GAUSS
        case 'comp':
            return SimulationModel.COMP
        case _:
            return SimulationModel.EXP