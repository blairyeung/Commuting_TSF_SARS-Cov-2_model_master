import math
import numpy as np
import Dependency as Dependency
import ModelData

class Model:

    dependency = None
    _model_data = None

    def __init__(self, forecast_days=1000):
        self._initialize_dependencies()
        self._model_data = ModelData(forecast_days, self.dependency)
        # TODO: Initialize
        return

    def run_one_cycle(self):
        """
        Firstly, we convert the number of newly infected cases to active cases
        """
        self._model_data
        # TODO: Run one cycle
        return

    def _get_new_cases(self, cases, contact_type=(0, 0)):
        return np.dot(self._model_data.dependency.matrix_by_class[contact_type[0]][contact_type[1]], cases)

    def _initialize_dependencies(self):
        self.dependency = Dependency.Dependency()


if __name__ == '__main__':
    m = Model(forecast_days=100)
