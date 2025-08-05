import torch
import numpy as np

from core.envs.base_env.components.base_component import BaseComponent

class ObjectObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        # TODO