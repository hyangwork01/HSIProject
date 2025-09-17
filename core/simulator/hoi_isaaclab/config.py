from dataclasses import dataclass
from core.simulator.base_simulator.config import ConfigBuilder, SimParams, SimulatorConfig


@dataclass
class IsaacLabPhysXParams(ConfigBuilder):
    """PhysX physics engine parameters."""
    num_threads: int = 4
    solver_type: int = 1  # 0: pgs, 1: tgs
    num_position_iterations: int = 4
    num_velocity_iterations: int = 0
    contact_offset: float = 0.02
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 10.0
    default_buffer_size_multiplier: float = 10.0

    """PhysX physics engine parameters."""
    gpu_found_lost_pairs_capacity: int = 2**21
    gpu_max_rigid_contact_count: int = 2**23
    gpu_found_lost_aggregate_pairs_capacity: int = 2**25


@dataclass
class IsaacLabSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""
    physx: IsaacLabPhysXParams = IsaacLabPhysXParams()


@dataclass
class IsaacLabSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacLab simulator."""
    sim: IsaacLabSimParams  # Override sim type
    def __post_init__(self):
        self.w_last = False  # IsaacLab uses wxyz quaternions
