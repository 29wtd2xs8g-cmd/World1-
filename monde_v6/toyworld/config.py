"""World parameters. Every value comes from the world specification (docs/world_spec.md, section 5).

This file is frozen before the first experiment (checklist 0.2). `D` was measured by
`scripts/calibrate.py` (D = 2 * C1) and copied here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    # Time
    dt: float = 0.1              # s, duration of one step
    T: int = 500                 # steps per episode (episode truncated at t = T - 1)

    # Geometry (m)
    arena: float = 10.0          # square [0, arena]^2
    wall_x: float = 5.0          # vertical wall
    openings: tuple = ((5.0, 0.5), (1.5, 0.5))   # (y of centre, half-width): A direct, B longer
    x0: tuple = (1.0, 1.0)       # start
    r0: float = 1.5              # radius around the start that contains no zone
    goal: tuple = (9.0, 9.0)
    r_goal: float = 0.5

    # Dynamics
    a_max: float = 1.0           # m/s^2
    gamma: float = 0.5           # 1/m, quadratic drag
    sigma: float = 0.1           # m s^(-3/2), process noise
    sigma_o: float = 0.02        # m and m/s, observation noise
    sigma_n: float = 0.1         # damage units/s, noise of the nociceptive sensor

    # Zones (K = 3): Z1 in opening A, Z2 in opening B, Z3 on the approach route (phantom zone)
    z: tuple = ((5.0, 5.0), (5.0, 1.5), (3.0, 3.0))
    rho: tuple = (1.0, 1.0, 1.0)
    theta0: tuple = (1.0, 0.3, 0.0)              # nominal damage rate, damage units/s
    e: tuple = ((-1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))   # direction of the hidden force, per zone

    # Failure threshold (damage units) = 2 * C1', with C1' = 1.4908 measured by scripts/calibrate.py:
    # baseline agent (docs/agents.md), route A, no drift, 200 episodes on the calibration seeds, 95% CI +-0.011.
    # Frozen before any trigger experiment (checklist 0.2). It replaces the first calibration
    # (5.005 = 2 * 2.5027, scripted crosser), which left the competent baseline a margin of 3.3 times its damage.
    D: float = 2.981
