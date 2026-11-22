"""Reference agents used to test the world (they are not research agents).

`ScriptedCrosser`: follows waypoints with a PD controller on the noisy observation.
It is the "frozen" agent of the section-1.3 tests: it adapts to nothing.
`retreat_action`: damage-free retreat (cancel the velocity, then go back towards x0).
"""
import numpy as np

ROUTES = {
    "A": [(4.2, 5.0), (5.8, 5.0), (9.0, 9.0)],
    "B": [(4.2, 1.5), (5.8, 1.5), (9.0, 9.0)],
}


class ScriptedCrosser:
    def __init__(self, route="A", kp=2.0, kd=3.0, reach=0.3):
        self.wps = [np.array(p, float) for p in ROUTES[route]]
        self.kp, self.kd, self.reach = kp, kd, reach
        self.i = 0

    def reset(self):
        self.i = 0

    def act(self, obs):
        x, v = obs[0:2], obs[2:4]
        while self.i < len(self.wps) - 1 and np.linalg.norm(self.wps[self.i] - x) < self.reach:
            self.i += 1
        u = self.kp * (self.wps[self.i] - x) - self.kd * v
        return np.clip(u, -1.0, 1.0)


def retreat_action(x, v, x0, kp=1.5, kd=3.0):
    """Retreat: brakes first (the -kd*v term dominates), then returns towards x0.
    Valid on the start side of the wall; from the other side the agent would have to go back
    through an opening."""
    u = kp * (np.asarray(x0, float) - np.asarray(x, float)) - kd * np.asarray(v, float)
    return np.clip(u, -1.0, 1.0)
