"""Shared prop presets for the `--ball` CLI flag (play.py, swap_experiment.py)."""


def default_ball_prop():
    """A single physics-enabled sphere spawned just in front of the robot."""
    return {
        "name": "ball",
        "shape": "sphere",
        "size": 0.15,
        "mass": 1.5,
        "pos": [0.8, 0.0, 1.0],
        "restitution": 0.6,
        "friction": 0.8,
        "linear_damping": 1.5,  # 1/s decay rate on linear velocity -- viscous drag, so it coasts to a stop
        "color": [1.0, 0.1, 0.1, 1.0],
    }
