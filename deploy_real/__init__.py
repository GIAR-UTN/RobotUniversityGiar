"""
Real-hardware deployment package — kept separate from legged_gym/control/ on
purpose. This package is allowed to import unitree_sdk2py; legged_gym/control/
is not, and never should be, so that everything importing legged_gym.control
stays installable on a machine with no SDK and no robot attached (this repo
was built/tested entirely on a Mac with neither).
"""
