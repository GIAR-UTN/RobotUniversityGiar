"""This repo's own mjlab task registrations (phase 3 of the mjlab
migration, see docs/mjlab_migration.md). Import this module to populate
mjlab's task registry with our tasks, the same way `import mjlab.tasks`
populates mjlab's own -- e.g.:

    import mjlab_tasks  # noqa: F401
    from mjlab.tasks import registry
    registry.load_env_cfg("Rugiar-G1-Mimic")

Named `mjlab_tasks`, not `src` -- unitree_rl_mjlab's own top-level package
is literally named `src`, which is part of why it isn't pip-installable
(see docs/mjlab_migration.md §0). Don't repeat that mistake here.
"""
from . import tracking  # noqa: F401
