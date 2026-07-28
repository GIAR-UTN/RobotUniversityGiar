# Simulator Abstraction Layer

**Generated:** 2025-04-03
**Path:** legged_gym/simulator/

## OVERVIEW

Multi-simulator abstraction supporting Genesis, IsaacGym, and IsaacLab through a common interface. Enables same training code to run on any simulator via `SIMULATOR` env var.

## STRUCTURE

```
legged_gym/simulator/
├── simulator.py           # Abstract base class (ABC)
├── genesis_simulator.py   # Genesis backend (~49KB)
├── isaacgym_simulator.py  # IsaacGym backend (~79KB)
└── isaaclab_simulator.py  # IsaacLab backend (~57KB)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new simulator | Extend `Simulator` ABC | Implement all abstract methods |
| Genesis-specific | `genesis_simulator.py` | XML-based, heightfield terrain |
| IsaacGym-specific | `isaacgym_simulator.py` | URDF-based, trimesh terrain |
| IsaacLab-specific | `isaaclab_simulator.py` | URDF-based, trimesh terrain |
| Simulator selection | `base_task.py` | `SIMULATOR` env var determines class |

## CONVENTIONS

**Simulator Selection:**
```python
export SIMULATOR=genesis    # or isaacgym, isaaclab
```

**Asset Path Conventions:**
- IsaacGym/IsaacLab: `cfg.asset.file` (URDF path)
- Genesis: `cfg.asset.xml_file` (XML path required)

**ABC Pattern:**
```python
class MySimulator(Simulator):
    def step(self): ...
    def reset_idx(self, env_ids): ...
    def get_dof_state(self): ...
```

## ANTI-PATTERNS

1. **IsaacGym Reset Bug**: After `reset()`, call `forward()` before reading rigid body states
2. **IsaacLab CPU Tensors**: Domain randomization tensors must be on CPU (`set_material_properties`, `set_masses`, `set_coms`)
3. **Genesis XML Required**: Must provide XML file path when using Genesis simulator
4. **Heightfield Limitation**: Heightfield terrain not implemented for IsaacLabSimulator
5. **Trimesh in Genesis**: Trimesh terrain not validated for Genesis (use heightfield)
6. **IsaacGym Rigid Body Props**: Cannot be modified on the fly (fixed at init)
7. **Torque Dimensions**: Must match DOF count even for non-actuated DOFs
8. **Genesis GPU Device Mismatch**: Genesis getters (`get_vel()`, `get_ang()`, `get_dofs_limit()`, etc.) may return CPU tensors even when the scene runs on CUDA. Additionally, Genesis's internal `sanitize_index` force-moves all indices to CPU, causing `scatter_` failures on CUDA buffers. The workaround is runtime monkey-patches in `genesis_simulator.py` plus explicit `.to(self._device)` on all getter results before PyTorch tensor operations.

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| Simulator | ABC | `simulator.py` | Base interface |
| GenesisSimulator | Class | `genesis_simulator.py` | Genesis backend |
| IsaacGymSimulator | Class | `isaacgym_simulator.py` | IsaacGym backend |
| IsaacLabSimulator | Class | `isaaclab_simulator.py` | IsaacLab backend |

## IMPLEMENTATION NOTES

- Each simulator manages its own GPU tensors
- Buffer initialization in `_init_buffers()`
- DOF indices alignment handled in `_dof_indices`
- Camera/sensor updates in `update_sensors()`
