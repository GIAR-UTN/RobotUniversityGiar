# GIAR's real robot: Unitree G1-EDU-U8

GIAR/UTN now has a real humanoid. This doc covers what the robot is and how RUgiar
simulates it, so that every policy can be checked in sim before it touches hardware.

## The robot

Source: Big Dipper product 7921, code `G1-EDU-U8`. The spec was read from Big Dipper's API on
2026-11-17; their product page renders client-side.

| | |
|---|---|
| Total DOF | **41**: legs 6+6, waist **3**, arms 7+7 (the full 29-DOF body), plus 12 in the hands |
| Hands | Two five-finger **BrainCo Revo 2 touch** hands with tactile sensing. These are **not** Unitree Dex3-1. |
| Compute | 8-core CPU, plus an expansion board rated at 100 TOPS (Jetson Orin NX class). High- and low-level secondary development are both supported. |
| Sensors | Depth camera and 3D lidar (Livox MID-360) |
| Size / mass | 1270 × 450 × 200 mm, about 35 kg with battery |

Developer docs: <https://support.unitree.com/home/en/G1_developer/about_G1>

### From the donor's shared folder ("Humanoide", GIAR Drive, 2026-11-17)

The folder holds 3 Unitree PDFs (Remote Control, Battery and Charger, Waist Fastener) and a
links doc, which confirms:
- Camera: **Intel RealSense D435i**
  ([librealsense](https://github.com/realsenseai/librealsense),
  [realsense-ros](https://github.com/realsenseai/realsense-ros)).
- Lidar: **Livox MID-360** ([Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2),
  [livox_ros2_driver](https://github.com/Livox-SDK/livox_ros2_driver)).
- SDKs: [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) (C++),
  [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) (what
  `RealAdapter` targets), [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2), and
  [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) for sim2sim.

**Waist fastener.** The G1-EDU ships a waist-fixing kit: two brackets and M5 screws that
lock the waist's roll and pitch. It is enabled in the Unitree Explorer app under
"waist locking". **Which mode the robot runs in decides the sim model:**
- Unlocked: 3-DOF waist. This is what `Rugiar-G1Edu-Velocity*` and Unitree's `velocity/v0`
  policy assume.
- Fastened: roll and pitch are fixed, which matches
  `g1_description/g1_29dof_lock_waist*`. Neither of our tasks covers this mode yet. It would
  need its own task, and a policy trained for it.

Check how the GIAR robot is configured before running any policy on it.

## Sim model

The body is the **29-DOF G1 with the waist unlocked**. That is exactly mjlab's own
`unitree_g1` asset, so the EDU tasks build on mjlab. In the Genesis asset tree the same body
is `resources/robots/unitree_robotics/g1_description/g1_29dof_rev_1_0.*`.

**The hands are not modeled yet.** `g1_29dof_with_hand*` is the Dex3 hand, which is the
wrong one. Locomotion ignores the hands, but their extra mass at the wrists is not
randomized yet. Do that before running anything on hardware.

The Genesis locomotion policies (`g1`, `walk_gpu_*`, `stable*`) are **12-DOF, legs only**.
They do not drive the EDU's waist or arms.

## Tasks (mjlab)

| Task | Actor obs | Use |
|---|---|---|
| `Rugiar-G1Edu-Velocity` | 96 = ang_vel 3, gravity 3, joint_pos 29, joint_vel 29, last_action 29, command 3. No `base_lin_vel`, because the real robot can't measure it. | Training our own walking policies |
| `Rugiar-G1Edu-Velocity-UnitreeObs` | 98 = unitree_rl_mjlab's contract: ang_vel, gravity, command, gait phase (0.6 s), joint_pos, joint_vel, last_action. Uses Unitree's home pose. | Running Unitree's published G1 policy, unmodified |

Both tasks are registered in `mjlab_tasks/velocity/`. The driver picks
`MjlabVelocityAdapter` (`legged_gym/control/mjlab_adapter.py`) for any task that has a
`twist` command. That adapter gives the web UI a working Command panel: manual vx/vy/yaw,
clamped to the trained envelope, and pinned against mjlab's own resampling.

```bash
export SIMULATOR=genesis
rugiar train --task Rugiar-G1Edu-Velocity --name <name> --max_minutes 600
# then drive it from the control web (rugiar_driver_mjlab.py --task Rugiar-G1Edu-Velocity-UnitreeObs)
```

## Policies

| Policy | Task | Status (headless, CPU, 2026-11-17) |
|---|---|---|
| `unitree_g1_29dof_velocity_v0` | `...-UnitreeObs` | **Walks.** Unitree's own policy (unitree_rl_mjlab `velocity/v0`, Apache-2.0). 0 falls at standing, 0.5 and 1.0 m/s forward, 0.4 m/s lateral and 0.8 rad/s turning, over 15–20 s each. Heading drifts at speed. |
| `g1edu_walk_smoke` | `Rugiar-G1Edu-Velocity` | Pipeline smoke only: 230 iterations, 10 min of CPU, 64 envs. It stands but does not walk yet. A real run needs hours. |

Other external candidates, best first (the survey was done 2026-11-17):
- unitree_rl_lab `g1_29dof` velocity: Isaac Lab, a 480-value observation with a 5-frame
  history, and a different joint order.
- GR00T-WholeBodyControl Walk/Balance: 15-DOF lower body, 516-value observation.
- FALCON / ASAP `g1_29dof`: HumanoidVerse joint order.
- NVIDIA GEAR-SONIC: whole-body tracking. The heaviest of these stacks.

Each one needs its own observation-contract task, built the same way as the UnitreeObs
variant.

## Before hardware

1. Randomize the mass of the Revo 2 hands, or model them.
2. Finish `RealAdapter` with `unitree_sdk2py` (see its `NotImplementedError` notes; AGENTS.md
   rule 6 no longer blocks this, since the hardware exists).
3. Sim2sim against Unitree's own MuJoCo deploy (`unitree_rl_mjlab/deploy`) with the same
   ONNX, so that our observation builder can be compared with theirs value by value.
