# 🦿 LeggedGym-Ex

A [legged_gym](https://github.com/leggedrobotics/legged_gym) based framework for training legged robots in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main), [IsaacGym](https://developer.nvidia.com/isaac-gym) and [IsaacSim](https://developer.nvidia.com/isaac/sim).

## 🌟 Features

- **Totally based on [legged_gym](https://github.com/leggedrobotics/legged_gym)**

  This framework keeps most apis and conventions consistent with legged_gym, providing good readability and better control over training pipeline.

- **Integration of multiple simulators**
  
  We support training in either of three simulators: IsaacGym, Genesis and IsaacSim.
  
  Quick tips for choosing among three simulators: 
  
  - Faster training but worse rendering -> IsaacGym
  - Both training speed and support for fluid, soft materials -> Genesis 
  - More realistic rendering at the cost of training speed -> IsaacSim.

- **Accelerated depth image rendering based on warp**
  
  We integrate [warp](https://github.com/nvidia/warp) into this framework to accelerate the depth image rendering on Nvidia GPU.

- **Incorporation of various methods in published RL papers**
  
  | Method | Paper Link | Code |
  |--------|------------|----------|
  | Periodic Gait Reward | [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387) | [go2_wtw](https://github.com/lupinjia/LeggedGym-Ex/blob/main/legged_gym/envs/go2/go2_wtw/go2_wtw.py#L322) |
  | Walk These Ways | [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://gmargo11.github.io/walk-these-ways/) | [go2_wtw](https://github.com/lupinjia/LeggedGym-Ex/blob/main/legged_gym/envs/go2/go2_wtw) |
  | System Identification | [Learning Agile Bipedal Motions on a Quadrupedal Robot](https://arxiv.org/abs/2311.05818) | [go2_sysid](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/go2/go2_sysid) |
  | Teacher-Student | [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/) | [go2_ts](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/go2/go2_ts) |
  | Explicit Estimator | [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) | [go2_ee](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/go2/go2_ee) |
  | Constraints as Terminations | [CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning](https://constraints-as-terminations.github.io/) | [go2_cat](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/go2/go2_cat) |
  | DreamWaQ | [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602) | [go2_dreamwaq](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/go2/go2_dreamwaq) |
  | SPO (Simple Policy Optimization) | [Simple Policy Optimization](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/MyRepositories-hub/Simple-Policy-Optimization&ved=2ahUKEwjL9vLX7auSAxVZlFYBHWkFBkIQFnoECBgQAQ&usg=AOvVaw1nGHIXtdYwpu3WV9lUgRWN) | [`legged_robot_config.py`](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/base/legged_robot_config.py) |
  | CTS (Concurrent Teacher Student) | [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://clearlab-sustech.github.io/concurrentTS/) | [go2_cts](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/go2/go2_cts) |
  | DeepMimic | [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/abs/1804.02717) | [g1_deepmimic](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/g1/g1_deepmimic) |
  | AMP (Adversarial Motion Priors) | [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/abs/2104.02180) | [k1_amp](https://github.com/lupinjia/LeggedGym-Ex/tree/main/legged_gym/envs/k1/k1_amp) |

## 🛠 Installation and Usage

Please refer to the [doc of this repo](https://leggedgym-ex-doc.readthedocs.io/en/latest/).

## 🖼️ Gallery

| Robot | Sim | Real |
|--- | --- | --- |
| Unitree Go2 | <img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/ee_demo.gif?raw=true" width="428"> | [video](https://www.bilibili.com/video/BV1FPedzZEdi/) |
| TRON1_PF | <img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/tron1_pf_ee_demo.gif?raw=true" width="428"> | [video](https://www.bilibili.com/video/BV1MdePzcEvk/?spm_id_from=333.1387.homepage.video_card.click&vd_source=50fc92fb0e7291031bbb72e3c05b2edc) |
| TRON1_SF | <img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/tron1_sf_demo.gif?raw=true" width="428"> | |
| Unitree G1 DeepMimic | <img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/g1_mimic_demo.gif?raw=true" width="428"> | |
| Booster K1 | <img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/booster_k1_demo.gif?raw=true" width="428"> | [video](https://www.bilibili.com/video/BV1GyXgBmEa9/) |
| Go2 Parkour with Depth Image | <img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/go2_depth_parkour_demo.gif?raw=true" width="428"> | [video](https://www.bilibili.com/video/BV1riwMzmEZT/?spm_id_from=333.1387.homepage.video_card.click) |


## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym)
- [isaaclab](https://github.com/isaac-sim/IsaacLab/tree/8e15af9f2ca18a0c3940b44e36fdc128995ecf16)
- [aerial_gym_simulator](https://github.com/ntnu-arl/aerial_gym_simulator)
- [warp](https://github.com/nvidia/warp)

## TODO

- [x] Add support for IsaacSim simulator (2026/02/15)
- [x] Add support for DeepMimic Implementation (2026/02/28)
- [x] Add support for Booster K1 Robot (2026/03/18)
- [x] Add support for AMP Implementation (2026/03/28)
- [x] Add support for warp-based depth camera (2026/04/26)
- [x] Add Dockerfiles and viser visualization support for training in cloud server (2026/06/15)
- [ ] Add support for warp-based height query
- [ ] Add support for latest rsl-rl-lib

## Contacts

You can add our Feishu group to get latest update or ask questions:

<img src="https://github.com/lupinjia/leggedgym-ex-doc/blob/main/source/_static/images/feishu_group_qrcode.png?raw=true" width="350">

Or you add our [discord channel](https://discord.gg/s9Hd8B7r6) to get updates or ask questions:
