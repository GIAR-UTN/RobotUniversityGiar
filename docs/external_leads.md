# External leads — triage log

Leads that other LAS agents forward to RuGIAR (the first batch came from Monkey on
2026-11-17 and was taken from José's Instagram saves). The claims below are the post
authors' own and have not been verified. Each lead is tried one at a time, and its
result is written here. A lead with no result has not been tried yet.

| # | Lead | Relevance to RUgiar | Status / result |
|---|---|---|---|
| 1 | Sim2real guide, rotary inverted pendulum ([reel](https://www.instagram.com/reel/DfQ4K_-ui-c/)) | Course material: a small sim2real example before the G1 | not tried |
| 2 | MuJoCo explained ([reel](https://www.instagram.com/reel/DeQ7CMLDvbk/)) | Course material: mjlab is our default driver now | not tried |
| 2b | Bonsai, generative sim with rare cases ([reel](https://www.instagram.com/reel/DetQA12DbAN/)) | Rare-case/domain-randomization ideas for G1 training | not tried |
| 3a | SuperMimic, motion imitation ([reel](https://www.instagram.com/reel/DfRea5CvXaS/)) | Compare with our mjlab `Rugiar-G1-Mimic` pipeline (BeyondMimic lineage) | **next candidate** |
| 3b | NVIDIA MotionBricks, walk from a T-pose ([reel](https://www.instagram.com/reel/DfHyJofDMqS/)) | Reference-motion generation for mimic clips | not tried |
| 3c | Humanoid learning to fall with judo ([reel](https://www.instagram.com/reel/Decg7t0j5kE/)) | Fall/recovery policy; today only `damping` handles falls | not tried |
| 4 | NVIDIA SimReady, ~3,500 OpenUSD objects ([reel](https://www.instagram.com/reel/De_tywQN2z6/)) | Props for WHRG arenas (needs USD → MJCF conversion) | not tried |
| 5–7 | Manipulation (π0, LeRobot Diffusion Policy, cheap arms), MCP-driven CAD/PCB, vision/lidar | Out of current scope (locomotion/mimic only) | skipped |

## Eden (from Javier Villalba, 2026-11-17)

<https://embodied-ai-nexus.github.io/Eden/> ([repo](https://github.com/embodied-ai-nexus/Eden),
Apache-2.0). A robot-learning framework built on Genesis: one typed config per task,
GPU-parallel training, the same policy in sim2sim (MuJoCo) and on a real robot. It claims
47 robots, 66 tasks and a 21-policy zoo (`eden inference --policy <name>`). It has **G1
variants that include `G1Dofs29`**, which is the body of our G1-EDU-U8, plus Dex3/Inspire
hand composites. The Revo 2 hand isn't listed. It is Linux+NVIDIA first, and macOS is
best-effort (`--cpu`). It is very new (6 commits on main, 256 stars, Nov 2026), and the
policy weights weren't located.

**Status: noted, not tried.** Today our 29-DOF EDU path is mjlab
(`Rugiar-G1Edu-Velocity*`, see `docs/g1_edu_u8.md`), and mjlab is the default stack
(Genesis is legacy here). Eden would be worth a look for one of these reasons:
(a) its policy zoo has a G1Dofs29 locomotion or fall-recovery policy whose obs contract we
can reproduce, like the UnitreeObs variant; (b) we want its sim2real deploy path as a
reference for `RealAdapter`; (c) we go back to Genesis for GPU training. Check the zoo
first. Don't adopt the framework wholesale.

Rules: if a lead turns out to be useful, its notes move into the doc that the lead affects
(for example `docs/mjlab_migration.md`). This file holds only the triage.
