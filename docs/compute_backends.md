# Backends de cómputo — dónde corre realmente cada entrenamiento

Autoridad única sobre la pregunta *"¿qué se entrena en mi máquina y qué se entrena en la
nube?"*. Si otro doc dice algo distinto sobre cómputo, gana este.

Un **training backend** es una respuesta a "dónde y cómo corre este job". No lo elegís por
venv ni por intérprete: pedís uno con `rugiar train --backend <nombre>` y el registry
(`legged_gym/control/backends/`) resuelve `(backend pedido, stack que necesita la task)` →
descriptor concreto. Una task de locomoción Genesis (`g1`, `go2`, …) y una de
motion-tracking mjlab (`Rugiar-G1-Mimic`) usan el mismo comando; el despacho es automático.

---

## Los backends de hoy

| Backend | Tipo de máquina | Simulador | Tasks que sirve | Cómo se pide | Estado |
|---|---|---|---|---|---|
| `local-genesis` | Tu máquina (probado en Mac Apple Silicon), **CPU** | Genesis | locomoción (`g1`, `go2`, `k1`, `tron1*`, …) | `rugiar train --backend local` | **activo** |
| `local-mjlab` | Tu máquina, cualquier plataforma | mjlab (MuJoCo) | motion-tracking (`Rugiar-G1-Mimic`) | `rugiar train --backend local` | **activo** |
| `kaggle` | Kernel remoto de Kaggle (GPU Tesla P100) | Isaac Gym | locomoción únicamente | `rugiar train --backend kaggle` | **activo** |
| `local-nvidia` | GPU NVIDIA local dedicada | Isaac Lab / Isaac Gym (a definir) | — | — | **placeholder, no implementado** |
| `nvidia-cloud` | Cómputo NVIDIA en la nube | Isaac Lab / Isaac Gym (a definir) | — | — | **placeholder, no implementado** |

`--backend local` es un solo valor pedible: `local-genesis` y `local-mjlab` se distinguen
por la task, no por lo que escribís.

### `local-genesis`

- Corre `legged_gym/scripts/web_train.py` como subproceso, con el intérprete de `.venv`
  y `SIMULATOR=genesis`.
- **Hoy corre en CPU.** El descriptor fija los flags `--headless --cpu`, y `--cpu` de
  `web_train.py` es `store_true, default=True`. `web_train.py` acepta `--gpu`, que hace
  `gs.init(backend=gs.gpu)` — y `gs.gpu` lo resuelve Genesis según la plataforma (Metal en
  macOS, CUDA en Linux+NVIDIA) — pero el registry **no** pasa ese flag. Ver
  `legged_gym/scripts/web_train.py:142` y `TrainingManager.system_info()`.
- **Decisión actual: se queda en CPU.** Usar Metal en Mac no es prioridad hoy — se
  evaluará como mejora futura si hace falta acelerar el entrenamiento local de Genesis.

### `local-mjlab`

- Corre `legged_gym/scripts/mjlab_train.py` con el intérprete de `.venv-mjlab`.
- **CPU forzado siempre**, en cualquier plataforma: el backend exporta
  `CUDA_VISIBLE_DEVICES=""` y `mjlab_train.py --device` default es `cpu` (rechaza `--gpu`
  explícitamente; para GPU se pasaría `--device cuda:0`).
- Necesita `--motion_file` (un clip de referencia); los knobs de locomoción
  (`--cmd_vx`, `--push_robots`, targets de estabilidad) no aplican y dan error temprano.

### `kaggle`

- Backend **remoto**: no hay subproceso local. Un `KaggleRunner` en un thread sube y
  pollea un kernel.
- El kernel **clona el repo desde GitHub** (`https://github.com/GIAR-UTN/RobotUniversityGiar.git`,
  branch `main` por defecto, `--depth 1`): entrena el **HEAD remoto**, no tu working tree.
  Si no pusheaste, tu cambio no viaja.
- Usa **Isaac Gym**, no Genesis, y no por omisión: el free tier de Kaggle da una P100
  (Pascal, sm_60) y el JIT de GPU de Genesis necesita Volta+ (sm_70+). El pipeline GPU de
  PhysX de Isaac Gym sí corre en Pascal.
- Solo tasks Genesis/locomoción — el bootstrap del kernel es específico de Isaac Gym.
- Un `--from_checkpoint` local no se pasa tal cual: el runner sube el archivo como Dataset
  privado y reapunta la ruta al mount `/kaggle/input/`.
- Setup de credenciales (`~/.kaggle/kaggle.json`): README §2 "Kaggle (cloud GPU training)".

### `local-nvidia` y `nvidia-cloud` — reservados, sin implementar

No hay implementación todavía. Los archivos
`legged_gym/control/backends/local_nvidia.py` y
`legged_gym/control/backends/nvidia_cloud.py` existen como **lugar reservado**: ahí va una
GPU NVIDIA local dedicada y el cómputo NVIDIA en la nube (Isaac Lab / Isaac Gym), este
último en desarrollo por otro equipo en paralelo. Nada los pide todavía desde el CLI.

---

## Cómo se agrega un backend nuevo

El paquete es `legged_gym/control/backends/`:

| Archivo | Qué es |
|---|---|
| `base.py` | El descriptor `TrainingBackend` y sus hooks (`interpreter`, `prepare_env`, `validate_params`, `preflight`, `launch_remote`, …). |
| `local_genesis.py`, `local_mjlab.py`, `kaggle.py` | Los tres backends activos. |
| `local_nvidia.py`, `nvidia_cloud.py` | Placeholders — **empezá acá** si venís a sumar cómputo NVIDIA. |
| `__init__.py` | El registry: `BACKENDS`, `REQUESTABLE_BACKENDS`, `resolve_training_backend()`. |

La forma, en dos pasos:

1. Escribí los hooks que tu backend necesita en su propio archivo. Uno **local** aporta
   intérprete + script + entorno; uno **remoto** se saltea todo eso y aporta un
   `preflight()` (¿hay credenciales?) y un `launch_remote()`.
2. Agregá una entrada `TrainingBackend(...)` a `BACKENDS`. Si estrenás un
   `requested_as` nuevo, ese nombre pasa a ser válido en `--backend` automáticamente:
   `REQUESTABLE_BACKENDS` y los mensajes de error se derivan de `BACKENDS`, no se
   mantienen a mano.

Leé los comentarios de cabecera del registry antes de escribir: documentan qué campo del
descriptor es *forma persistida* (`job_backend`, `simulator` terminan en `meta.json` y en
la UI) y por lo tanto no se puede cambiar libremente.

---

## Los dos venvs locales

Los dos simuladores locales **no pueden convivir en un mismo venv** (colisión de
`rsl_rl` vendorizado vs. `rsl-rl-lib` de PyPI, y `mujoco` 3.10 vs 3.11 — ver
`docs/mjlab_migration.md` R1). Por eso son dos:

| Venv | Backend | Cómo se arma |
|---|---|---|
| `.venv` | `local-genesis` (y el CLI `rugiar`, y `kaggle` desde acá) | `./install.sh` (agregá `--with-kaggle` para el extra de Kaggle) |
| `.venv-mjlab` | `local-mjlab` | `./install.sh --with-mjlab`, o a mano (abajo) |

A mano, el de mjlab:

```bash
python3.12 -m venv .venv-mjlab
.venv-mjlab/bin/pip install --upgrade pip
.venv-mjlab/bin/pip install -e .[mjlab]
```

Nunca instales `[mjlab]` en `.venv`. E invocá siempre ese venv con `-I` (modo aislado)
cuando corras algo a mano desde la raíz del repo, para que el `rsl_rl/` vendorizado del
repo no le gane a `rsl-rl-lib`:

```bash
SIMULATOR=mjlab CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -I <script>
```

No hace falta activar el venv correcto antes de entrenar: `rugiar train` elige el
intérprete solo según la task.

---

## Ver también

- `legged_gym/control/ARCHITECTURE.md` §1 / §1b — el área Training y sus fronteras.
- `docs/mjlab_migration.md` — por qué mjlab vive en su propio venv (R1).
- `docs/mjlab_training_contract.md` — el contrato del entrenamiento mjlab.
- `.claude/skills/rugiar/SKILL.md` — uso día a día del CLI `rugiar`.
