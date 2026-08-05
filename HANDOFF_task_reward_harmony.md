# HANDOFF — Harmonizing "task" vs "reward config" vs "target variable"

> **Read this first.** Fresh session, no memory. This file + the repo state (`main`, this is
> now the only branch — `policy-context-target-ui` was merged and deleted) is your full
> continuity for this sub-thread. Companion docs: `HANDOFF_control_web.md` (the broader
> control-web architecture) and `HANDOFF_stability_curriculum.md` (why `entropy_coef` is
> configurable, and why policies now save as `policies/<name>/checkpoint+raw`).
>
> **This is a plan, not a diff.** Nothing in this document has been implemented yet — it's
> analysis + a proposed path, written to be picked up and executed in a follow-up session.

## TL;DR (español)

Confundimos "task" con "reward config" porque, sin darnos cuenta, terminamos con **tres
mecanismos distintos que escriben el mismo tipo de cosa** (a qué apunta el reward de una
policy), con niveles de curación muy distintos:

1. **Task** (`--task g1` / `g1_cautious` / `g1_crouch`, registrado en código Python,
   `legged_gym/envs/__init__.py:112-116`) — el mecanismo *original*, de la comunidad
   `legged_gym`/Isaac Gym, no algo que inventamos nosotros.
2. **Target variable** (panel "Create Policy" → "Target — what the trained policy should
   do", `web/app.js` + `VARIABLE_REGISTRY` en `training.py:329-340`) — un selector guiado
   (Absolute/Raise-lower/Lowest-highest) que hoy **solo cubre una variable: altura de
   pelvis**.
3. **Reward weights (advanced)** — un grid crudo, sin curar, que expone *cualquier*
   atributo de `<Cfg>.rewards.scales` de la task elegida, sin explicación de qué significa
   cada uno ni rango razonable.

Los tres terminan escribiendo, en distintos grados de guía, al mismo lugar: la config de
reward que ve PPO. La confusión que notaste hoy es real y tiene una causa concreta,
verificable en el propio código: **`g1_cautious` es 100% redundante** con lo que ya se
puede hacer desde la UI hoy (5 overrides de reward-scale sobre `g1` + clone-from `stable`
— ver §3). `g1_crouch`, en cambio, **sí necesitaba** ser una task nueva, porque introduce
un reward TERM que no existía (`crouch_depth`), no solo un número distinto — pero ese
término quedó fuera del selector guiado (`VARIABLE_REGISTRY`), así que hoy solo es
accesible desde el grid crudo, inconsistente con cómo se expone la altura de pelvis.

La propuesta (§5) es trazar una línea clara: **si el cambio es solo un número/rango sobre
un reward que ya existe → va en la UI, sobre una task existente. Si el cambio agrega un
reward TERM nuevo, una condición de terminación nueva, o cambia obs/action → recién ahí
amerita una task nueva.** Y limpiar lo que ya se desvió de esa regla.

---

## 1. "Task" es un concepto conocido, no algo nuestro

`legged_gym/utils/task_registry.py`'s `TaskRegistry.register(name, task_class, env_cfg,
train_cfg)` es el patrón original de `legged_gym` (ETH Zürich Robotic Systems Lab), heredado
sin cambios de nombre/forma a través de la cadena de forks que este repo documenta en su
propio README (§0, "Where this fork sits in the family tree": `legged_gym` → `unitree_rl_gym`
→ `lupinjia/LeggedGym-Ex` → este fork). Es el mismo patrón que usa Isaac Gym Envs / Isaac Lab
y todo el ecosistema `rsl_rl`: **una task = un nombre que resuelve a `(clase de entorno,
config de reward/comandos/domain-rand, config de PPO)`, elegido con `--task <name>` al
entrenar.**

No es un concepto ambiguo ni improvisado — es la unidad estándar de "qué está aprendiendo el
robot" en toda esta familia de códigos. El problema no es que "task" sea confuso en sí mismo;
es que evolucionamos un segundo sistema (la UI de Create Policy) que resuelve el mismo tipo
de pregunta ("¿qué debería hacer esta policy?") sin dejar clara la relación entre los dos.

## 2. Los tres mecanismos, con código real

### (a) Task — código Python, requiere commit

`legged_gym/envs/__init__.py:112-116`:
```python
task_registry.register("g1",          G1Robot, G1RoughCfg(),    G1RoughCfgPPO())
task_registry.register("g1_cautious", G1Robot, G1CautiousCfg(), G1CautiousCfgPPO())
task_registry.register("g1_crouch",   G1Robot, G1CrouchCfg(),   G1CrouchCfgPPO())
```
Elegir una task en el panel ("Task" dropdown) selecciona uno de estos nombres. Cambiar lo que
una task premia/castiga requiere escribir una clase `Cfg` nueva y registrarla — código,
review, commit.

### (b) Target variable — UI guiada, un solo campo curado

`legged_gym/control/training.py:329-340`, `TrainingManager.VARIABLE_REGISTRY`:
```python
VARIABLE_REGISTRY = {
    "base_height": {
        "label": "Pelvis height", "unit": "m", "source": "sim_ground_truth",
        "flag": "base_height_target", "config_attr": "base_height_target",
        "range_attr": "base_height_target_range",
        "note": "...",
    },
}
```
Un solo entry hoy. El panel (`web/index.html:751-778`) lo expone con tres modos
(Absolute/Raise-lower/Lowest-highest, `web/app.js:1203-1314`) que resuelven a **el mismo
flag CLI** (`--base_height_target`, ver `web_train.py`). Sin código nuevo, sin commit — el
usuario mueve un valor y listo.

### (c) Reward weights (advanced) — grid crudo, sin curar

`web/app.js:1316+` (`renderRewardScaleFields`, ya lo tocamos hoy para el bug del input
invisible) — un `<input>` por cada atributo no-cero de `<Cfg>.rewards.scales` de la task
elegida, con el nombre del atributo en mayúsculas como única pista (`TORQUES`,
`TRACKING_LIN_VEL`, `CROUCH_DEPTH` si la task lo define). Ningún texto explicativo, ningún
rango sugerido — a diferencia de (b), que sí tiene label/unit/note/reference.

## 3. La prueba: `g1_cautious` es puro (b)/(c), `g1_crouch` es genuinamente nuevo

Diff real de las tres tasks del robot G1, `legged_gym/envs/g1/g1_config.py`:

**`G1CautiousCfg`** (línea 74, docstring propio lo dice explícito: *"only the reward weights
differ"*):
```python
class G1CautiousCfg(G1RoughCfg):
    class rewards(G1RoughCfg.rewards):
        class scales(G1RoughCfg.rewards.scales):
            tracking_lin_vel = 0.5   # was 1.0
            dof_acc = -2.5e-6        # was -2.5e-7
            dof_vel = -1e-2          # was -1e-3
            action_rate = -0.1       # was -0.01
            torques = -0.0002        # was un-penalized
```
Cinco números, todos sobre reward terms que **`G1RoughCfg` (task `g1`) ya define**. Esto es
100% reproducible HOY, sin ningún código nuevo, como: task=`g1`, clone-from=`stable`, y 5
overrides en el grid (c) — `tracking_lin_vel`, `dof_acc`, `dof_vel`, `action_rate`,
`torques`. No hay ninguna razón estructural para que `g1_cautious` sea una task separada.

**`G1CrouchCfg`** (línea 96) — v3, reemplazó dos intentos anteriores (v1: altura fija 0.6 +
comandos en cero; v2: altura fija 0.4 + comandos/pushes activos, nunca estabilizó). v1 y v2
también hubieran sido pura (b): un `base_height_target` fijo + rango de comando en cero —
ambos ya expresables desde la UI de hoy. Pero v3, la que está en uso, es distinta:
```python
class rewards(G1RoughCfg.rewards):
    crouch_depth_reference = 0.78
    class scales(G1RoughCfg.rewards.scales):
        lin_vel_z = -1.0
        base_height = 0.0      # apagado — reemplazado por crouch_depth
        crouch_depth = 3.0     # reward TERM nuevo, ver _reward_crouch_depth()
```
`crouch_depth` es una función de reward nueva (`_reward_crouch_depth()` en
`legged_gym/envs/base/legged_robot.py`) — open-ended (`-base_height`, sin setpoint fijo) en
vez de la distancia-al-cuadrado que usa `base_height`. **Eso no es un número distinto, es un
reward TERM que no existía.** No hay forma de producir ese comportamiento desde `g1` +
overrides existentes — necesitaba código nuevo. `g1_crouch` como task separada está
justificada.

Pero: `crouch_depth`/`crouch_depth_reference` tienen exactamente la forma que (b) necesita
(un label, una unidad, un valor de referencia) y **no están en `VARIABLE_REGISTRY`** — solo
son visibles/editables desde el grid crudo (c), sin guía, mientras que `base_height` (que
literalmente esta task apaga) sí tiene el selector curado. Inconsistente: la variable que
más importa en `g1_crouch` es la peor expuesta en la UI.

## 4. Por qué pasó esto (para no repetirlo)

No fue negligencia — fue orden cronológico. Las tasks (`g1`, `g1_cautious`, luego
`g1_crouch` v1) se escribieron primero, cuando la única forma de cambiar un reward era
escribir Python. Meses después se construyó el panel de Create Policy con overrides en vivo
(command envelope, target variable, reward-scale grid) para poder iterar sin tocar código —
una mejora genuina. Pero nadie volvió a mirar atrás y preguntó "¿alguna de las tasks
existentes ahora es redundante con lo que la UI ya puede hacer sola?" — así que quedaron
dos caminos al mismo destino, sin que ninguno de los dos supiera del otro.

## 5. Plan propuesto

**Estado (actualizado en la sesión de ejecución):** paso 1 (regla documentada, §5b de
`HANDOFF_control_web.md`) ✅. Paso 2 (nota explicativa para `crouch_depth` en el grid crudo,
NO promovido a `VARIABLE_REGISTRY` — ver corrección abajo) ✅. Paso 3 (`g1_cautious`
retirada) ✅ — código y clases borradas, `scripts/finetune_cautious.py` renombrado a
`finetune_from_checkpoint.py` (el nombre viejo ya era engañoso, README.md lo decía
explícitamente). Pasos 4-6 quedan para continuar.

**Corrección al paso 2, encontrada al implementar:** el docstring de `G1CrouchCfg` dice
explícito que `crouch_depth_reference` es *"a numerical zero-point only, not a target"* — no
es un setpoint como `base_height_target`, es una constante de offset para que el reward
open-ended no colapse el robot. Promoverlo a `VARIABLE_REGISTRY` (que es para *targets*
reales) habría sido engañoso. En cambio, se agregó `REWARD_SCALE_NOTES` en
`training.py` — un diccionario de notas cortas para términos del grid crudo que lo necesiten
(hoy solo `crouch_depth`, el peso del reward, que sí es lo correcto exponer ahí) — sin
prometer que sea un target ajustable.

Ninguno de los pasos 4-6 está implementado todavía — es el orden sugerido para continuar.

1. **Regla explícita, documentada** (en `HANDOFF_control_web.md` o un nuevo
   `CONTRIBUTING`-style doc): una task nueva se justifica SOLO si agrega algo estructural —
   un reward term/función nuevo, una condición de terminación nueva, un cambio de
   obs/action space, o un asset/URDF distinto. Si el cambio es "estos 3 números son
   distintos", va como override desde la UI sobre una task existente, no como task nueva.
   Esto es lo que rompe el ciclo de "evolucionar dos sistemas para lo mismo".

2. **Expandir `VARIABLE_REGISTRY`** para cubrir `crouch_depth` (y auditar si hay otros
   reward terms con la misma forma "setpoint con referencia/rango" enterrados en el grid
   crudo) — mover del nivel (c) sin curar al nivel (b) curado. Esto es aditivo, bajo
   riesgo, y arregla la inconsistencia concreta de §3.

3. **Retirar `g1_cautious` como task registrada**, reemplazándola por un *preset* de la UI
   (una configuración guardada de Create Policy: task=`g1`, clone-from=`stable`, los 5
   overrides de reward-scale ya escritos) — no requiere ningún cambio de reward/entorno,
   solo mover dónde vive esa información (de una clase Python a un JSON de preset). Antes
   de borrar el registro, confirmar que ninguna policy ya entrenada bajo `g1_cautious`
   pierde su ruta de fine-tuning (`--from_checkpoint` apunta al checkpoint, no al nombre de
   task, así que debería ser seguro — verificar igual).

4. **Auditar el resto de tasks registradas** (`g1_deepmimic`, `g1_motion_vis`, las de
   `go2`/`k1`/`tron1pf`/`tron1sf`/`bipedal_walker` si existen) con la misma pregunta de §4 —
   este handoff solo cubrió G1 en detalle porque es lo que se usó hoy. ✅ **Hecho** — ver
   resultado en §4a.

### 4a. Resultado de la auditoría (19 tasks, excluyendo g1/g1_cautious/g1_crouch)

Comparando `num_observations`/`num_privileged_obs`/clase de PPO contra la base de cada
robot: un obs/privileged-obs shape distinto o una clase de runner PPO distinta
(teacher-student, AMP, DeepMimic, ...) es evidencia de arquitectura de red y/o pipeline de
entrenamiento distintos, no solo de pesos de reward.

| Task | Config | Clasificación | Razón |
|---|---|---|---|
| `k1` | `legged_gym/envs/k1/k1_config.py:6` | BASE-N/A | Task base del robot K1. |
| `k1_deepmimic` | `k1/k1_deepmimic/k1_deepmimic_config.py:8-18` | ESTRUCTURAL | Imitación de movimiento: obs con frame-stack propio, `num_actions=22`, PPO propio. |
| `k1_motion_vis` | `k1/k1_motion_vis/k1_motion_vis_config.py:7-11` | ESTRUCTURAL (utilidad) | Marcado explícito "for motion visualization, not for training" (`__init__.py:109`). |
| `k1_amp` | `k1/k1_amp/k1_amp_config.py:11-20,110-126` | ESTRUCTURAL | Adversarial Motion Prior: `LeggedRobotAMPCfgPPO`, replay buffer/preload propios. |
| `k1_cts_amp` | `k1/k1_cts_amp/k1_cts_amp_config.py:11-24,40` | ESTRUCTURAL (marcado "unvalidated") | Concurrent teacher-student + AMP: `num_teacher` divide envs, privileged-obs shape distinto. |
| `g1_deepmimic` | `g1/g1_deepmimic/g1_deepmimic_config.py:8-18` | ESTRUCTURAL | Mismo patrón DeepMimic, 29 DoF. |
| `g1_motion_vis` | `g1/g1_motion_vis/g1_motion_vis_config.py:7-11` | ESTRUCTURAL (utilidad) | Igual que k1_motion_vis. |
| `go2` | `go2/go2_config.py:5` | BASE-N/A | Task base del robot GO2. |
| `go2_wtw` | `go2/go2_wtw/go2_wtw_config.py:4-14` + `go2_wtw.py:150-356` | ESTRUCTURAL | Walk-These-Ways: behavior params con dinámica de resampling propia, obs shape distinto. |
| `go2_ts` | `go2/go2_ts/go2_ts_config.py:5-20,51` | ESTRUCTURAL | Teacher-student: privileged/history obs distintos, encoder propio. |
| `go2_ee` | `go2/go2_ee/go2_ee_config.py:5-16,48` | ESTRUCTURAL | Explicit-estimator: critic-obs shape distinto. |
| `go2_cts` | `go2/go2_cts/go2_cts_config.py:5-17,48` | ESTRUCTURAL | Concurrent teacher-student: `num_teacher` divide envs. |
| `go2_dreamwaq` | `go2/go2_dreamwaq/go2_dreamwaq_config.py:5-18,54` | ESTRUCTURAL | DreamWaQ: decoder output/obs shape propios. |
| `go2_cat` | `go2/go2_cat/go2_cat_config.py` (hereda `Go2TSCfg`) | ESTRUCTURAL | Agrega `class constraints: enable="cat"` — condición de terminación nueva. |
| `go2_ts_depth` | `go2/go2_ts_depth/go2_ts_depth_config.py:5-19,180` | ESTRUCTURAL (marcado "unvalidated") | Agrega obs de cámara de profundidad (`num_privileged_obs=268`). |
| `go2_nav` | `go2/go2_nav/go2_nav_config.py:5-11,100` | ESTRUCTURAL | Navegación: obs incluye heightmap (187 celdas). |
| `tron1pf` | `tron1pf/tron1pf_config.py:4` | BASE-N/A | Task base de TRON1 (perfil PF). |
| `tron1pf_ee` | `tron1pf/tron1pf_ee/tron1pf_ee_config.py:4-16,176` | ESTRUCTURAL | Explicit-estimator, critic-obs distinto. |
| `tron1sf` | `tron1sf/tron1sf_config.py:5` | BASE-N/A | Morfología de robot distinta (SF vs PF, `num_actions=8` vs `6`) — base propia, no variante. |

**Conclusión: 0 candidatas a retirar.** A diferencia de `g1_cautious`, ninguna de estas 19
es un caso de "solo cambian pesos de reward" — todas las no-base tienen una diferencia real
de obs space, privileged-obs, o clase de PPO/termination. `g1_cautious` era la única
anomalía en todo el árbol de tasks.

5. **UI: explicar la diferencia al elegir task.** Cuando el panel muestra el dropdown
   "Task", agregar una nota de una línea por task explicando qué la hace estructuralmente
   distinta (ej. para `g1_crouch`: "adds an open-ended crouch reward term instead of a
   fixed height target") — así la próxima persona que mire el panel entiende por qué existe
   como task y no como override, sin tener que leer el código fuente.

6. **(Opcional, más adelante) Unificar visualmente** las tres secciones del panel
   ("Command envelope", "Target variable", "Reward weights (advanced)") bajo un único
   framing de "curado vs. crudo" — por ejemplo, dejar claro con una etiqueta que (b) es la
   forma recomendada/guiada y (c) es modo avanzado para lo que (b) todavía no cubre. Baja
   prioridad, es cosmético una vez que §1-5 estén resueltos.

## 6. Qué NO tocar sin pensarlo dos veces

- El propio mecanismo de `task_registry` (§1) — es el estándar de la comunidad, no hay que
  reinventarlo, solo usarlo con más disciplina.
- Los checkpoints ya entrenados bajo `g1_cautious`/`g1_crouch` — cualquier limpieza de
  tasks registradas es sobre el CÓDIGO de definición, no sobre los `.pt` ya generados, que
  siguen siendo válidos y cargables sin importar qué pase con el registro.
