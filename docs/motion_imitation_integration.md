# Integrando el trabajo de Javier Villalba (imitación de video / bailes)

Este doc es el resultado de la Fase 1 (investigación) para traer al repo el
trabajo de **Ing. Javier Villalba** — imitación de movimiento/baile a partir
de video para el G1 — de modo que él pueda seguir trabajándolo desde acá en
vez de en su entorno privado. Cubre: qué hizo, qué formatos usa, qué ya
tenemos nosotros que sirve tal cual, y qué falta convertir/escribir.

## Fuentes revisadas

1. **Kaggle notebook de Javier** — `kaggle.com/code/jvillalba007/unitree-rl-mimic`.
   **Corrección tras bajar el output real**: no corre sobre `unitree_rl_gym`
   como se pensaba en la Fase 1 (eso es solo lo que menciona el texto del
   chat) — el kernel en realidad clona y entrena sobre
   `unitreerobotics/unitree_rl_mjlab` (familia MuJoCo/IsaacLab, NO
   IsaacGym), exactamente el repo que Javier señaló por separado como "el
   intermedio que me sirve" (mensaje del 2026-08-16). Llegó hasta la
   iteración 7000 (`model_7000`). Javier aclaró que es el **G1 EDU
   completo, 29 motores** (cuerpo entero: piernas + cintura + brazos), a
   diferencia del trabajo previo del equipo que era solo 12 DOF
   (piernas). El checkpoint en sí no llegó a mandarse por WhatsApp — hay que
   bajarlo del propio notebook de Kaggle (sus outputs/logs).
2. **Dataset `exptech/g1-moves`** (Hugging Face, CC-BY-4.0) — el que Javier
   encontró y compartió. 60 clips (danza/karate/bonus, ~30 min a 60 FPS),
   organizados en 4 etapas por clip:
   - **Capture**: mocap crudo BVH (esqueleto humano 51 joints) + FBX/GIF/MP4.
   - **Retarget**: `.pkl`/`.csv` con **ángulos de joint ya retargeteados al
     G1 de 29 DOF** — 3 (root pos) + 4 (root quat) + 29 (dof) = 36 columnas.
   - **Training**: `.npz` con cinemática directa ya calculada sobre el
     retargeting — posiciones/velocidades de joints, posiciones/orientaciones
     de bodies en world frame, velocidades lineales/angulares.
   - **Policy**: checkpoints `.pt` + export `.onnx` (con normalización de
     observación "horneada" adentro), video de rollout, config de entorno/
     entrenamiento, métricas. Red: "160 obs → [512, 256, 128] → 29 actions".
   - Whitepaper: `exptech-g1-moves.static.hf.space/G1_Moves_Whitepaper.pdf`.
   - Repo: `github.com/experientialtech/g1-moves`.
3. **`GIAR-UTN/GIAR-motion_viewer`** — ya subido por Javier al org del equipo,
   deploy vivo en `giar-mv.9zteam.pp.ua`. Visor WebGL 100% client-side: soltás
   una carpeta con URDF/MJCF+STL y un `.npz`/`.pkl`, reproduce sin instalar
   nada. Formatos que acepta:
   - `.npz`: `fps`, `body_pos_w` `(N, num_bodies, 3)`, `body_quat_w`
     `(N, num_bodies, 4)` en **wxyz**.
   - `.pkl`: `fps`, `root_pos` `(N,3)`, `root_rot` `(N,4)` en **xyzw**
     (convención MuJoCo), `dof_pos` `(N, num_dof)` — hace FK automático.
   - Usa Z-up internamente (Isaac Sim/MuJoCo), convierte a Y-up solo para
     renderizar con Three.js.
4. **`unitreerobotics/unitree_rl_mjlab`** — mencionado por Javier como un
   intermedio con soporte de "movimientos" que ya le sirve. No lo profundicé
   todavía (queda para cuando se decida si conviene como referencia de reward/
   obs para nuestro propio entrenamiento de mimic).

## Lo que YA tenemos en el repo (sin usar, nadie lo mencionó hasta ahora)

Antes de traer nada de afuera, esto ya está armado y sin ningún policy
entrenada encima:

- **Task `g1_deepmimic`** (`legged_gym/envs/g1/g1_deepmimic/`) — usa
  `G1Flat29DofCommonCfg` (el mismo esqueleto de 29 DOF que necesita Javier:
  piernas + waist_yaw/roll/pitch + brazos con muñeca de 3 DOF). Ya calcula
  observación de referencia (`ref_motion_obs`), reset-desde-referencia, y
  reward contra la trayectoria objetivo.
- **Task `g1_motion_vis`** — para reproducir/visualizar una referencia sin
  policy (útil para "verlo" antes de entrenar nada).
- **`legged_gym/utils/motion_loader.py`** (`MotionLoader`) — carga un único
  `.pkl` por env con este esquema exacto:
  ```
  fps                          escalar
  root_pos                     (N, 3)   world frame
  root_rot                     (N, 4)   xyzw, world frame
  root_lin_vel                 (N, 3)   opcional (default: ceros)
  root_ang_vel                 (N, 3)   opcional (default: ceros)
  dof_pos                      (N, 29)  mismo orden que dof_names de la cfg
  dof_vel                      (N, 29)  opcional (default: ceros)
  key_body_pos_relative_to_base (N, 19, 3) opcional (default: ceros)
  ```
- **15 clips ya en `resources/reference_motion/unitree_g1/`**, cada uno en
  4 variantes: `raw_run/` (fuente: root_pos/root_rot float64 +
  `local_body_pos` de 38 links + `link_body_list` — SIN velocidades ni
  key_body_pos_relative_to_base) y `{genesis,isaacgym,isaaclab}_run/`
  (derivados: agregan `root_lin_vel`/`root_ang_vel`/`dof_vel` por diferencia
  finita y `key_body_pos_relative_to_base` extraído/transformado del
  `local_body_pos` de 38 links a los 19 "key bodies" que usa
  `G1Flat29DofCommonCfg.key_bodies`). **Comparé `genesis_run` vs
  `isaacgym_run` del mismo clip byte a byte en los campos numéricos — son
  idénticos** (misma root_pos/root_rot/dof_pos/dof_vel), o sea el split por
  simulador no cambia los valores, al menos en este caso.
- Estos 15 clips son de tipo mocap (correr, girar, side-step — AMASS-style),
  **no bailes** — son de otro origen (probablemente el compañero que armó
  `g1_deepmimic`), no de Javier.
- **Ningún policy entrenado existe todavía para `g1_deepmimic` ni
  `g1_motion_vis`** en `policies/` — el pipeline está escrito pero nadie lo
  corrió.

## Compatibilidad: g1-moves vs nuestro MotionLoader

Buena noticia — la convención de joints de g1-moves (29 DOF: piernas
pitch/roll/yaw + rodilla + tobillo, waist yaw/roll/pitch, brazos
shoulder pitch/roll/yaw + codo + muñeca roll/pitch/yaw) **coincide en orden y
agrupación** con `G1Flat29DofCommonCfg.dof_names` — ambos derivan
aparentemente de la misma convención de referencia de Unitree. La rotación
de root también usa la misma convención xyzw. Esto es más compatible de lo
esperado.

Lo que falta para convertir un clip de g1-moves a algo que `MotionLoader`
pueda leer:

| Campo requerido | De dónde sale en g1-moves |
|---|---|
| `root_pos`, `root_rot`, `dof_pos` | Directo del `.pkl` de la etapa **Retarget** |
| `dof_vel`, `root_lin_vel`, `root_ang_vel` | Están en el `.npz` de la etapa **Training** (o se derivan por diferencia finita, como ya hace el pipeline existente para `raw_run` → `{sim}_run`) |
| `key_body_pos_relative_to_base` | El `.npz` de Training trae posiciones de bodies en world frame — hay que restar la posición del root y rotar al frame del body base, igual que ya se hace para los 15 clips existentes (falta encontrar/escribir ese script — no está commiteado en este repo) |

**Gap real, no menor**: el script que convierte `raw_run` → `{genesis,
isaacgym, isaaclab}_run` (diferencia finita + extracción de key bodies) no
está en el repo — alguien lo corrió una vez para generar los 15 clips
existentes, pero no quedó como herramienta reusable. Para meter bailes de
g1-moves (o de Javier directamente) hace falta escribir ese conversor una
sola vez, y después es genérico para cualquier clip nuevo, sea de g1-moves,
del pipeline privado de Javier, o de otro clip que traiga alguien más.

## Sobre las *policies* de g1-moves (los `.pt`/`.onnx`)

Son tentadoras porque evitan entrenar, pero **no es un simple "cargar y
listo"** — mismo riesgo que ya vimos con `stable` para el G1 de piernas: la
red fue entrenada con SU propia convención de observación de tracking (los
"160 obs" que documentan, que casi seguro no coinciden exactamente con los
`ref_motion_obs` que arma nuestro `G1DeepMimic.compute_observations()`). Un
dimension-match no garantiza un semantic-match. El camino confiable sigue
siendo el mismo patrón que ya usa este repo (`rugiar distill`): traer el
**movimiento** (dato, no la policy), y entrenar/destilar contra nuestra
propia convención de obs.

## Recomendación de orden (para las fases 2 y 3)

1. **Fase 2 (control, esta sesión)**: dejar el *mecanismo* listo — un campo
   extra en el catálogo de policies (robot/categoría/source) para poder
   distinguir "G1 piernas" vs "G1 cuerpo completo" vs futuras familias, y
   que la Family/Policy panel de la web lo muestre. Cargar `g1_motion_vis`
   con al menos un clip existente (de los 15 que ya están) para probar que
   se puede elegir family+policy y verlo correr en la web — sin bajar nada
   externo todavía, valida el mecanismo con lo que ya tenemos.
2. **Fase 3 (entrenamiento, después)**: escribir el conversor
   retarget-pkl+training-npz → formato `MotionLoader`, traer 1-2 clips
   concretos (de g1-moves o directo de Javier) como prueba, y recién ahí
   entrenar `g1_deepmimic` sobre ellos con el mismo criterio de escala que ya
   documentamos para caminar (Kaggle, `num_envs=4096`, miles de iteraciones,
   nunca confiar en el reward solo — mirarlo caminar/bailar).

## Fase 2 — lo que quedó armado (mecanismo mínimo)

Validado de punta a punta esta sesión, sin bajar nada externo todavía:

- **`--motion_file`** — nuevo flag en `rugiar train` (y en `web_train.py`/
  `TrainingManager.start()` por debajo) para elegir contra qué `.pkl` de
  `resources/reference_motion/` entrenar `g1_deepmimic`, en vez del clip
  hardcodeado en la config. Da error claro si se usa en una task sin
  `motion_file` (por ejemplo `g1`).
- **`category`** — campo nuevo, opcional y de solo cosmética, en
  `meta.json` → `TrainingManager.register_source()`/`discover_local_
  policies()` → `catalog()` → paneles de Fuse/Clone-from en `web/app.js`.
  No participa en ningún chequeo de compatibilidad (eso lo sigue haciendo
  `task`) — es para poder distinguir, dentro de una misma `task`, un
  policy que entrenamos nosotros de uno importado (de Javier, de
  g1-moves, etc). Se setea a mano en el `meta.json`, mismo patrón que ya
  usa `policies/stable/`.
- **`g1_deepmimic_smoke`** — primer policy jamás entrenado y registrado
  para `g1_deepmimic` (20 iteraciones, `num_envs=64`, clip
  `C3_-_run_stageii_genesis.pkl` vía `--motion_file`, `category:
  "g1-full-body-smoke"`). No es un buen mimic (mismo criterio de
  "Scale matters" del skill de `rugiar` — 20 iteraciones es solo para
  probar el cableado), pero confirma que la familia de 29 DOF completa
  bootea y corre en `rugiar_driver.py --task g1_deepmimic`.
- **Dos bugs preexistentes encontrados y arreglados** (nadie había
  entrenado+cargado `g1_deepmimic` de punta a punta antes de hoy):
  1. `web_train.py` no resolvía `train_cfg.runner.load_run` (quedaba en
     el sentinel `-1`) antes de exportar — crasheaba con
     `TypeError: unsupported operand type(s) for +: 'int' and 'str'`
     para cualquier task cuyo exportador cae en la rama por default de
     `helpers.py::PolicyExporter` (MLP no recurrente, sin rama dedicada
     en `play.py::export_policy`). `play.py` ya tenía el fix (vía
     `train_cfg.runner.resume = True` + `get_load_path()`); portado a
     `web_train.py` seteando `train_cfg.runner.load_run =
     os.path.basename(log_dir)` antes de exportar.
  2. `legged_gym/control/policy.py::load_policy_backend()` solo sabía
     distinguir dos formatos jit (explicit-state recurrente vs
     internal-state recurrente de Unitree) — un export MLP sin estado
     (como el de `g1_deepmimic`/`g1_motion_vis`) caía por default en
     `ExplicitStatePolicy`, que le pasaba `(obs, h, c)` a un `forward()`
     que solo acepta `obs`. Se agregó `StatelessPolicy` y la detección
     ahora primero mira la cantidad de argumentos del propio schema de
     `forward()` (4 = explicit-state; 2 = internal-state si tiene
     buffers `hidden_state`/`cell_state`, si no, stateless).

## Fase 3 — primer baile real de punta a punta

Con `--motion_file`/`category` ya andando, se probó el pipeline completo
contra un dato real (no uno de los 15 mocap existentes):

- **Fuente**: `exptech/g1-moves` en Hugging Face — CC-BY-4.0 — clip
  `dance/B_DadDance` (2509 frames, 60fps). Bajado directo de
  `datasets/exptech/g1-moves/resolve/main/dance/B_DadDance/retarget/
  B_DadDance.pkl` (root_pos/root_rot xyzw/dof_pos) +
  `.../training/B_DadDance.npz` (para el whitepaper/dataset card, no
  terminó haciendo falta — ver más abajo).
- **Compatibilidad confirmada con el dataset card**: el orden de 29 DOF
  de g1-moves (piernas → waist yaw/roll/pitch → brazos shoulder pitch/
  roll/yaw+elbow+wrist roll/pitch/yaw) es **idéntico** al de
  `G1Flat29DofCommonCfg.dof_names`, y su quaternion de root en el `.pkl`
  de retarget ya viene en **xyzw** — cero reordenamiento necesario. (El
  `.npz` de training usa **wxyz** para `body_quat_w` — distinto del
  `.pkl` — por eso el `.pkl` de retarget es la fuente correcta para
  `root_rot`, no el npz.)
- **El conversor ya existía** — `legged_gym/scripts/
  process_reference_motion.py` — no hizo falta escribir uno nuevo (la
  Fase 1 no lo había encontrado). Toma un `.pkl` "raw" (fps/root_pos/
  root_rot/dof_pos, `local_body_pos`/`link_body_list` opcionales) y
  genera el derivado por-simulador con `root_lin_vel`/`root_ang_vel`/
  `dof_vel` por diferencia finita y `key_body_pos_relative_to_base`
  calculado por **FK real contra nuestro propio robot** (stepea el
  sim con cada frame de `dof_pos`/`root_pos` y lee la posición real de
  los 19 key bodies) — mejor que intentar mapear a mano el orden de
  30 bodies de g1-moves, que el dataset card no documenta completo.
  Por eso el `.npz` de training terminó sin usarse — el conversor no
  lo necesita.
- **Uso**: el `.pkl` de retarget de g1-moves se copia tal cual (con
  `local_body_pos`/`link_body_list` en `None`) a
  `resources/reference_motion/unitree_g1/raw_run/g1moves_<clip>.pkl`,
  y se corre:
  ```bash
  export SIMULATOR=genesis
  python legged_gym/scripts/process_reference_motion.py \
      --task g1_deepmimic --headless --cpu \
      --motion_file unitree_g1/raw_run/g1moves_<clip>.pkl \
      --motion_out_dir unitree_g1/genesis_run
  ```
  Genera `unitree_g1/genesis_run/g1moves_<clip>_genesis.pkl`, mismo
  esquema exacto que los 15 clips ya existentes — usable directo con
  `--motion_file` en `rugiar train --task g1_deepmimic`.
- **Tres bugs preexistentes más, encontrados y arreglados** en este
  script (nunca se había corrido headless contra una task real):
  1. `g1_deepmimic.py::_init_buffers()` llamaba `MotionLoader(self.num_envs,
     self.dt, ...)` — pasaba el timestep (float) donde iba la cantidad de
     key bodies (int). Solo explota cuando el `.pkl` de entrada NO trae
     `key_body_pos_relative_to_base` (o sea, exactamente el caso de un
     clip nuevo sin procesar) — nadie lo había disparado antes. Arreglado
     a `len(self.simulator.key_body_indices)`.
  2. `process_reference_motion.py` llamaba a
     `env.simulator.draw_debug_vis(...)` en cada frame incondicionalmente
     — crashea bajo `--headless` (Genesis no tiene contexto de viewer).
     Guardado detrás de `if not env.headless`.
- **Resultado**: `g1_deepmimic_daddance_smoke` (20 iteraciones, smoke —
  no es un baile bueno todavía) entrenado contra el clip real de
  g1-moves, cargado en `rugiar_driver.py` junto con el smoke anterior, y
  confirmado el switch en vivo entre ambos sin caerse.

### El panorama más amplio en Hugging Face

No es solo `exptech/g1-moves` — buscando aparecieron varios datasets más
de movimiento retargeteado a G1 (útil para cuando se quiera escalar más
allá de un clip de prueba):

- **`openhe/g1-retargeted-motions`** — 174 secuencias (locomoción, danza,
  deportes, expresivo), retargeteadas desde SMPL vía el pipeline Mink.
- **`bones-studio/seed`** — 142.220 animaciones anotadas (lenguaje
  natural + segmentación temporal), en formato SOMA **y Unitree G1**.
- **`fleaven/Retargeted_AMASS_for_robotics`** y
  **`lvhaidong/LAFAN1_Retargeting_Dataset`** — AMASS/LAFAN1
  retargeteados a varios humanoides incluyendo G1.
- **`nvidia/Kimodo-G1-RP-v1`** — modelo (no dataset) entrenado sobre
  datos retargeteados a G1 de 34 joints (esqueleto distinto al nuestro
  de 29 — revisar antes de asumir compatibilidad directa).

Ninguno de estos se probó todavía — quedan como candidatos para escalar
Fase 3 más allá de un solo clip de prueba, mismo camino de conversión
(`process_reference_motion.py`) en principio, aunque habría que
confirmar la convención de DOF/quaternion de cada uno como se hizo acá
con g1-moves antes de asumir que calzan sin reordenar.

**Licencia**: `g1-moves` es CC-BY-4.0 — exige atribución. Si estos clips
terminan en un policy que se distribuye, hay que dejar constancia del
crédito a `exptech`/g1-moves en el mismo lugar donde se documenta la
procedencia (meta.json `note`, como ya se hizo acá, o en un doc central).

## El checkpoint real de Javier — bajado, pero incompatible tal cual

Javier nunca mandó un archivo por WhatsApp — todo lo que compartió fue el
link a su Kaggle notebook público. Se bajó directo con
`kaggle kernels output jvillalba007/unitree-rl-mimic` (credenciales ya
configuradas en esta máquina): el kernel clona y entrena sobre
`unitreerobotics/unitree_rl_mjlab` (664MB de repo completo con logs). De
ahí se rescataron dos checkpoints útiles, registrados en `policies/`:

- **`javier_mjlab_model_7000`** — `logs/rsl_rl/g1_tracking/2026-08-13_06-
  08-32/policy.onnx`, el export final de su run de 7000 iteraciones.
- **`javier_mjlab_dance1_subject2`** — `deploy/robots/g1/config/policy/
  mimic/dance1_subject2/exported/policy.onnx` (+ `.onnx.data`), un export
  curado para deploy de una policy de mimic de UN movimiento específico
  ("dance1_subject2") — no necesariamente relacionado con el clip
  `B_DadDance` de g1-moves ya convertido en este repo.

**Actualización 2026-08-18 — resuelto vía migración a mjlab, no vía
`g1_deepmimic`.** Lo de abajo describe el estado tal como se encontró en
esta sesión (correcto en su momento); ver `docs/mjlab_migration.md` para
el estado actual. Resumen: no se hizo compatible con `g1_deepmimic` (eso
seguía siendo un mismatch de dimensión real, 154 vs 1380, y sigue siendo
cierto — `g1_deepmimic` no cambió). En cambio, se migró el stack de motion
imitation completo a `mjlab` (opción 1 de abajo, pero contra `mjlab`
directamente, no contra `unitree_rl_mjlab` — ver §0 de
`docs/mjlab_migration.md`). Ambos checkpoints ahora cargan y corren en
loop cerrado contra `Rugiar-G1-Mimic`, la task propia de este repo
(`mjlab_tasks/`), confirmado con rollouts de 400 pasos
(`tests/test_javier_checkpoints_track.py`): `dance1_subject2` no cae
ninguna vez, error de tracking ~10cm; `model_7000` cae 6 veces (probó
contra un motion clip desconocido, no necesariamente `dance1_subject2` —
pregunta abierta para Javier). `meta.json` de ambos ya no dice
"incompatible as-is" — dice `"task": "Rugiar-G1-Mimic"`.

**No son plug-and-play con `g1_deepmimic`** (esto sigue siendo cierto).
Ambos onnx esperan una observación de **154 dims** (`onnxruntime` lo
confirma: `obs [1, 154]`) — la convención propia de `mjlab`, un solo
frame. Nuestro `g1_deepmimic` espera **1380 dims** (`frame_stack=5 × (151
propioceptivo + 125 de referencia de movimiento)`, ver
`g1_deepmimic_config.py`). Es un mismatch de **dimensión**, no solo
semántico como con `stable` — ni `rugiar_driver.py --task g1_deepmimic`
ni `rugiar distill` (que valida dimensiones antes de aceptar un teacher)
lo van a aceptar tal cual, y eso no cambió.

Dos caminos reales para aprovecharlos, ninguno trivial (**camino 1
tomado, ver arriba**):
1. ~~Registrar una task nueva con la convención de obs de 154 dims de
   `unitree_rl_mjlab`~~ — hecho, pero contra `mjlab` directamente
   (`mjlab_tasks/`), no contra el fork de `unitree_rl_mjlab` (ver §0 de
   `docs/mjlab_migration.md` para por qué el fork no era el target
   correcto).
2. Usarlos solo como fuente de movimiento, no de pesos — seguía siendo
   una opción, no hizo falta: el camino 1 resultó viable directamente.

Se descartó el clon completo de `unitree_rl_mjlab` (664MB) después de
extraer estos dos archivos — si hace falta algo más de ahí (otros
`model_N.pt` intermedios, otros exports de `deploy/robots/g1/config/
policy/`), hay que volver a bajarlo con el mismo comando.

## `unitree_rl_gym` vs `unitree_rl_mjlab` — de dónde sale realmente cada uno

Esto no depende de lo que diga nadie en el chat — el código y la actividad
de ambos repos (ambos de `unitreerobotics`, no forks de terceros) ya lo
muestran:

| | `unitree_rl_gym` (base de este fork/`rugiar`) | `unitree_rl_mjlab` |
|---|---|---|
| Último push | 2025-07-25 (~1 año congelado) | 2026-04-13 (activo) |
| Issues abiertos | 59 | 42 |
| Motion imitation nativo | **No existe** (`legged_gym/envs/` solo tiene base/g1/go2/h1/h1_2, ningún mimic) | **Sí, de primera clase** — conversión CSV→NPZ, tasks `Unitree-G1-Tracking-No-State-Estimation`, integración con BeyondMimic |

Conclusión: `g1_deepmimic`/`MotionLoader` en este repo fueron construidos
**desde cero** por el equipo porque `unitree_rl_gym` nunca tuvo nada
equivalente — no había nada de upstream para copiar. Javier no eligió
`mjlab` por preferencia personal de entorno: es el proyecto que Unitree
mismo sigue manteniendo y que ya trae resuelto justo lo que a nuestra
base le faltaba. Mismo patrón que Isaac Gym → Isaac Lab (que Ramiro señaló
para el toolchain de NVIDIA), solo que nadie lo había dicho en voz alta
para este otro par.

**Corrección**: `dance1_subject2` (el checkpoint que bajamos) es un
motion de EJEMPLO que viene con el propio repo `unitree_rl_mjlab`
(`src/assets/motions/g1/dance1_subject2.csv` en su tutorial de motion
imitation) — no algo que Javier haya retargeteado él mismo. No hay que
confundirlo con trabajo original de Javier.

**Decisión pendiente, no resuelta acá**: si conviene migrar el stack
completo (`legged_gym`/Genesis/`g1_deepmimic`) a `mjlab` es una decisión
de arquitectura grande — MuJoCo en vez de Genesis, otra API de obs/task,
toca todo. Vale la pena tenerlo en el radar del equipo, pero no se decide
en este documento ni se ejecuta sin una conversación explícita primero.
Mientras tanto, el camino más chico sigue siendo el ya documentado arriba:
usar `mjlab`/g1-moves solo como fuente de DATO de movimiento, no de pesos.

## Preguntas abiertas para Javier

- ¿Tiene el `model_7000` del Kaggle a mano para subirlo directo, o hay que
  bajarlo del notebook?
- Su pipeline privado (LLM→video→reconstrucción 3D→retargeting) — ¿en qué
  formato entrega el resultado final? Si ya es compatible con el `.pkl` de
  Retarget de g1-moves, el conversor de la Fase 3 le sirve directo a él
  también, no solo para g1-moves.
