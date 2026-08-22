# Motion imitation — handoff al equipo de movimiento

> **Documento hermanado.** Este archivo tiene una versión hermana en inglés:
> [`motion_team_handoff.en.md`](motion_team_handoff.en.md). Los dos tienen que
> mantenerse sincronizados — si cambiás uno, cambiá el otro.

**Para quién es:** el colaborador que desarrolló la parte de motion imitation /
mimic motion (retargeting de bailes, entrenamiento sobre `unitree_rl_mjlab`,
los checkpoints de Kaggle) en otro repo, y ahora necesita seguir trabajando
*adentro* de RUgiar.

**Para qué:** orientación, no historia. Dónde quedó cada cosa, qué está
realmente validado, qué se simplificó y — sobre todo — dónde y cómo seguir
construyendo. Para el relato de cómo llegamos acá están
[`docs/mjlab_migration.md`](mjlab_migration.md) y
[`docs/motion_imitation_integration.md`](motion_imitation_integration.md); no
hace falta rehacer esa investigación.

---

## 1. La versión de 60 segundos

Tu trabajo entró por el PR #3 (`g1-fullbody-motion-imitation`, mergeado como
`0d65112`). La integración tomó una decisión arquitectónica grande:

**Motion imitation en RUgiar corre ahora sobre `mjlab`, no sobre el stack
Genesis original del repo.** No forkeamos `unitree_rl_mjlab`: dependemos de
`mjlab` mismo (PyPI, pineado en `mjlab==1.6.0`) y registramos **nuestra propia
task, `Rugiar-G1-Mimic`**, que llama al
`unitree_g1_flat_tracking_env_cfg()` de mjlab sin modificar. O sea: el contrato
de observación de 154 dims contra el que se entrenaron tus checkpoints queda
**idéntico bit a bit** — lo único nuestro son `task_id` y `experiment_name`.

Consecuencias para tener presentes:

- Tus dos checkpoints ONNX cargan y corren **en lazo cerrado** acá, sin tocar
  nada.
- Todo lo que los rodea (web de control, protocolo WebSocket, catálogo de
  policies, CLI `rugiar`, jobs de entrenamiento, gráficos de reward) es la
  maquinaria que el repo ya tenía — motion imitation es una tercera "familia",
  no un subsistema aparte.
- El camino Genesis viejo (`g1_deepmimic`, `MotionLoader`, `g1_motion_vis`)
  sigue existiendo pero está **deprecado para esta familia**. No inviertas ahí
  — ver §7.

---

## 2. Mapa — dónde está cada cosa

| Qué | Ruta |
|---|---|
| **Registro de nuestra task mjlab** (`Rugiar-G1-Mimic`) | `mjlab_tasks/__init__.py`, `mjlab_tasks/tracking/` |
| Config del env (wrapper fino sobre el de mjlab) | `mjlab_tasks/tracking/g1_env_cfg.py` |
| Hiperparámetros PPO / runner | `mjlab_tasks/tracking/rl_cfg.py` |
| **Conversor de motion** `.pkl` crudo → `.npz` mjlab | `legged_gym/scripts/process_reference_motion_mjlab.py` |
| **Entrypoint de entrenamiento** (mjlab) | `legged_gym/scripts/mjlab_train.py` |
| Dispatch de entrenamiento / registry de backends | `legged_gym/control/training.py` (`BACKENDS`, `training_backend_for_task()`, `resolve_training_backend()`) |
| **Driver** (sim + servidor de control, mjlab) | `legged_gym/scripts/rugiar_driver_mjlab.py` |
| Adapter de backend (estado del robot ↔ motor de control) | `legged_gym/control/mjlab_adapter.py` |
| RPCs de listado / cambio de clip | `legged_gym/control/service.py` (`list_motions()`, `switch_motion()`, `motion_clip_rows()`) |
| Web UI (panel Motion, Create Policy) | `web/index.html`, `web/app.js` |
| CLI (`rugiar train`, `rugiar drive`) | `legged_gym/cli/rugiar.py`; atajos en el `Makefile` |
| **Clips de motion, formato mjlab** | `resources/reference_motion/unitree_g1/mjlab_run/*.npz` |
| Clips de motion, formato fuente crudo | `resources/reference_motion/unitree_g1/raw_run/*.pkl` |
| **Checkpoints** | `policies/<nombre>/` (`checkpoint.onnx` / `checkpoint.pt` + `meta.json`) |
| Referencias upstream de solo lectura | `third_party/unitree_rl_mjlab/*.reference` |
| Contratos de diseño / planes | `docs/mjlab_migration.md`, `docs/mjlab_training_contract.md` |
| Tests | `tests/test_mjlab_*.py`, `tests/test_javier_checkpoints_track.py`, `tests/test_process_reference_motion_mjlab.py` |

### Los dos intérpretes (a todos les pasa una vez)

Hay **dos virtualenvs a propósito**:

- `.venv` — stack Genesis (`SIMULATOR=genesis`), tasks de locomoción.
- `.venv-mjlab` — stack mjlab (`SIMULATOR=mjlab`), motion tracking.

No se pueden unificar: el repo vendorea un paquete top-level `rsl_rl/` que
tapa al `rsl-rl-lib` de PyPI, y el extra `genesis` pinea `mujoco==3.10.0`
mientras mjlab necesita `~=3.11.0`. Ver `docs/mjlab_migration.md` **R1**.
Cualquier script nuevo de mjlab tiene que reordenar `sys.path` (raíz del repo
**al final**, no removida) antes de importar `rsl_rl` / `mjlab_tasks` — copiá
el encabezado de `legged_gym/scripts/mjlab_train.py` tal cual;
`tests/conftest.py` hace lo mismo para toda la sesión de tests.

Nunca necesitás `.venv-mjlab/bin/rugiar` — no existe. `rugiar` corre desde
`.venv` y despacha solo al intérprete correcto.

---

## 3. Qué está validado (corridas reales, no chequeos de import)

- **`policies/javier_mjlab_dance1_subject2`** — la policy mjlab de referencia.
  Rollout en lazo cerrado de 400 pasos contra su propio `dance1_subject2.npz`:
  **0 caídas**, error medio de posición de bodies ~0,07–0,10 m. Usala como
  baseline de calidad. Cubierto por `tests/test_javier_checkpoints_track.py`.
- **`policies/javier_mjlab_model_7000`** — carga y corre, pero **6 caídas** en
  los mismos 400 pasos contra `dance1_subject2` (~0,18 m de error). Casi
  seguro porque se entrenó contra otro clip, desconocido (ver §8, pregunta
  abierta). **No es baseline de calidad.**
- **La tabla de observación de 154 dims** está especificada término a término y
  asertada en un test (`tests/test_mjlab_env_smoke.py`), así que un bump de
  versión de mjlab no puede reordenarla en silencio bajo tus checkpoints. Ver
  `docs/mjlab_migration.md` §2.
- **Conversión de motion** — `g1moves_B_DadDance.pkl` (2509 frames @ 60 fps) →
  `g1moves_B_DadDance.npz` (2090 @ 50 fps), carga en un env de tracking real y
  stepea sin caerse. `tests/test_process_reference_motion_mjlab.py`.
- **Manejo desde la web de control** — `Rugiar-G1-Mimic` corre sobre el mismo
  protocolo WebSocket que cualquier task de Genesis: switch de policy en vivo,
  pause/resume/restart, telemetría, overlay ghost de la referencia, selector de
  clips.
- **Entrenamiento de punta a punta, local** — una corrida corta real vía
  `TrainingManager.start()` llegó a `status="done"`, exportó un ONNX stateless,
  y la policy resultante fue autodescubierta y **hot-loadeada en una sesión
  viva sin relanzar el proceso** (`tests/test_mjlab_training_hotload.py`).
- **Sesión damping-only** — el driver sobrevive a no tener ninguna policy
  local, así que podés previsualizar el ghost de un clip con el robot
  simplemente sosteniendo una pose (`tests/test_mjlab_damping_only_session.py`).

---

## 4. Qué NO está hecho / se simplificó

Conviene tener esto explícito antes de planificar:

1. **Nunca se ejecutó un entrenamiento en GPU por el camino mjlab.** Solo
   CPU/mujoco-warp. `--device cuda:0` existe en `mjlab_train.py` pero está sin
   probar. Es el paso no verificado más grande de la migración (**R5**). macOS
   es evaluation-only según la propia doc de mjlab.
2. **El backend Kaggle está explícitamente rechazado para tasks mjlab.** Su
   bootstrap es específico de IsaacGym. `rugiar train --backend kaggle` da error
   en una task mjlab en vez de lanzar algo condenado. Las tasks Genesis no se
   ven afectadas. **No existe todavía un camino real de GPU en la nube para
   entrenar mimic.**
3. **Un movimiento por policy.** La task de tracking de mjlab carga un solo
   `motion_file` al construir el env — sin resampling, sin policy multi-clip.
   Cambiar de baile implica cambiar de policy *y* relanzar el proceso (por eso
   `switch_motion()` relanza en vez de intercambiar en caliente). **R8**.
4. **Todavía no existe ninguna policy mimic self-trained de calidad real.** Lo
   que hay en `policies/` para esta familia es: tus dos checkpoints importados,
   más policies de smoke (`g1_deepmimic_smoke`, `g1_deepmimic_daddance_smoke` —
   20 iteraciones, era Genesis, pruebas de cableado, *no* buenos mimics).
5. **Solo hay 2 clips en formato mjlab**: `dance1_subject2.npz` (fixture
   vendoreado de upstream) y `g1moves_B_DadDance.npz`. Los otros 15 clips de
   `raw_run/` son mocap de locomoción estilo AMASS y solo se convirtieron al
   formato Genesis.
6. **Los checkpoints `policy_154/*.onnx` de g1-moves NO son drop-in.** Sus dims
   de obs/action coinciden exactamente con las nuestras, pero toman una segunda
   entrada, `time_step`, y nuestro `load_onnx_backend()` trataría cualquier
   entrada después de `obs` como estado recurrente y realimentaría el tensor
   equivocado — mal en silencio, sin crash. Hace falta un backend real
   `OnnxPhaseConditionedPolicy` antes de registrar ninguno. Detalle en
   `HANDOFF_mimic_motion_library_ux.md`.
7. **El deploy en robot real está fuera de alcance.** El camino real de Unitree
   es una FSM en C++ (`third_party/unitree_rl_mjlab/State_Mimic.cpp.reference`),
   no nuestro `RealAdapter` de Python. Además las ganancias difieren entre
   mjlab, nuestras configs Genesis y el YAML de deploy — tres sets tuneados
   independientemente (**R6**/**R7**, §6 de la migración).
8. **Los catálogos de policies Genesis y mjlab son disjuntos para siempre.** No
   se fusiona ni se destila entre stacks — física distinta, checkpoints
   estructuralmente distintos (**R4**). Además mujoco-warp no es determinista:
   nunca escribas un test que asuma rollouts bit-exactos, usá umbrales con
   margen.

---

## 5. Cómo hacer las cuatro cosas que vas a querer hacer

### 5.1 Agregar un clip de motion nuevo

El pipeline es: **`.pkl` fuente → `raw_run/` → conversor → `mjlab_run/*.npz`**.

1. Conseguí un `.pkl` de etapa retarget con `fps`, `root_pos (N,3)`,
   `root_rot (N,4)` en **xyzw**, `dof_pos (N,29)`. Un
   `retarget/<clip>.pkl` de `exptech/g1-moves` sirve tal cual — su orden de 29
   DOF está confirmado idéntico al nuestro y su quaternion de root ya viene en
   xyzw. (El `.npz` de la etapa training usa **wxyz** — no saques `root_rot` de
   ahí.)
2. Ponelo en
   `resources/reference_motion/unitree_g1/raw_run/<clip>.pkl` (con
   `local_body_pos` / `link_body_list` en `None` si no los tenés).
3. Convertí:

   ```bash
   CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python \
       legged_gym/scripts/process_reference_motion_mjlab.py \
       --motion_file unitree_g1/raw_run/<clip>.pkl \
       --motion_out_dir unitree_g1/mjlab_run
   ```

4. **Miralo antes de entrenar contra él.** Se puede previsualizar el ghost del
   clip sin necesitar ninguna policy:

   ```bash
   rugiar drive mjlab --motion_file resources/reference_motion/unitree_g1/mjlab_run/<clip>.npz
   ```

   O elegilo desde el panel **Motion** de la web de control (lista todos los
   `.npz` de ese directorio y marca cuáles ya tienen policy — los clips sin
   policy deliberadamente *no* se deshabilitan).

Si tu pipeline privado (LLM → video → reconstrucción 3D → retargeting) puede
emitir ese mismo formato de retarget-`.pkl`, el paso 1 desaparece y el pipeline
entero te sirve gratis. Si emite otra cosa, el lugar correcto para agregar un
adapter de entrada es `process_reference_motion_mjlab.py` — cambiás el lado de
la entrada y dejás la matemática de MotionLoader/`run_sim` intacta.

**Licencias:** `g1-moves` es CC-BY-4.0 y exige atribución. Dejá constancia de
la fuente en el campo `note` del `meta.json` de la policy derivada, igual que
las entradas que ya están.

### 5.2 Entrenar una policy contra un clip

`rugiar train` autodetecta el backend a partir de la task — mismos flags en
ambos casos, nunca elegís el intérprete:

```bash
rugiar train --list_motions --task Rugiar-G1-Mimic     # qué hay, qué ya tiene policy
rugiar train --task Rugiar-G1-Mimic --list_reward_scales

rugiar train --task Rugiar-G1-Mimic --name mimic_dance \
    --num_envs 4096 --max_iterations 3000 \
    --motion_file resources/reference_motion/unitree_g1/mjlab_run/<clip>.npz
```

Notas:

- `--motion_file` es **obligatorio** en una task de tracking; la CLI da error
  de entrada. Tiene que ser el `.npz` de `mjlab_run/`, no el `.pkl`.
- Los flags de solo-Genesis (`--cmd_vx_range`, `--push_interval_s`, …) se
  rechazan — una task de tracking no tiene comando de velocidad.
- Los **9 términos de reward** de esta task (este es el vocabulario; no reuses
  `tracking_lin_vel` etc. de Genesis):
  `motion_global_root_pos` (0.5), `motion_global_root_ori` (0.5),
  `motion_body_pos` (1.0), `motion_body_ori` (1.0), `motion_body_lin_vel` (1.0),
  `motion_body_ang_vel` (1.0), `action_rate_l2` (−0.1), `joint_limit` (−10.0),
  `self_collisions` (−10.0). Se sobreescriben con
  `--reward_scale motion_body_pos 2.0`.
- El mismo job se puede lanzar desde el panel **Create Policy** de la web, que
  muestra campos con forma mjlab (selector de clip, sin envolvente de
  velocidad).
- Los ETA están calibrados por backend — los históricos de throughput de mjlab
  y Genesis nunca se mezclan.
- **Nunca confíes solo en la curva de reward.** Cargá el `checkpoint.onnx`
  resultante en el driver y miralo contra el overlay ghost.

### 5.3 Registrar un checkpoint entrenado afuera

Creá `policies/<nombre>/` con el ONNX (y el `.onnx.data` si el export viene
partido) más un `meta.json`:

```json
{
  "task": "Rugiar-G1-Mimic",
  "trained_via": "external-import",
  "simulator": "mjlab",
  "category": "g1-mjlab-mimic",
  "motion_file": "resources/reference_motion/unitree_g1/mjlab_run/<clip>.npz",
  "note": "procedencia: de dónde vino, licencia, contra qué se entrenó, cómo se validó"
}
```

`task` es lo que maneja los chequeos de compatibilidad; `category` es cosmético
(distingue importado de self-trained en la UI); `motion_file` es lo que hace
que el badge `has_policy` del panel Motion sea exacto en vez de una heurística
de nombre. Las policies se autodescubren — no hay archivo de registro para
editar. Probalo con un rollout real antes de decir que anda (mismo patrón que
`tests/test_javier_checkpoints_track.py`).

### 5.4 Tocar la task en sí (rewards, DR, hiperparámetros)

- Pesos de reward / rangos de domain randomization / cualquier cosa del env:
  mutá la cfg que devuelve `rugiar_g1_mimic_env_cfg()` en
  `mjlab_tasks/tracking/g1_env_cfg.py`. **Agregá un delta, no copies adentro la
  config de mjlab** — ese es justamente el punto de llamar a su factory, para
  que los fixes de upstream sigan llegando.
  ⚠️ Cualquier cambio acá cambia el contrato de observación/reward que tus
  checkpoints existentes asumen. Si tocás los términos de *observación*, los
  checkpoints de Javier dejan de ser válidos, y
  `tests/test_mjlab_env_smoke.py` te lo va a avisar (para eso está).
- Hiperparámetros de PPO, `max_iterations`, `save_interval`,
  `experiment_name`: `mjlab_tasks/tracking/rl_cfg.py`.
- Comportamiento del worker de entrenamiento (chunking, archivos de
  progress/result, forma del export ONNX):
  `legged_gym/scripts/mjlab_train.py`, especificado en
  `docs/mjlab_training_contract.md`. Ojo que el export usa a propósito el
  `export_policy_to_onnx` de la clase padre para obtener un grafo limpio de
  1 entrada / 1 salida — el auto-export del runner de tracking es de
  2 entradas / 7 salidas y nuestro loader lo manejaría mal.
- Agregar un backend de entrenamiento entero (una máquina con GPU, una segunda
  nube): agregá un descriptor `TrainingBackend(...)` a `BACKENDS` en
  `legged_gym/control/training.py` — sin tocar `start()`.

---

## 6. El día a día

```bash
make drive-mjlab                       # web de control, task Rugiar-G1-Mimic
make drive-genesis                     # el lado de locomoción, para contrastar
rugiar drive mjlab --task Rugiar-G1-Mimic --motion_file <npz>
rugiar drive mjlab --headless          # smoke test scripteado, sin viewer
```

Web de control en `:9017`, viewer viser crudo en `:9006` por defecto. El
launcher frena primero lo que ya esté en el puerto de control — una sesión por
puerto, nunca levantes una segunda al lado.

Tests:

```bash
SIMULATOR=genesis .venv/bin/python -m pytest tests/ -q
SIMULATOR=mjlab CUDA_VISIBLE_DEVICES="" .venv-mjlab/bin/python -m pytest tests/ -q
```

El prefijo `SIMULATOR=` es obligatorio en ambos casos
(`legged_gym/__init__.py` lo exige). Corré **las dos** suites — un cambio en
`legged_gym/control/` afecta a los dos stacks.

---

## 7. Dónde *no* invertir esfuerzo

- **`g1_deepmimic` / `g1_motion_vis` / `legged_gym/utils/motion_loader.py`** —
  el camino de motion imitation de la era Genesis. Funciona, y su observación
  de 1380 dims (`frame_stack=5 × (151 + 125)`) es fundamentalmente incompatible
  con el contrato de 154 dims de mjlab. Está deprecado, no borrado, con una
  decisión explícita de mantener-o-matar con fecha **2026-11-16**
  (`docs/mjlab_migration.md` §7). Todo lo nuevo de mimic va a mjlab.
- **Forkear / depender de `unitree_rl_mjlab`.** Deliberadamente no lo hacemos.
  Su task de tracking es una copia casi textual de la propia de mjlab, pinea
  versiones viejas y no es instalable con pip (paquete top-level literalmente
  llamado `src`). Vendoreamos exactamente tres archivos de referencia de solo
  lectura en `third_party/unitree_rl_mjlab/`; traé más de ahí solo cuando lo
  necesites, como referencia. Fundamento: `docs/mjlab_migration.md` §0.
- **Construir un panel "puente" entre las familias Genesis y mjlab.**
  Rechazado por diseño — mjlab es una tercera familia en el panel Family que ya
  existe, sin puente.
- **Subir `mjlab` de 1.6.0 a la ligera.** El pin protege el contrato de
  observación de tus checkpoints. Si lo subís, corré primero
  `tests/test_mjlab_env_smoke.py` y `tests/test_javier_checkpoints_track.py`;
  son el canario.

---

## 8. Preguntas abiertas para vos

1. **¿Contra qué clip de motion se entrenó `model_7000`?** Tu directorio de
   logs de Kaggle (`g1_tracking/2026-08-13_06-08-32`) no lo registra, y es la
   explicación más probable de sus 6 caídas contra `dance1_subject2` (**R3**).
2. **¿En qué formato entrega tu pipeline privado (LLM → video → reconstrucción
   3D → retargeting)?** Si ya es el formato retarget-`.pkl` (fps, root_pos,
   root_rot xyzw, dof_pos ×29), el conversor de §5.1 te sirve sin cambios y no
   hay nada que construir. Si no, decinos la forma y el adapter de entrada es
   un agregado chico y de una sola vez.
3. **¿Tenés checkpoints intermedios `model_N.pt`, u otros exports de
   `deploy/robots/g1/config/policy/` que valga la pena importar?** Descartamos
   el clon de 664 MB de `unitree_rl_mjlab` después de extraer los dos ONNX;
   podemos volver a bajarlo, pero si los tenés a mano es más rápido.
4. **¿Tenés acceso a una máquina con GPU NVIDIA para entrenar?** Ese es el gap
   bloqueante (§4.1/§4.2) — todo lo demás está listo y validado en CPU.

---

## 9. Próximos pasos sugeridos, en orden

1. **Responder §8.1** (contra qué clip apunta `model_7000`) — barato, y
   desbloquea la interpretación del único dato de calidad que tenemos además de
   `dance1_subject2`.
2. **Meter una corrida de entrenamiento en GPU** por
   `rugiar train --task Rugiar-G1-Mimic` a escala real (`num_envs=4096`, miles
   de iteraciones) contra `g1moves_B_DadDance.npz`. Eso produce la **primera
   policy mimic self-trained genuinamente buena** del repo y de paso cierra el
   riesgo más grande sin probar de toda la migración.
3. **Importar clips en lote desde `exptech/g1-moves`** (~61 clips; solo hace
   falta `retarget/<clip>.pkl`, ~230 MB de archivos realmente útiles) con el
   conversor de §5.1, para que el panel Motion sea una biblioteca de verdad y
   no dos entradas.
4. **Si además querés sus policies preentrenadas**, primero arreglá el manejo
   de la entrada `time_step` (§4.6) — escribí el backend ONNX
   phase-conditioned, verificalo contra un rollout real, y *recién ahí*
   registrá alguna.
5. **Conectar la salida de tu propio pipeline de retargeting** a `raw_run/`
   para que los bailes nuevos entren sin paso manual.
