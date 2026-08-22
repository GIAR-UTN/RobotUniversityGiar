# Handoffs — archivo histórico de sesiones de desarrollo

Estos son bitácoras de sesiones de desarrollo pasadas (escritas para que la sesión
siguiente retomara el trabajo), no documentación de onboarding. Si buscás cómo y
dónde seguir trabajando en el sistema hoy, empezá por `docs/compute_backends.md`
y `legged_gym/control/ARCHITECTURE.md` — no por acá.

**Los 4 están cerrados.** Ninguno tiene trabajo pendiente activo. Verificá siempre
contra el código antes de confiar en el "STATUS" que cada archivo se autoasigna —
uno de estos quedó desactualizado (ver abajo) y solo se detectó leyendo el código,
no el handoff.

| Archivo | Qué documenta | Estado real | Reemplazado / confirmado por |
|---|---|---|---|
| `HANDOFF_dagger_distillation.md` | Diagnóstico de por qué behavior-cloning de un paso no alcanza para clonar una policy (covariate shift); proponía `dagger` como siguiente paso. | **Cerrado.** El propio archivo dice "not implemented yet" pero eso quedó viejo: `dagger_train()` se implementó después, en `c3356e3`. | `legged_gym/control/distillation.py:316` (`dagger_train`), wireado en `web_distill.py`, tests en `tests/test_distillation.py`. |
| `HANDOFF_distillation_hidden_state_bug.md` | Bug de LSTM (hidden state no se reseteaba en boundaries de episodio) durante distillation. | **Cerrado.** Fix confirmado y en el código. | `legged_gym/control/distillation.py` (`collect_rollout`), `tests/test_distillation.py`. |
| `HANDOFF_mimic_motion_library_ux.md` | Selector de clips, preview sin policy, panel Create Policy mjlab-aware, backend de training mjlab real. | **Cerrado**, los 4 ítems validados en su momento. | `legged_gym/control/backends/`, `legged_gym/control/ARCHITECTURE.md` §1b, `docs/compute_backends.md`. |
| `HANDOFF_mjlab_migration.md` | Narrativa de la decisión de migrar de `unitree_rl_gym`/Genesis a `unitree_rl_mjlab` para motion imitation. | **Superseded.** El propio archivo dice que el plan vivo está en otro doc. | `docs/mjlab_migration.md`. |

Si estás por integrar o continuar trabajo de motion imitation, el doc que te
corresponde es `docs/mimic_motion_team_handoff.en.md` / `.es.md`, no ninguno de
estos cuatro.
