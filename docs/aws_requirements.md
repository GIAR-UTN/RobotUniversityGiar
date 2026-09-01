# Requerimientos de infraestructura — GIAR / RUgiar

**Para:** solicitud de créditos y GPU en AWS
**Fecha:** 2026-08-25
**Alcance:** hosting de la plataforma RUgiar (entrenamiento, simulador web, y la app web con login multiusuario)

> **Nota de idioma:** los docs de este repo son en inglés por defecto. Este es una excepción explícita — va dirigido al director del GIAR y a AWS — ver "Documentation language" en `AGENTS.md`.

---

## 1. Qué es la plataforma, en tres partes

La app tiene tres cargas de trabajo con perfiles de costo muy distintos. Tratarlas como una sola "instancia grande" sobrestima el gasto — casi todo el costo real está concentrado en una sola de las tres.

| Parte | Qué hace | Perfil de cómputo | Costo relativo |
|---|---|---|---|
| **A. Entrenamiento (training)** | Corre RL (PPO) para entrenar/afinar policies de locomoción y motion-tracking | GPU intensivo, jobs cortos y en ráfaga (minutos–~1h) | 🔴 Alto — es donde va casi todo el gasto |
| **B. Simulador en vivo (viewer)** | Sirve la visualización 3D del robot corriendo una policy ya entrenada (inferencia, no entrenamiento) | Liviano, CPU alcanza, sin GPU | 🟢 Bajo |
| **C. Web app (frontend + backend)** | Sitio, login, multiusuario, catálogo de policies, compartir/publicar | Web estándar: API + DB + estático | 🟢 Bajo |

Hoy A corre local (CPU en laptops) y en Kaggle (GPU gratuita, cola compartida, sin control de prioridad). B y C corren local vía Docker Compose. Nada de esto está hosteado en la nube todavía — este documento es la base para migrar.

---

## 2. Parte A — Entrenamiento (la que necesita GPU)

- Cada training job es una corrida PPO de un modelo pequeño (policy de locomoción o motion-tracking), no un LLM: son minutos a ~1 hora en una GPU consumer/datacenter chica (clase T4/A10G), no se necesita una GPU de gama alta tipo A100/H100.
- Referencia real: el free tier de Kaggle (GPU Tesla P100, Pascal) ya es suficiente para entrenar — o sea, el piso de hardware requerido es bajo.
- Hoy hay un backend NVIDIA **local** implementado (`rugiar train --backend local-nvidia`, GPU CUDA de la máquina — ver `docs/compute_backends.md`); el backend NVIDIA en la **nube** (`nvidia-cloud`) sigue siendo placeholder, reservado para esta migración.
- **Punto crítico de diseño: el entrenamiento debe pasar por una cola con concurrencia limitada** ("semáforo"), no ejecutarse indiscriminadamente. Esto es tanto una decisión de costo como de seguridad de cuota — ver §4.

## 3. Parte B — Simulador (viewer en la web)

- Corre inferencia (forward-only) sobre un checkpoint ya entrenado — no reentrena nada. Hoy demostrado corriendo 100% en CPU, incluso en una Mac sin GPU.
- Cada sesión de usuario viendo el robot caminar necesita ~1-2 vCPU y poca RAM (bajo GB). No requiere GPU.
- Escala horizontalmente sin fricción: son contenedores livianos, uno por sesión activa o por policy en demo.

## 4. Parte C — Web app (login, multiusuario, futuro)

- Próximo hito: login + cuentas + que cada usuarie pueda crear/guardar/compartir sus propias policies.
- Necesita: base de datos (usuarios, catálogo de policies, metadata de runs), storage de archivos (los `.pt`/`.onnx` de las policies, del orden de decenas de MB cada uno), autenticación, API, frontend estático.
- Es la parte más barata y más estándar de las tres — cualquier stack web convencional en AWS la cubre sin necesidad de nada especial.

---

## 5. El problema a resolver: 200 personas, ¿cuánta GPU en simultáneo?

El escenario de referencia es una innovaton con ~200 participantes queriendo entrenar su propia policy. Si todos entrenan a la vez sin control, el pico de demanda de GPU es indefendible en costo y en cupo de servicio de AWS. Estas son las opciones, de más cara/no-viable a la recomendada:

| Opción | Descripción | GPUs simultáneas | Viabilidad |
|---|---|---|---|
| **Sin límite** | Los 200 entrenan al mismo tiempo, sin cola | ~200 | ❌ No viable — ni el cupo de servicio de AWS lo permite por defecto, ni el costo se justifica para jobs de minutos |
| **Mitad simultánea** | Hasta 100 en paralelo | ~100 | ❌ Sigue siendo desproporcionado para la duración real de un job |
| **Un slot único (cola estricta)** | Un solo training corriendo, todos los demás esperan turno | 1 | ⚠️ Viable en costo, pero con 200 personas y jobs de ~20-40 min cada uno, la cola total ronda 70-130 horas — demasiado lento para un evento de un día |
| **Pool acotado con cola (semáforo)** | N jobs en paralelo (ej. 6-8), el resto espera turno en cola automática, con límite de 1 job activo por usuario | 6-8 | ✅ **Recomendada** — balancea costo, cupo de AWS y tiempo total del evento |

**Cálculo de referencia (Opción recomendada):** 200 personas × 1 policy × ~30 min promedio = ~100 GPU-hora de trabajo total. Con un pool autoescalado de 6-8 GPUs corriendo en paralelo durante una ventana de evento de ~12-16 horas, el evento entero se cubre cómodo, incluso dejando margen para reintentos.

**Controles a implementar junto con esto (no son solo un límite de infraestructura, son producto):**
- Un job activo por usuario a la vez (no se puede encolar 5 jobs propios).
- Límite de duración/iteraciones por job (evita que alguien deje un entrenamiento corriendo indefinidamente).
- Cola visible (posición, tiempo estimado) para que la espera no se sienta como que "se colgó".
- Autoscaling del pool de GPU: sube instancias cuando hay cola, baja a 0 cuando no hay demanda (esto es lo que evita pagar por GPU ociosa fuera de eventos).

---

## 6. Propuesta de arquitectura en AWS

| Componente | Servicio AWS sugerido | Notas |
|---|---|---|
| Workers de entrenamiento (Parte A) | EC2 `g6.4xlarge`/`g6.2xlarge` o `g5.4xlarge`/`g5.2xlarge` (NVIDIA L4/A10G, 1 GPU c/u) en un Auto Scaling Group, 0→8 instancias, o Spot Fleet con fallback on-demand | Spot para bajar costo — los jobs son cortos y tolerantes a interrupción con checkpointing |
| Cola de jobs | SQS + un scheduler simple (o AWS Batch, que ya trae cola + autoscaling de GPU integrado) | AWS Batch es probablemente el camino de menor esfuerzo para implementar el "semáforo" del §5 |
| Simulador/viewer (Parte B) | ECS Fargate (contenedores livianos, sin GPU), autoescalado por sesión | Sin necesidad de GPU, costo marginal |
| Web app (Parte C) | Frontend: Amplify Hosting o S3+CloudFront. Backend/API: ECS Fargate o Lambda. | Estándar |
| Base de datos | RDS PostgreSQL (instancia chica, ej. `db.t4g.micro`/`small`) | Usuarios, catálogo de policies, metadata |
| Storage de policies/checkpoints | S3 | Archivos chicos (decenas de MB), tráfico bajo |
| Auth | Cognito (o Auth0 si ya lo vienen usando) | Login + multiusuario |

---

## 7. Lo que le pedimos a AWS

1. **Cupo de servicio (service quota) para instancias GPU** en la familia `g6`/`g5` (NVIDIA L4/A10G): al menos **8 GPUs concurrentes** en la región elegida (hoy el default de una cuenta nueva suele ser 0-1).
2. **Créditos AWS** para cubrir:
   - Cómputo GPU bajo demanda/spot para entrenamiento: estimamos **~100-150 GPU-hora por evento tipo innovaton** (200 participantes), más un uso continuo bajo del equipo (desarrollo, testing) fuera de eventos — presupuestar algo como **300-500 GPU-hora/mes** de margen mientras el equipo activo desarrolla y prueba.
   - Cómputo liviano (Fargate/EC2 chico) para el simulador y la web app: costo bajo, estimable en el orden de USD 50-150/mes con uso moderado.
   - RDS + S3 + transferencia de datos: costo bajo, del orden de USD 20-50/mes.
3. Sin monto exacto todavía porque no tenemos aún facturación histórica en AWS (venimos de local + Kaggle gratuito) — pedimos que el otorgamiento sea revisable/ampliable una vez que tengamos 1-2 meses de datos reales de uso.

---

## 8. Resumen para copiar/pegar si hace falta algo más corto

> Necesitamos: (1) cupo de GPU en AWS (familia g6/g5, NVIDIA L4/A10G, ~8 concurrentes) para entrenamiento RL de policies — jobs cortos (minutos a ~1h), no requieren GPU de alta gama; (2) cómputo liviano sin GPU para servir el simulador web y la app (login/multiusuario); (3) créditos que cubran ~300-500 GPU-hora/mes estimadas, más ~USD 100-200/mes de infraestructura liviana (web, DB, storage). El entrenamiento va a estar limitado por una cola con concurrencia acotada (no todos entrenan a la vez), así que el pico de gasto es predecible y no escala 1:1 con la cantidad de usuarios.
