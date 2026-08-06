# Ejecución paso a paso: Entrenamiento con Active Learning (AL)

Este documento explica cómo ejecutar el entrenamiento con Active Learning usando el código ya integrado en `esnlir/training/train.py`. Incluye requisitos, configuración por JSON, flags por CLI, ejemplos, verificación de resultados y un smoke test.

## 1) Requisitos previos

- Python 3.10
- Paquetes del proyecto instalados (incluye PyTorch y Transformers). Si no está instalado, desde la raíz del repo:

```powershell
# Con uv (recomendado)
uv sync

# O con pip si no usa uv
python -m pip install -r requirements.txt
```

- Dataset en `data/` con los archivos `train.json`, `val.json` y al menos un `test*.json`.
- Archivo de parámetros base (p. ej. `params/train_xlmroberta.json`). Ejemplo (ya incluido):

```json
{
  "model_type": "FacebookAI/xlm-roberta-base",
  "dataset_folder": "data",
  "n_epochs": 3,
  "output_folder": "models/xlmroberta",
  "monitor": "f1_score",
  "patience": 3,
  "batch_size": 64,
  "only_premise": false,
  "warmup_steps": 0,
  "random_seed": 42,
  "learning_rate": 2e-5,
  "device": "cpu",
  "max_samples": 1000000,
  "max_len": 256
}
```

Sugerencia: Cambiar `device` a `"cuda"` si tiene GPU con CUDA.

## 2) Configuración de Active Learning (en JSON opcional)

Puede activar AL desde el JSON (sobre-escribible por CLI):

- `"active_learning"`: true | false
- `"al_strategy"`: "NegE" | "Random" | "Rem"
- `"al_L"`: iteraciones de AL (por defecto 500)
- `"al_K"`: adquisiciones por iteración (por defecto 8)
- `"al_remove_k"`: solo para REM; cantidad a eliminar cada iteración
- `"al_scoring_batch_size"`: batch para scoring (por defecto 64)
- `"warm_start"`: true | false (por defecto true)

Ejemplo mínimo para AL NegE con pequeñas iteraciones:

```json
{
  "active_learning": true,
  "al_strategy": "NegE",
  "al_L": 1,
  "al_K": 2
}
```

## 3) Ejecución desde CLI (Windows PowerShell)

Las flags de CLI tienen prioridad sobre el JSON.

- Entrenamiento normal (sin AL):

```powershell
uv run python esnlir\training\train.py --config-file params\train_xlmroberta.json
```

- Active Learning con Negative Energy (1 iteración, K=2):

```powershell
uv run python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy NegE --al_L 1 --al_K 2
```

- Active Learning con selección aleatoria (Random):

```powershell
uv run python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy Random --al_L 1 --al_K 2
```

- Active Learning con REM (eliminar 5 de baja utilidad, luego seleccionar 2):

```powershell
uv run python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy Rem --al_remove_k 5 --al_L 1 --al_K 2
```

Parámetros útiles:

- `--al_scoring_batch_size`: controla el tamaño de batch durante el scoring (por defecto 64).
- `--al_warm_start`: si false, reinicia pesos a partir del checkpoint base cada iteración; si true (por defecto), continúa fine-tuning acumulado.

### Guardando la salida en logs

Para capturar toda la salida del entrenamiento en la carpeta `logs/`:

```powershell
# Crear carpeta logs si no existe
New-Item -ItemType Directory -Force logs | Out-Null

# Ejecutar y guardar toda la salida en logs/xlmroberta.log
uv run python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy NegE --al_L 1 --al_K 2 *> "logs\xlmroberta.log"

# O para ver el output en tiempo real Y guardarlo en logs
uv run python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy NegE --al_L 1 --al_K 2 2>&1 | Tee-Object -FilePath "logs\xlmroberta.log"
```

## 4) ¿Qué hace el ciclo de Active Learning?

Por iteración (t):

1. Construye un `DataLoader` del subset no etiquetado (pool).
2. La estrategia puntúa las muestras (con `torch.no_grad()` y batching) y selecciona `K` índices globales. En REM también elimina `remove_k` de baja utilidad.
3. El `PoolManager` mueve los seleccionados a etiquetados y elimina los descartados (si aplica).
4. Se reentrena el modelo con el subset etiquetado. Por defecto, warm-start (continúa el fine-tuning).
5. Se evalúa y se guardan métricas y artefactos de la iteración.

## 5) Salidas generadas

Las salidas se guardan bajo la carpeta `output_folder` del JSON, en un subdirectorio por estrategia. Por ejemplo, con `output_folder = "models/xlmroberta"` y estrategia NegE:

```
models/xlmroberta/xlm-roberta-base_active_NegE/
  selected_indices_iter_0.csv
  removed_indices_iter_0.csv   (solo para REM)
  metrics_iter_0.json
  iter_0/
    test/total/*.csv           (métricas por split/género/dominio)
    ...
```

Notas:

- El nombre del subdirectorio incluye el nombre corto del modelo (`model_type` sin el prefijo del repositorio HF) y la estrategia.
- `selected_indices_iter_t.csv` contiene los índices globales seleccionados en la iteración t.
- `metrics_iter_t.json` incluye un resumen simple de validación.

## 6) Smoke test (rápido)

Se incluye un script de humo que ejecuta una iteración pequeña y valida archivos mínimos:

```powershell
uv run python scripts\run_al_smoke_test.py
```

Verifica que existan:

- `selected_indices_iter_0.csv`
- `metrics_iter_0.json`

en la carpeta `models/{model_short}_active_NegE/`.

## 7) Buenas prácticas y troubleshooting

- Dispositivo: use `"cuda"` en `params/*.json` si dispone de GPU. Ej.: `"device": "cuda"`.
- Memoria: si el scoring o entrenamiento se queda sin memoria, reduzca `batch_size` (entrenamiento) y/o `al_scoring_batch_size` (scoring AL), o `max_len`.
- Reproducibilidad: la semilla (`random_seed`) se fija con `seed_everything`.
- Datos: asegúrese de que `data/train.json`, `data/val.json` y `data/test*.json` existen y tienen el formato esperado.
- Carpeta de salida: la carpeta `output_folder` se crea automáticamente si no existe.

## 8) Valores por defecto recomendados

- `al_L = 500` (iteraciones)
- `al_K = 8` (adquisiciones por iteración)
- `al_scoring_batch_size = 64`
- `warm_start = true`

## 9) Estrategias disponibles

- `Random`: selección aleatoria (baseline)
- `NegE`: incertidumbre mediante energía `E(x) = -logsumexp(logits)` (más alta = más incierto)
- `Rem`: elimina `remove_k` de menor utilidad y luego selecciona `K` (aleatorio por defecto u opción de incertidumbre)

Para más detalles teóricos y de implementación, consulte `esnlir/active_learning/README.md`.
