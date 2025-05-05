# Guía de Uso: CytoFLEX Mie Analysis Tool

## 1. Introducción

Este conjunto de scripts de Python está diseñado para analizar datos de citometría de flujo (específicamente del CytoFLEX), aplicando la **Teoría de Mie** para:

1.  **Calibrar** el citómetro usando partículas de referencia (ej. perlas de sílice) y relacionar la intensidad de la luz dispersada (SSC - Side Scatter) con las propiedades físicas.
2.  **Procesar** datos de muestras experimentales (ej. fitoplancton, bacterias) para estimar el **diámetro (tamaño)** y el **índice de refracción (RI)** de cada partícula detectada, basándose en la calibración previa.

### Estructura del Código

El código está organizado en varios módulos:

* `main.py`: El script principal que se ejecuta desde la línea de comandos. Orquesta todo el proceso.
* `config.py`: Define y gestiona la configuración por defecto y la configuración cargada/guardada.
* `calibration.py`: Contiene las funciones para los cálculos de la teoría de Mie (coeficientes, sección eficaz) y el ajuste de calibración (cálculo de K y B).
* `solver.py`: Implementa los algoritmos para la inversión de las señales SSC y obtener el tamaño y RI. Incluye la gestión de tablas de consulta (lookup tables) para acelerar el proceso.
* `reader.py`: Se encarga de leer los archivos de datos `.fcs`, extrayendo las señales SSC necesarias y metadatos relevantes. Incluye un lector binario como fallback.
* `plots.py`: Genera las gráficas de calibración y de resultados del análisis.
* `logging_config.py`: Configura el sistema de logging (mensajes informativos, advertencias, errores).

## 2. Configuración (`config.py` y `results/constants.json`)

La configuración es fundamental para el correcto funcionamiento del código.

* **`config.py`:** Define la **estructura** y los **valores por defecto** de todos los parámetros (físicos, del solver, de entrada/salida, etc.). Se usa si no se encuentra un archivo de constantes.
* **`results/constants.json`:** Este archivo (generado/actualizado por el comando `calibrate`) almacena la **configuración activa** que usarán los comandos `process` y `batch`. Guarda tanto los parámetros generales como las **constantes de calibración calculadas (K y B)**. **¡Es el archivo clave que vincula tu calibración con el análisis de tus muestras!**

### Parámetros de Configuración Clave (en `constants.json`)

Muchos de estos parámetros se pueden sobrescribir temporalmente usando opciones en la línea de comandos.

* **Parámetros Científicos/Físicos:**
    * `n_medium`: Índice de refracción del medio (ej. 1.337 para agua/PBS a 488nm).
    * `angle_range`: Rango angular [min, max] en grados para la colección de SSC (ej. `[35.0, 145.0]`).
    * `lambda_blue`, `lambda_violet`: Longitudes de onda de los láseres en metros (ej. `4.88e-07` para 488nm).
    * `n_particle_ref`: RI de las perlas usadas en `calibrate` (ej. 1.44 para sílice).
    * `allow_complex_index`: `true` o `false`. Si `true`, el solver puede intentar calcular una parte imaginaria (k) para el RI.
    * `core_shell_defaults`: Parámetros para el modelo núcleo-corteza (si se usa en el solver).
* **Parámetros del Solver:**
    * `diameter_min`/`max`/`step`, `index_min`/`max`/`step`, `k_min`/`max`/`step`: Rangos y pasos para la tabla de consulta y límites para la solución del solver.
    * `solver_strategy`: `'sequential'` (primero estima d, luego n,k) o `'joint'` (estima d y n simultáneamente). `'sequential'` suele ser más estable si `allow_complex_index` es `true`.
    * `solver_tolerance`, `solver_cost_threshold`: Criterios de convergencia del solver.
    * `lookup_table_enabled`: `true` para usar la tabla de consulta (acelera `process`/`batch` después de una generación inicial).
    * `lookup_table_path`: Dónde guardar/buscar la tabla.
    * `parallel_enabled`, `parallel_jobs`: Usar múltiples núcleos para acelerar (útil en `batch` y generación de tabla).
* **Parámetros de Entrada/Salida:**
    * `output_dir`: Carpeta donde se guardan los resultados (defecto: `results`).
    * `output_patterns`: Formato para los nombres de los archivos de salida (CSV, PNG).
    * `channel_mappings`: **¡MUY IMPORTANTE!** Define qué índices de canal corresponden a SSC Azul y SSC Violeta según una palabra clave encontrada en el nombre del archivo `.fcs`. Permite adaptar el código a diferentes protocolos experimentales. Si el nombre del archivo contiene "picoeukaryotes", usará el mapeo definido para "picoeukaryotes". Si no, usará el "default".
    * `channel_name_patterns`: Patrones de texto (ej. "SSC-A", "SSC_1-H") que el código intentará usar para *adivinar* los canales SSC si no encuentra un mapeo específico en `channel_mappings`.
    * `max_events_per_file`: Límite de eventos a procesar si no se especifica `--max-events`.
* **Parámetros de Calibración (Generados por `calibrate`):**
    * `K488`, `K405`: Factor de calibración $[canales / m^2]$.
    * `B488`, `B405`: Offset del detector [canales].
    * `min_I488`, `max_I488`, `min_I405`, `max_I405`: Rango de intensidades observado en la calibración (informativo).

## 3. Uso desde la Línea de Comandos (`main.py`)

El script se ejecuta con la estructura:
`python main.py [OPCIONES_GENERALES] COMANDO [OPCIONES_ESPECÍFICAS]`

### Comandos Principales

* **`calibrate`**: Realiza la calibración del instrumento.
    * **Uso:** `python main.py calibrate --calibration-file RUTA_AL_JSON`
    * **Obligatorio:**
        * `--calibration-file RUTA`: Especifica el archivo JSON con los datos de las perlas de referencia. Este JSON debe contener `metadata` (con `n_medium` y `n_particles`/`n_particle`/`n_particle_ref`) y `data` (una lista de diccionarios con `diameter_nm`, `Blue_SSC-H`, `Violet_SSC-H`).
    * **Opcional:**
        * `--angle-min ANG`, `--angle-max ANG`, `--n-medium N`: Sobrescriben los valores de `config.py`.
    * **Salida:** Crea o actualiza `results/constants.json` con los parámetros K, B, etc., y genera `results/calibration_plot.png`.

* **`process`**: Procesa un único archivo FCS para obtener tamaño y RI.
    * **Uso:** `python main.py process -s RUTA_AL_FCS [OPCIONES]`
    * **Obligatorio:**
        * `-s RUTA` o `--sample RUTA`: Ruta al archivo `.fcs` a analizar.
    * **Opciones Clave:**
        * `--constants RUTA`: Usa un archivo `constants.json` específico. Si no se usa, carga `results/constants.json`.
        * `--max-events N`: Analiza solo los primeros N eventos (o una selección aleatoria de N).
        * `--lookup-table`: Usa la tabla de consulta precalculada para acelerar (requiere generación previa si no existe o la config cambió).
        * `--sample-index N_ABS`: **FIJA el índice de refracción absoluto (n)** de la muestra a `N_ABS`. El script calculará $m = N_{ABS} / n_{medio}$ y resolverá *solo para el diámetro*. Muy útil si conoces el RI esperado y solo quieres el tamaño. Si no se usa, el solver busca `d` y `n` (y `k`).
        * `--d-min/max`, `--n-min/max`: Sobrescriben temporalmente los límites del solver definidos en `constants.json`.
        * `--dry-run`: Lee el archivo pero no ejecuta la inversión. Útil para probar la lectura y mapeo de canales.
    * **Salida:** Crea `results/{nombre_muestra}_results.csv` y `results/{nombre_muestra}_plot.png`.

* **`batch`**: Procesa múltiples archivos FCS en un directorio.
    * **Uso:** `python main.py batch --sample-dir RUTA_DIRECTORIO [OPCIONES]`
    * **Obligatorio:**
        * `--sample-dir RUTA`: Directorio que contiene los archivos `.fcs`.
    * **Opciones:**
        * `--pattern "*.fcs"`: Patrón para buscar archivos (ej. `"*.fcs"`, `"Experimento_A*.fcs"`).
        * `--compare` / `--no-compare`: Generar/no generar `results/comparison.png`.
        * Otras opciones (`--constants`, `--max-events`, etc.) funcionan igual que en `process`.
    * **Salida:** Archivos CSV y plot individuales para cada muestra procesada, y opcionalmente `results/comparison.png`.

### Opciones Generales (aplican a todos los comandos)

* `-v` o `--verbose`: Muestra más información (logs de nivel DEBUG).
* `-o RUTA` o `--output-dir RUTA`: Especifica la carpeta de salida (defecto: `results`).
* `--config-file RUTA`: Carga un archivo de configuración inicial (sobrescribe los defaults de `config.py` pero *antes* de cargar `constants.json`).
* `--save-config`: **¡Cuidado!** Guarda en `results/constants.json` cualquier parámetro modificado mediante la línea de comandos (ej. si usaste `--n-medium`). Útil para hacer cambios permanentes, pero úsalo con precaución.
* `--parallel`, `--n-jobs N`: Activa/configura el uso de múltiples núcleos de CPU.
* `--progress` / `--no-progress`: Muestra/oculta barras de progreso.
* `--just-validate`: Solo comprueba argumentos y configuración, no ejecuta nada.

## 4. Flujo de Trabajo Sugerido

1.  **Prepara tu archivo de calibración:** Crea un archivo `.json` (ej. `mis_perlas.json`) con los datos de tus perlas de referencia (diámetros e intensidades SSC azul y violeta medias o medianas) y los metadatos (`n_medium`, `n_particles`).
2.  **Calibra:** Ejecuta `python main.py calibrate --calibration-file mis_perlas.json`.
3.  **Revisa la Calibración:** Examina la salida del log (valores K, B, R², errores) y la gráfica `results/calibration_plot.png`. Asegúrate de que el ajuste sea bueno. Si no lo es, revisa tus datos de calibración o los parámetros en `config.py`. El archivo `results/constants.json` ahora contiene tu calibración activa.
4.  **(Opcional) Genera Tabla de Consulta:** Si vas a procesar muchas muestras o eventos y quieres acelerar, ejecuta una vez `python main.py process -s TU_FCS_MAS_GRANDE --lookup-table` (o `batch`). La primera vez generará la tabla (`results/lookup_table.pkl`), lo cual puede tardar bastante. Las siguientes veces la usará directamente si `--lookup-table` está presente y `constants.json` no ha cambiado.
5.  **Procesa Muestras:**
    * Para una muestra: `python main.py process -s archivo_muestra.fcs` (añade opciones como `--max-events` o `--sample-index` si es necesario).
    * Para muchas muestras: `python main.py batch --sample-dir ./mis_muestras --pattern "*.fcs"` (añade opciones si es necesario).
6.  **Analiza Resultados:** Revisa los archivos `.csv` generados (con columnas `diameter_nm`, `n_particle`, etc.) y las gráficas `.png`.

## 5. Consejos y Posibles Problemas

* **Mapeo de Canales Incorrecto:** Si los resultados parecen extraños, verifica que el nombre de tu archivo `.fcs` coincida con alguna clave en `channel_mappings` de `constants.json` o que los `channel_name_patterns` sean adecuados para tus nombres de canal (ej. SSC-H vs SSC-A). Usa `--dry-run` para verificar qué índices se están seleccionando sin procesar todo.
* **Resultados de Tamaño Inesperados:**
    * Verifica que las **ganancias del citómetro** fueran las **mismas** durante la adquisición de la calibración y de la muestra. Es la causa más común de discrepancias.
    * Revisa la calidad de tu calibración. ¿Era buena? ¿Cubre el rango de intensidades de tu muestra?
    * Considera las incertidumbres: el resultado es un tamaño/RI de *esfera equivalente*. La forma real, estructura interna o RI variable de tus partículas introducirá desviaciones.
* **Mala Calibración:** Si `calibrate` da malos R² o errores altos, revisa los datos de entrada (diámetros, intensidades), el `n_particle_ref` y `n_medium` usados. Mira el `calibration_plot.png` para identificar posibles puntos outliers.
* **Errores del Solver:** Si muchas partículas no convergen (columna `converged` en el CSV es `False`), prueba a ajustar los rangos del solver (`--d-min/max`, `--n-min/max`) o a usar/no usar la tabla de consulta.
* **Índice de Refracción Complejo:** Si `allow_complex_index` es `true` y la estrategia es `sequential`, los resultados de `n` y `k` dependerán de los `priors` definidos en la configuración. Si solo te interesa el tamaño, considera usar `--sample-index` si tienes una buena estimación del RI de tu muestra. Modifica `main.py` si necesitas guardar la parte imaginaria `k` en el CSV.