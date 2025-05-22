import numpy as np
from scipy.optimize import root, least_squares, brentq
from scipy.interpolate import RegularGridInterpolator
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import pickle
import time
from calibration import sigma_sca_ssc
from config import get_config

# Configuración del logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cytoflex_solver')

def create_lookup_table(constants, save_to_disk=True, output_path=None, use_checkpoints=True, checkpoint_interval=50):
    """
    Crea una tabla de búsqueda para aceleración del proceso de inversión.
    Implementa checkpoints para reanudar cálculos interrumpidos.
    
    Args:
        constants: Diccionario con constantes de calibración
        save_to_disk: Si se debe guardar la tabla en disco para reutilización
        output_path: Ruta donde guardar la tabla (opcional, usa valor de config si es None)
        use_checkpoints: Si se deben utilizar checkpoints (por defecto True)
        checkpoint_interval: Número de filas a calcular antes de guardar un checkpoint
        
    Returns:
        Tupla (tabla_I488, tabla_I405, diámetros, índices)
    """
    config = get_config()
    
    # Obtener constantes necesarias
    K488, K405 = constants['K488'], constants['K405']
    n_medium = constants['n_medium']
    angle_range = constants['angle_range']
    λ488 = config.lambda_blue
    λ405 = config.lambda_violet
    
    # Obtener los rangos de la configuración
    diam_params = config.solver_params['diameter_range']
    n_params = config.solver_params['index_range']
    
    # Generar arrays de diámetros e índices
    diameters = np.arange(diam_params[0], diam_params[1], diam_params[2])
    n_particles = np.arange(n_params[0], n_params[1], n_params[2])
    
    # Determinar la ruta de la tabla de lookup
    if output_path is None:
        output_path = config.solver.get('lookup_table_path', 'results/lookup_table.pkl')
    
    # Ruta para checkpoints
    checkpoint_path = output_path.replace('.pkl', '_checkpoint.pkl')
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calcular el hash de la configuración actual para verificar compatibilidad
    # Incluimos las constantes de calibración en el hash
    current_hash = config.calculate_config_hash(include_calibration=True)
    
    # Inicializar variables de control para checkpoints
    start_row = 0
    checkpoint_exists = False
    
    # Comprobar si existe un checkpoint para reanudar
    if use_checkpoints and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Verificar que el checkpoint sea compatible usando el hash de configuración
            if checkpoint.get('config_hash') == current_hash:
                # Verificar que los arrays de diámetros e índices coincidan
                cp_diams = checkpoint.get('diameters')
                cp_indices = checkpoint.get('n_particles')
                
                if (len(cp_diams) == len(diameters) and 
                    len(cp_indices) == len(n_particles) and
                    np.allclose(cp_diams, diameters) and 
                    np.allclose(cp_indices, n_particles)):
                    
                    # Extraer datos de checkpoint válido
                    I488_table = checkpoint.get('I488_table')
                    I405_table = checkpoint.get('I405_table')
                    start_row = checkpoint.get('last_row', 0) + 1
                    
                    # Solo continuar si hay filas ya calculadas y aún faltan por calcular
                    if 0 < start_row < len(diameters):
                        logger.info(f"Reanudando cálculo desde la fila {start_row} (completado: {start_row}/{len(diameters)} = {start_row/len(diameters)*100:.1f}%)")
                        checkpoint_exists = True
                    else:
                        logger.info("Checkpoint completado o sin progreso. Iniciando cálculo desde el principio.")
                else:
                    logger.info(f"Arrays de checkpoint no coinciden con la configuración actual. Regenerando tabla...")
            else:
                # Si el hash no coincide, regenerar la tabla
                logger.info(f"Hash de configuración ha cambiado. Regenerando tabla...")
        except Exception as e:
            logger.warning(f"Error al cargar checkpoint: {e}. Iniciando cálculo desde el principio.")
    
    # Comprobar si ya existe una tabla completa en disco con el mismo hash
    if not checkpoint_exists and os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                cache = pickle.load(f)
                
            # Verificar que la tabla sea compatible usando el hash de configuración
            if cache.get('config_hash') == current_hash:
                logger.info(f"Cargando tabla de lookup completa desde {output_path} (hash: {current_hash[:8]}...)")
                return (cache['I488_table'], cache['I405_table'], 
                        cache['diameters'], cache['n_particles'])
            else:
                # Si el hash no coincide, regenerar la tabla
                logger.info(f"Hash de configuración ha cambiado ({cache.get('config_hash', 'none')[:8]}... → {current_hash[:8]}...). Regenerando tabla...")
        except Exception as e:
            logger.warning(f"Error al cargar tabla de lookup desde disco: {e}. Regenerando...")
    
    # Inicializar tablas (desde cero o desde checkpoint)
    if not checkpoint_exists:
        I488_table = np.zeros((len(diameters), len(n_particles)))
        I405_table = np.zeros((len(diameters), len(n_particles)))
    
    logger.info(f"Generando tabla de búsqueda: {len(diameters)}x{len(n_particles)} puntos{' (continuando desde checkpoint)' if checkpoint_exists else ''}")
    
    # Determinar si usar paralelismo para generación de tabla
    n_jobs = config.solver.get('parallel_jobs', -1)
    parallel_enabled = config.solver.get('parallel_enabled', False)
    
    # Registrar tiempo de inicio para medir rendimiento
    start_time = time.time()
    
    # Función para calcular una fila de la tabla (un diámetro para todos los índices)
    def calculate_row(d, n_particles_array):
        row_I488 = np.zeros(len(n_particles_array))
        row_I405 = np.zeros(len(n_particles_array))
        r = d / 2e9  # nm → m
        
        for j, n_part in enumerate(n_particles_array):
            # Calcular secciones eficaces
            σ488 = sigma_sca_ssc(r, n_part, λ488, n_medium, angle_range)
            σ405 = sigma_sca_ssc(r, n_part, λ405, n_medium, angle_range)
            
            # Convertir a intensidades
            row_I488[j] = K488 * σ488
            row_I405[j] = K405 * σ405
        
        return row_I488, row_I405
    
    # Procesar solo las filas pendientes
    remaining_diameters = diameters[start_row:]
    
    # Función para guardar checkpoint
    def save_checkpoint(current_idx, I488_table, I405_table):
        cp_row = start_row + current_idx - 1
        checkpoint_data = {
            'I488_table': I488_table,
            'I405_table': I405_table,
            'diameters': diameters,
            'n_particles': n_particles,
            'constants': {
                'K488': K488,
                'K405': K405,
                'n_medium': n_medium,
                'angle_range': angle_range
            },
            'config_hash': current_hash,
            'last_row': cp_row,
            'timestamp': time.time()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.info(f"Checkpoint guardado en {checkpoint_path} (fila {cp_row}/{len(diameters)-1}, {cp_row/(len(diameters)-1)*100:.1f}% completado)")
        
        # También guardar un informe de progreso en texto plano para monitoreo
        progress_path = checkpoint_path.replace('.pkl', '.txt')
        with open(progress_path, 'w') as f:
            f.write(f"Progreso de generación de tabla de lookup\n")
            f.write(f"======================================\n")
            f.write(f"Timestamp: {time.ctime()}\n")
            f.write(f"Hash de configuración: {current_hash}\n")
            f.write(f"Filas completadas: {cp_row+1}/{len(diameters)} ({(cp_row+1)/len(diameters)*100:.1f}%)\n")
            f.write(f"Tiempo transcurrido: {(time.time() - start_time)/60:.1f} minutos\n")
            
            # Estimar tiempo restante basado en el promedio por fila
            avg_time_per_row = (time.time() - start_time) / (cp_row + 1)
            remaining_rows = len(diameters) - (cp_row + 1)
            est_remaining_time = avg_time_per_row * remaining_rows
            
            f.write(f"Tiempo promedio por fila: {avg_time_per_row:.2f} segundos\n")
            f.write(f"Filas restantes: {remaining_rows}\n")
            f.write(f"Tiempo estimado restante: {est_remaining_time/60:.1f} minutos\n")
    
    # Generar la tabla en paralelo o secuencial
    min_items_for_parallel = config.solver.get('min_size_for_parallel', 5) # Defaulting to 5 if not in config
    if parallel_enabled and len(remaining_diameters) >= min_items_for_parallel:
        # Procesamiento paralelo con joblib
        logger.info(f"Usando modo paralelo con {n_jobs if n_jobs != -1 else 'todos los'} cores")
        
        # Dividir en lotes para poder guardar checkpoints
        batch_size = min(checkpoint_interval, len(remaining_diameters))
        total_batches = (len(remaining_diameters) - 1) // batch_size + 1
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(remaining_diameters))
            batch_diameters = remaining_diameters[batch_start:batch_end]
            
            logger.info(f"Procesando lote {batch_idx+1}/{total_batches} (filas {start_row+batch_start} a {start_row+batch_end-1})")
            
            # Calcular lote en paralelo
            results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_row)(d, n_particles) 
                for d in tqdm(batch_diameters, desc=f"Lote {batch_idx+1}/{total_batches}")
            )
            
            # Actualizar tablas con resultados del lote
            for i, (row_I488, row_I405) in enumerate(results):
                global_row = start_row + batch_start + i
                I488_table[global_row, :] = row_I488
                I405_table[global_row, :] = row_I405
            
            # Guardar checkpoint después de cada lote completo
            if use_checkpoints and batch_idx < total_batches - 1:  # no necesario para el último lote
                save_checkpoint(batch_start + batch_end, I488_table, I405_table)
    else:
        # Procesamiento secuencial con guardado de checkpoints
        logger.info("Usando modo secuencial")
        checkpoint_counter = 0
        
        for i, d in enumerate(tqdm(remaining_diameters, desc="Generando lookup table")):
            # Calcular fila actual
            I488_table[start_row + i, :], I405_table[start_row + i, :] = calculate_row(d, n_particles)
            
            # Guardar checkpoint periódicamente
            checkpoint_counter += 1
            if use_checkpoints and checkpoint_counter >= checkpoint_interval and i < len(remaining_diameters) - 1:
                save_checkpoint(i+1, I488_table, I405_table)
                checkpoint_counter = 0
    
    elapsed_time = time.time() - start_time
    logger.info(f"Tabla de búsqueda generada en {elapsed_time:.1f} segundos")
    
    # Eliminar archivo de checkpoint si existe ya que la tabla está completa
    if use_checkpoints and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint eliminado: {checkpoint_path} (tabla completa)")
        except:
            logger.warning(f"No se pudo eliminar el checkpoint: {checkpoint_path}")
    
    # Guardar en disco para reutilización
    if save_to_disk:
        cache = {
            'I488_table': I488_table,
            'I405_table': I405_table,
            'diameters': diameters,
            'n_particles': n_particles,
            'constants': {
                'K488': K488,
                'K405': K405,
                'n_medium': n_medium,
                'angle_range': angle_range
            },
            'config_hash': current_hash,  # Añadir hash de configuración para verificación
            'timestamp': time.time()
        }
        with open(output_path, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Tabla de lookup completa guardada en {output_path} (hash: {current_hash[:8]}...)")
    
    return I488_table, I405_table, diameters, n_particles

def create_interpolators(lookup_table):
    """
    Crea interpoladores para diámetro e índice basados en la tabla de lookup
    
    Args:
        lookup_table: Tupla de (tabla_I488, tabla_I405, diámetros, índices)
        
    Returns:
        Tupla con (interpolador_diámetro, interpolador_índice)
    """
    I488_table, I405_table, diameters, n_particles = lookup_table
    
    # Transformar a espacio logarítmico para mejor interpolación
    log_I488 = np.log10(np.maximum(I488_table, 1e-10))  # Evitar log(0)
    log_I405 = np.log10(np.maximum(I405_table, 1e-10))
    
    # Crear una malla regular para los valores de intensidad
    d_mesh, n_mesh = np.meshgrid(diameters, n_particles, indexing='ij')
    
    # Reshape para interpolación
    log_I488_flat = log_I488.flatten()
    log_I405_flat = log_I405.flatten()
    
    # Crear interpoladores
    points = np.column_stack((log_I488_flat, log_I405_flat))
    d_values = d_mesh.flatten()
    n_values = n_mesh.flatten()
    
    # Eliminar puntos duplicados para evitar problemas de RegularGridInterpolator
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    points_unique = points[unique_indices]
    d_values_unique = d_values[unique_indices]
    n_values_unique = n_values[unique_indices]
    
    # Construir y retornar los interpoladores
    return {
        'points': points_unique,
        'd_values': d_values_unique,
        'n_values': n_values_unique
    }

def interpolate_initial_values(i488, i405, interpolators, d_min, d_max, n_min, n_max):
    """
    Interpola valores iniciales para el solver basado en la tabla de lookup
    
    Args:
        i488, i405: Intensidades observadas
        interpolators: Diccionario con datos para interpolación
        d_min, d_max, n_min, n_max: Límites para valores válidos
        
    Returns:
        Tupla con (diámetro_inicial, índice_inicial)
    """
    config = get_config()
    
    try:
        # Convertir a espacio logarítmico
        log_i488 = np.log10(max(i488, 1e-10))
        log_i405 = np.log10(max(i405, 1e-10))
        
        # Punto de consulta
        query_point = np.array([log_i488, log_i405])
        
        # Extraer datos de interpolación
        points = interpolators['points']
        d_values = interpolators['d_values']
        n_values = interpolators['n_values']
        
        # Encontrar el vecino más cercano
        distances = np.sum((points - query_point)**2, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Obtener valores iniciales
        d_init = d_values[nearest_idx]
        n_init = n_values[nearest_idx]
        
        # Aplicar límites de seguridad
        d_init = max(d_min, min(d_max, d_init))
        n_init = max(n_min, min(n_max, n_init))
        
    except Exception as e:
        logger.debug(f"Error en interpolación: {e}")
        # Usar valores por defecto desde la configuración
        d_init = config.get('default_diameter_init', 500)
        n_init = config.get('default_index_init', 1.05)
    
    return d_init, n_init

def solve_mie_equations(vars, i488, i405, K488, K405, λ488, λ405, n_medium, angle_range):
    """
    Función centralizada para resolver ecuaciones de Mie, usada por varios métodos.
    Calcula el coste en espacio logarítmico siguiendo el protocolo de de Rond (Eq. 5).
    
    Args:
        vars: Lista [d, n_part] o [d, n_part, k_part] con diámetro e índice (real y posiblemente imaginario)
             O [d, n_core, k_core] para modelo core-shell con valores fijos de la corteza
        i488, i405: Intensidades observadas
        K488, K405: Factores de calibración
        λ488, λ405: Longitudes de onda
        n_medium: Índice del medio
        angle_range: Rango de ángulos
        
    Returns:
        Lista [res1, res2] con residuos en espacio logarítmico para root() o residuals()
    """
    from calibration import sigma_sca_ssc, sigma_sca_ssc_coreshell
    
    config = get_config()
    allow_complex_index = config.scientific.get('allow_complex_index', False)
    
    # Verificar si el modelo núcleo-corteza está habilitado
    core_shell_defaults = config.scientific.get('core_shell_defaults', None)
    use_core_shell_model = core_shell_defaults is not None and len(core_shell_defaults) > 0
    
    if use_core_shell_model:
        # Modelo núcleo-corteza
        shell_thickness_nm = core_shell_defaults.get('shell_thickness_nm', 20)
        n_shell = core_shell_defaults.get('n_shell', 1.45)
        k_shell = core_shell_defaults.get('k_shell', 0.0)
        
        # Extraer los parámetros según el formato de vars
        if len(vars) == 2:
            # [diámetro, n_core] con k_core=0
            d_total, n_core = vars
            k_core = 0.0
        elif len(vars) == 3:
            # [diámetro, n_core, k_core]
            d_total, n_core, k_core = vars
        else:
            raise ValueError(f"Formato de variables no reconocido para modelo core-shell: {vars}")
        
        # Calcular dimensiones: el diámetro total incluye el grosor de la corteza
        r_total = d_total / 2e9  # nm -> m
        t_shell = shell_thickness_nm * 1e-9  # nm -> m
        r_core = r_total - t_shell
        
        # Si el núcleo es demasiado pequeño, usar modelo homogéneo
        if r_core <= 0:
            logger.warning(f"Radio de núcleo negativo o cero ({r_core*1e9:.1f} nm), usando modelo homogéneo")
            # Usar modelo homogéneo con propiedades de la corteza
            σ488 = sigma_sca_ssc(r_total, n_shell + 1j * k_shell, λ488, n_medium, angle_range)
            σ405 = sigma_sca_ssc(r_total, n_shell + 1j * k_shell, λ405, n_medium, angle_range)
        else:
            # Usar modelo núcleo-corteza con parámetros separados
            # Importante: n_core y k_core se pasan como parámetros separados a sigma_sca_ssc_coreshell
            σ488 = sigma_sca_ssc_coreshell(
                r_core, t_shell,  # Geometría
                n_core, k_core,   # Propiedades del núcleo (real, imag separados)
                n_shell, k_shell, # Propiedades de la corteza (real, imag separados)
                λ488, n_medium, angle_range
            )
            σ405 = sigma_sca_ssc_coreshell(
                r_core, t_shell,  # Geometría
                n_core, k_core,   # Propiedades del núcleo (real, imag separados)
                n_shell, k_shell, # Propiedades de la corteza (real, imag separados)
                λ405, n_medium, angle_range
            )
    else:
        # Modelo de esfera homogénea (original)
        # Extraer valores según el modo (índice real o complejo)
        if len(vars) == 2 or not allow_complex_index:
            # Modo estándar: diámetro e índice real
            d, n_part = vars[:2]
            k_part = 0.0
        else:
            # Modo con índice complejo
            d, n_part, k_part = vars[:3]
        
        # Convertir diámetro en nm a radio en metros
        r = d / 2e9
        
        # Construir índice complejo si es necesario
        n_complex = n_part + 1j * k_part if k_part != 0.0 else n_part
        
        # Calcular secciones eficaces
        σ488 = sigma_sca_ssc(r, n_complex, λ488, n_medium, angle_range)
        σ405 = sigma_sca_ssc(r, n_complex, λ405, n_medium, angle_range)
    
    # Convertir a intensidades teóricas
    I_theo_488 = K488 * σ488
    I_theo_405 = K405 * σ405
    
    # Calcular residuos en espacio logarítmico (de Rond, Eq. 5)
    # Proteger contra valores no positivos
    log_I_theo_488 = np.log10(max(I_theo_488, 1e-10))
    log_I_theo_405 = np.log10(max(I_theo_405, 1e-10))
    log_i488 = np.log10(max(i488, 1e-10))
    log_i405 = np.log10(max(i405, 1e-10))
    
    # Retornar residuos en espacio logarítmico
    return [log_I_theo_488 - log_i488, log_I_theo_405 - log_i405]

def invert_scatter(I488, I405, constants, use_parallel=None, lookup_table=None, n_jobs=None):
    """
    Invierte las señales de dispersión para obtener diámetro e índice de refracción
    
    Args:
        I488: Señales de dispersión lateral azul (488 nm)
        I405: Señales de dispersión lateral violeta (405 nm)
        constants: Diccionario con constantes de calibración
        use_parallel: Booleano para activar procesamiento paralelo (None=usar config)
        lookup_table: Tupla de (tabla_I488, tabla_I405, diámetros, índices) para aceleración
        n_jobs: Número de procesos paralelos (None = usar config)
        
    Returns:
        Tupla con (diámetros, índices de refracción, indicador de convergencia)
    """
    config = get_config()
    
    # Obtener constantes de calibración
    K488, K405 = constants['K488'], constants['K405']
    n_medium = constants['n_medium']
    angle_range = constants['angle_range']
    λ488 = config.lambda_blue
    λ405 = config.lambda_violet
    
    # Obtener límites de solver desde configuración
    solver_params = config.solver_params
    d_min, d_max, _ = solver_params['diameter_range']
    n_min, n_max, _ = solver_params['index_range']
    
    # Parámetros de convergencia
    tol = config.solver.get('solver_tolerance', 1e-6)
    cost_threshold = config.solver.get('solver_cost_threshold', 1e4)
    
    # Determinar si usar paralelismo
    if use_parallel is None:
        use_parallel = config.solver.get('parallel_enabled', True)
    
    # Verificar si se especificó un índice de refracción fijo
    fixed_index_mode = 'fixed_m' in constants and constants['fixed_m'] is not None
    if fixed_index_mode:
        fixed_m = constants['fixed_m']
        fixed_n = constants['fixed_index']
        logger.info(f"Usando modo de índice fijo: n={fixed_n}, m={fixed_m:.4f}")
    
    # Verificar que las intensidades son arrays
    I488 = np.asarray(I488)
    I405 = np.asarray(I405)
    
    # Crear interpoladores si tenemos tabla de lookup
    lookup_interpolators = None
    if lookup_table is not None:
        lookup_interpolators = create_interpolators(lookup_table)
    
    # Función para procesar una partícula
    def process_particle(i488, i405, idx=None):
        # Si las señales son muy bajas o cero, se saltea
        if i488 < 1 or i405 < 1:
            logger.debug(f"Partícula {idx}: señales muy bajas ({i488:.1f}, {i405:.1f})")
            return np.nan, np.nan, False, 0
        
        # Filtrar valores extremos
        max_I488 = constants.get('max_I488', 1e8)
        max_I405 = constants.get('max_I405', 1e8)
        if i488 > max_I488 or i405 > max_I405:
            logger.debug(f"Partícula {idx}: señales demasiado altas (i488={i488:.1f}, i405={i405:.1f})")
            return np.nan, np.nan, False, 0
            
        # Registrar señales bajas para depuración sin filtrarlas
        min_signal_488 = constants.get('min_I488', 0)
        min_signal_405 = constants.get('min_I405', 0)
        if i488 < min_signal_488 or i405 < min_signal_405:
            logger.debug(f"Señal baja permitida: {idx}: ({i488:.1f}, {i405:.1f}) < límites ({min_signal_488:.1f}, {min_signal_405:.1f})")
        
        # Obtener la estrategia de inversión desde la configuración
        allow_complex_index = config.scientific.get('allow_complex_index', False)
        strategy = solver_params.get('strategy', 'sequential' if allow_complex_index else 'simultaneous')
        
        # Modo de índice fijo: invertir solo para diámetro
        if fixed_index_mode:
            try:
                # Usar el promedio ponderado de las inversiones individuales
                # Con más peso en 488nm que generalmente es más estable
                d_488 = invert_d(i488, K488, λ488, fixed_n, n_medium, angle_range, d_min=d_min, d_max=d_max)
                d_405 = invert_d(i405, K405, λ405, fixed_n, n_medium, angle_range, d_min=d_min, d_max=d_max)
                
                # Si alguno falló, usar el otro
                if np.isnan(d_488) and not np.isnan(d_405):
                    d_estimated = d_405
                elif np.isnan(d_405) and not np.isnan(d_488):
                    d_estimated = d_488
                elif np.isnan(d_488) and np.isnan(d_405):
                    return np.nan, fixed_n, False, 0
                else:
                    # Promedio ponderado (más peso a 488nm)
                    d_estimated = (2*d_488 + d_405) / 3
                
                # Éxito en la inversión con índice fijo
                return d_estimated, fixed_n, True, 50
            except Exception as e:
                logger.debug(f"Error en inversión con índice fijo para partícula {idx}: {e}")
                return np.nan, fixed_n, False, 0
        
        # Estrategia secuencial (405 → 488 nm) para índices complejos
        if strategy == 'sequential' and allow_complex_index:
            logger.debug(f"Partícula {idx}: Attempting sequential inversion strategy.")
            try:
                # Paso 1: Estimar diámetro usando la señal de 405 nm (asumiendo k≈0 para 405nm)
                # Obtener valor inicial para n del núcleo desde la configuración
                core_shell_defaults = config.scientific.get('core_shell_defaults', {})
                n_prior = core_shell_defaults.get('n_core_prior', 1.39)
                
                # Invertir diámetro con 405 nm (menos afectado por absorción)
                d_est = invert_d(i405, K405, λ405, n_prior, n_medium, angle_range, d_min=d_min, d_max=d_max)
                logger.debug(f"Partícula {idx} (Sequential Strategy): d_est (from 405nm) = {d_est:.2f} nm, n_prior_for_d_est = {n_prior:.3f}")
                
                if np.isnan(d_est):
                    logger.debug(f"Partícula {idx}: falló inversión de diámetro con 405 nm")
                    return np.nan, np.nan, False, 0
                
                # Paso 2: Ajustar n y k usando la señal de 488 nm con el diámetro fijo
                # Este paso usa priors gaussianos para estabilizar la inversión
                n_est, k_est = fit_nk_fixed_d(d_est, i488, K488, λ488, n_medium, angle_range,
                                            n_prior=n_prior, 
                                            k_prior=core_shell_defaults.get('k_core_prior', 0.005))
                logger.debug(f"Partícula {idx} (Sequential Strategy): n_est = {n_est:.4f}, k_est = {k_est:.4f} (from 488nm fit with d_est={d_est:.2f} nm)")
                
                # Combinar resultados y verificar si son físicamente razonables
                if d_min <= d_est <= d_max and n_min <= n_est <= n_max:
                    # Construir índice complejo para retornar
                    n_complex = complex(n_est, k_est)
                    return d_est, n_complex, True, 100  # 100 es un valor ficticio para iteraciones
                else:
                    logger.debug(f"Partícula {idx}: inversión secuencial dio valores fuera de rango: d={d_est}, n={n_est}, k={k_est}")
                    return np.nan, np.nan, False, 0
                
            except Exception as e:
                logger.debug(f"Error en estrategia secuencial para partícula {idx}: {e}")
                # Fallback al método estándar si falló la estrategia secuencial
                logger.debug(f"Intentando método estándar para partícula {idx}")
                # El código continuará con el modo estándar
        
        # Modo estándar/simultáneo: inversión conjunta de diámetro e índice
        # Determinar valores iniciales
        if lookup_interpolators is not None:
            # Usar interpolación para valores iniciales
            d_init, n_init = interpolate_initial_values(
                i488, i405, lookup_interpolators, d_min, d_max, n_min, n_max
            )
        else:
            # Valores iniciales de configuración
            solver_init = solver_params.get('init_values', {})
            d_init = solver_init.get('diameter', 500)
            n_init = solver_init.get('index', 1.05)
        
        # Definir las funciones para resolver el sistema usando nuestra función centralizada
        def equations(vars):
            return solve_mie_equations(vars, i488, i405, K488, K405, λ488, λ405, n_medium, angle_range)
        
        def residuals(vars):
            return solve_mie_equations(vars, i488, i405, K488, K405, λ488, λ405, n_medium, angle_range)
        
        # Resolver sistema de ecuaciones con el primer método
        try:
            sol = root(equations, x0=[d_init, n_init], tol=tol, method='hybr')
            d_estimated, n_estimated = sol.x
            n_iter = sol.nfev  # Número de evaluaciones de función
            
            # Comprobar que la solución es físicamente razonable
            if d_min <= d_estimated <= d_max and n_min <= n_estimated <= n_max and sol.success:
                return d_estimated, n_estimated, True, n_iter
            
            # Si no converge o da valores no razonables, intentar otro método
            logger.debug(f"Partícula {idx}: primer método falló, intentando least_squares")
            
            # Límites para least_squares
            lb = [d_min, n_min]      # Lower bounds
            ub = [d_max, n_max]      # Upper bounds
            
            sol2 = least_squares(residuals, x0=[d_init, n_init], 
                               bounds=(lb, ub), method='trf')
            
            d_estimated, n_estimated = sol2.x
            n_iter += sol2.nfev
            
            # Verificar convergencia del segundo método
            # En least_squares, cost es 0.5*sum(residuos^2), así que multiplicamos por 2
            # para comparar con el umbral que se basa en la suma total de residuos
            total_cost = 2.0 * sol2.cost
            if sol2.success and total_cost < cost_threshold:
                return d_estimated, n_estimated, True, n_iter
            
            # Si todavía falla, devolver NaN
            logger.debug(f"Partícula {idx}: ambos métodos fallaron. Cost: {total_cost:.2e} > umbral {cost_threshold:.2e}")
            return np.nan, np.nan, False, n_iter
            
        except Exception as e:
            logger.debug(f"Error invirtiendo partícula {idx}: {e}")
            return np.nan, np.nan, False, 0
    
    # Procesar todas las partículas (en paralelo si se solicita)
    total = len(I488)
    logger.info(f"Invirtiendo {total} partículas{' en paralelo' if use_parallel else ''}")
    
    # Determinar si usar paralelismo basado en número de partículas y configuración
    min_particles_for_parallel = config.solver.get('min_size_for_parallel', 10)
    use_parallel_actual = use_parallel and total >= min_particles_for_parallel
    
    if use_parallel_actual:
        # Número de cores a usar
        if n_jobs is None:
            n_jobs = config.solver.get('parallel_jobs', -1)  # Usar valor de configuración o todos los cores
        
        # Mostrar información de paralelismo
        if n_jobs == -1:
            import multiprocessing
            cores = multiprocessing.cpu_count()
            logger.info(f"Usando todos los cores disponibles ({cores})")
        else:
            logger.info(f"Usando {n_jobs} cores para procesamiento paralelo")
        
        # Procesamiento paralelo con joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_particle)(i488, i405, i) 
            for i, (i488, i405) in enumerate(tqdm(zip(I488, I405), total=total, desc="Invirtiendo señales"))
        )
    else:
        # Procesamiento secuencial con barra de progreso
        results = []
        for i, (i488, i405) in enumerate(tqdm(zip(I488, I405), total=total, desc="Invirtiendo señales")):
            results.append(process_particle(i488, i405, i))
    
    # Extraer resultados
    diameters, n_particles, converged, n_iters = zip(*results)
    
    # Convertir a arrays NumPy
    diameters = np.array(diameters)
    n_particles = np.array(n_particles)
    converged = np.array(converged)
    
    # Calcular estadísticas de convergencia
    success_rate = np.mean(converged) * 100
    avg_iters = np.mean([n for n in n_iters if n > 0]) if any(n > 0 for n in n_iters) else 0
    
    logger.info(f"Inversión completada: {success_rate:.1f}% de éxito, {avg_iters:.1f} iteraciones promedio")
    
    return np.array(diameters), np.array(n_particles), np.array(converged)

def invert_d(I_obs, K, λ, n_part, n_medium, angle_range, d_min=None, d_max=None):
    """
    Invierte la señal para obtener el diámetro, manteniendo fijo el índice de refracción.
    
    Args:
        I_obs: Intensidad observada
        K: Factor de calibración
        λ: Longitud de onda en metros
        n_part: Índice de refracción de la partícula (fijo)
        n_medium: Índice de refracción del medio
        angle_range: Rango de ángulos para SSC [min, max] en grados
        d_min: Diámetro mínimo para la búsqueda (nm, opcional, se usa config si es None)
        d_max: Diámetro máximo para la búsqueda (nm, opcional, se usa config si es None)
        
    Returns:
        Diámetro estimado en nm, o np.nan si no se encuentra una solución.
    """
    # Importar con ámbito local para evitar dependencias circulares
    from calibration import sigma_sca_ssc, sigma_sca_ssc_coreshell
    from scipy.optimize import brentq
    import numpy as np
    from logging_config import get_logger
    logger = get_logger('cytoflex_solver')
    
    # Obtener configuración para límites y reintentos
    config = get_config()
    
    # Usar límites de configuración si no se especifican
    if d_min is None or d_max is None:
        solver_params = config.solver_params
        d_range = solver_params.get('diameter_range', [100, 5000, 10])
        d_min = d_min or d_range[0]
        d_max = d_max or d_range[1]
    
    # Parámetros adicionales de la configuración
    max_iterations = config.solver.get('d_solve_max_iterations', 100)
    absolute_d_min = config.solver.get('absolute_d_min', 50)
    absolute_d_max = config.solver.get('absolute_d_max', 20000)
    
    # Verificar si el modelo núcleo-corteza está habilitado
    core_shell_defaults = config.scientific.get('core_shell_defaults', None)
    use_core_shell_model = core_shell_defaults is not None and len(core_shell_defaults) > 0
    
    # Función que queremos hacer cero
    def f(d):
        if use_core_shell_model:
            # Modelo núcleo-corteza
            shell_thickness_nm = core_shell_defaults.get('shell_thickness_nm', 20)
            n_shell = core_shell_defaults.get('n_shell', 1.45)
            k_shell = core_shell_defaults.get('k_shell', 0.0)
            
            # Calcular dimensiones
            r_total = d / 2e9  # nm -> m
            t_shell = shell_thickness_nm * 1e-9  # nm -> m
            r_core = r_total - t_shell
            
            # Si el núcleo es demasiado pequeño, usar modelo homogéneo
            if r_core <= 0:
                logger.warning(f"Radio de núcleo negativo o cero ({r_core*1e9:.1f} nm), usando modelo homogéneo")
                n_complex = n_shell + 1j * k_shell
                σ = sigma_sca_ssc(r_total, n_complex, λ, n_medium, angle_range)
            else:
                # Extraer parte real e imaginaria para núcleo
                if isinstance(n_part, complex):
                    n_core = n_part.real
                    k_core = n_part.imag
                else:
                    # Usar valor fijo para parte imaginaria del núcleo
                    n_core = n_part
                    k_core = core_shell_defaults.get('k_core_prior', 0.005)
                
                # Calcular sección eficaz con modelo núcleo-corteza
                σ = sigma_sca_ssc_coreshell(
                    r_core, t_shell,      # Geometría
                    n_core, k_core,       # Propiedades del núcleo (floats separados)
                    n_shell, k_shell,     # Propiedades de la corteza (floats separados)
                    λ, n_medium, angle_range
                )
        else:
            # Modelo homogéneo original
            r = d / 2e9  # Convertir diámetro en nm a radio en metros
            σ = sigma_sca_ssc(r, n_part, λ, n_medium, angle_range)
        
        return K * σ - I_obs
    
    try:
        # Usar brentq para encontrar la raíz en el intervalo [d_min, d_max]
        d_estimated = brentq(f, d_min, d_max, maxiter=max_iterations)
        logger.debug(f"invert_d successful for n_part={n_part} in initial range [{d_min}, {d_max}]. d_estimated={d_estimated:.2f} nm")
        return d_estimated
    except ValueError as e:
        # Si la función no cambia de signo en el intervalo, no hay solución
        logger.debug(f"No se encontró solución para d con n_part={n_part} in initial range [{d_min}, {d_max}]: {e}")
        # Estrategia de reintentos con rangos más amplios
        retry_ranges = [
            (absolute_d_min, d_max),         # Probar con mínimo absoluto
            (d_min, absolute_d_max),         # Probar con máximo absoluto
            (absolute_d_min, absolute_d_max) # Probar con rango absoluto
        ]
        
        for i, d_range_retry in enumerate(retry_ranges):
            logger.debug(f"invert_d retrying for n_part={n_part} with range [{d_range_retry[0]}, {d_range_retry[1]}] (attempt {i+1}/{len(retry_ranges)})")
            try:
                d_estimated = brentq(f, d_range_retry[0], d_range_retry[1], maxiter=max_iterations + 50)
                logger.debug(f"invert_d successful on retry {i+1} with range [{d_range_retry[0]}, {d_range_retry[1]}]. d_estimated={d_estimated:.2f} nm")
                return d_estimated
            except ValueError:
                logger.debug(f"invert_d retry {i+1} failed for n_part={n_part} with range [{d_range_retry[0]}, {d_range_retry[1]}]")
                continue
        
        # Si todos los reintentos fallan, devolver NaN
        logger.debug(f"invert_d all retries failed for n_part={n_part}. Returning NaN.")
        return np.nan

def fit_nk_fixed_d(d, i488, K488, λ488, n_medium, angle_range, n_prior=None, k_prior=None, sigma_n=None, sigma_k=None):
    """
    Ajusta la parte real e imaginaria del índice de refracción para un diámetro fijo,
    usando priors gaussianos para estabilizar la inversión.
    
    Args:
        d: Diámetro fijo [nm]
        i488: Intensidad observada en 488 nm
        K488: Factor de calibración para 488 nm
        λ488: Longitud de onda 488 nm [m]
        n_medium: Índice de refracción del medio
        angle_range: Rango de ángulos [min, max] en grados
        n_prior: Valor previo para n (opcional, se usa config si es None)
        k_prior: Valor previo para k (opcional, se usa config si es None)
        sigma_n: Desviación estándar del prior de n (opcional, se usa config si es None)
        sigma_k: Desviación estándar del prior de k (opcional, se usa config si es None)
        
    Returns:
        Tupla (n_estimado, k_estimado)
    """
    # Importar con ámbito local para evitar dependencias circulares
    from calibration import sigma_sca_ssc, sigma_sca_ssc_coreshell
    from scipy.optimize import least_squares
    import numpy as np
    from logging_config import get_logger
    logger = get_logger('cytoflex_solver')
    
    config = get_config()
    
    # Obtener valores por defecto para los priors de la configuración
    core_shell_defaults = config.scientific.get('core_shell_defaults', {})
    n_prior = n_prior if n_prior is not None else core_shell_defaults.get('n_core_prior', 1.39)
    k_prior = k_prior if k_prior is not None else core_shell_defaults.get('k_core_prior', 0.005)
    
    # Obtener desviaciones estándar de la configuración si no se especifican
    if sigma_n is None:
        sigma_n = config.solver.get('sigma_n_prior', 0.05)
    if sigma_k is None:
        sigma_k = config.solver.get('sigma_k_prior', 0.002)
    
    # Obtener rangos para n y k de la configuración
    solver_params = config.solver_params
    n_range = solver_params.get('index_range', (1.01, 1.8, 0.01))
    k_range = solver_params.get('k_range', (0.0, 0.02, 0.002))
    n_min, n_max = n_range[0], n_range[1]
    k_min, k_max = k_range[0], k_range[1]
    
    # Verificar si el modelo núcleo-corteza está habilitado
    use_core_shell_model = core_shell_defaults is not None and len(core_shell_defaults) > 0
    
    # Función para calcular residuos con priors gaussianos
    def residuals(vars):
        n, k = vars
        
        if use_core_shell_model:
            # Modelo núcleo-corteza
            shell_thickness_nm = core_shell_defaults.get('shell_thickness_nm', 20)
            n_shell = core_shell_defaults.get('n_shell', 1.45)
            k_shell = core_shell_defaults.get('k_shell', 0.0)
            
            # Calcular dimensiones
            r_total = d / 2e9  # nm -> m
            t_shell = shell_thickness_nm * 1e-9  # nm -> m
            r_core = r_total - t_shell
            
            # Si el núcleo es demasiado pequeño, usar modelo homogéneo
            if r_core <= 0:
                logger.warning(f"Radio de núcleo negativo o cero ({r_core*1e9:.1f} nm), usando modelo homogéneo")
                n_complex = n_shell + 1j * k_shell
                σ = sigma_sca_ssc(r_total, n_complex, λ488, n_medium, angle_range)
            else:
                # Calcular sección eficaz con modelo núcleo-corteza usando parámetros separados
                σ = sigma_sca_ssc_coreshell(
                    r_core, t_shell,      # Geometría
                    n, k,                 # Propiedades del núcleo (real, imag) separados
                    n_shell, k_shell,     # Propiedades de la corteza (real, imag) separados
                    λ488, n_medium, angle_range
                )
        else:
            # Modelo homogéneo original
            r = d / 2e9  # Radio en metros
            # Calcular sección eficaz con índice complejo
            n_complex = n + 1j * k
            σ = sigma_sca_ssc(r, n_complex, λ488, n_medium, angle_range)
        
        # Intensidad teórica
        I_theo = K488 * σ
        
        # Residuo en espacio logarítmico
        log_I_theo = np.log10(max(I_theo, 1e-10))
        log_i_obs = np.log10(max(i488, 1e-10))
        
        # Penalización por desviación de los priors (distribución gaussiana)
        n_penalty = (n - n_prior) / sigma_n if n_prior is not None else 0.0
        k_penalty = (k - k_prior) / sigma_k if k_prior is not None else 0.0
        
        # Combinar residuo y penalizaciones
        return [log_I_theo - log_i_obs, n_penalty, k_penalty]
    
    # Valores iniciales
    n_init = n_prior
    k_init = k_prior
    
    # Límites para least_squares
    lb = [n_min, k_min]
    ub = [n_max, k_max]
    
    # Resolver usando least_squares
    try:
        sol = least_squares(residuals, x0=[n_init, k_init], bounds=(lb, ub), method='trf')
        n_est, k_est = sol.x
        
        # Validar que los valores están dentro de los límites físicos
        n_est = max(n_min, min(n_max, n_est))
        k_est = max(k_min, min(k_max, k_est))
        
        return n_est, k_est
    except Exception as e:
        logger.debug(f"Error en fit_nk_fixed_d: {e}")
        # En caso de error, devolver los valores previos
        return n_prior, k_prior