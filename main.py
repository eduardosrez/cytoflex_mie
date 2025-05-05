import json
import csv
import os
import argparse
import sys
import numpy as np
import logging
from pathlib import Path

# Importar configuración de logging al inicio, antes de otros imports
from logging_config import configure_logging, get_logger, quiet_noisy_loggers

# Resto de importaciones después de configurar logging
from calibration import load_calibration, fit_K, sigma_sca_ssc, evaluate_calibration_quality
from reader import read_ssc_signals
from solver import invert_scatter, create_lookup_table
from plots import plot_calibration, plot_scatter_results, plot_comparison, calculate_theory_curves, get_output_path
from config import get_config, Config

# Crear logger después de importar la configuración
logger = get_logger('cytoflex_main')

def setup_parser():
    """Configura el parser de argumentos de línea de comandos"""
    config = get_config()
    
    parser = argparse.ArgumentParser(
        description="CytoFLEX Mie - Estima diámetro e índice de refracción desde señales SSC"
    )
    
    # Argumentos generales
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Mostrar mensajes de debug")
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                       help="Directorio para guardar resultados")
    parser.add_argument('--config-file', type=str, default=None,
                       help="Archivo de configuración JSON (opcional)")
    parser.add_argument('--save-config', action='store_true',
                       help="Guardar la configuración actualizada en disco")
    parser.add_argument('--parallel', action='store_true',
                       help="Usar procesamiento paralelo")
    parser.add_argument('--n-jobs', type=int, default=None,
                       help="Número de procesos paralelos (por defecto: todos los cores)")
    parser.add_argument('--just-validate', action='store_true',
                        help="Solo validar configuración y parámetros sin ejecutar cálculos")
    parser.add_argument('--progress', action='store_true', default=True,
                       help="Mostrar barras de progreso para cálculos intensivos (activado por defecto)")
    parser.add_argument('--no-progress', action='store_false', dest='progress',
                       help="No mostrar barras de progreso")
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Calibración
    calib_parser = subparsers.add_parser('calibrate', help='Calibrar usando partículas de referencia')
    calib_parser.add_argument('--calibration-file', '-c', type=str, required=True,
                           help="Archivo JSON con datos de calibración")
    calib_parser.add_argument('--angle-min', type=float, default=config.angle_range[0],
                           help=f"Ángulo mínimo para SSC (grados, defecto: {config.angle_range[0]})")
    calib_parser.add_argument('--angle-max', type=float, default=config.angle_range[1],
                           help=f"Ángulo máximo para SSC (grados, defecto: {config.angle_range[1]})")
    calib_parser.add_argument('--n-medium', type=float, default=config.n_medium,
                           help=f"Índice de refracción del medio (defecto: {config.n_medium})")
    
    # Procesar muestra
    sample_parser = subparsers.add_parser('process', help='Procesar muestra FCS')
    sample_parser.add_argument('--sample', '-s', type=str, required=True,
                             help="Archivo FCS a procesar")
    sample_parser.add_argument('--constants', '-c', type=str, default=None,
                             help="Archivo JSON con constantes de calibración (por defecto: autodetección)")
    sample_parser.add_argument('--lookup-table', action='store_true',
                             help="Usar tabla de lookup para aceleración")
    sample_parser.add_argument('--max-events', '-m', type=int, default=None,
                             help="Número máximo de eventos a procesar (por defecto: todos)")
    sample_parser.add_argument('--d-min', type=float, default=None,
                             help=f"Diámetro mínimo [nm] (por defecto: {config.solver_params['diameter_range'][0]})")
    sample_parser.add_argument('--d-max', type=float, default=None,
                             help=f"Diámetro máximo [nm] (por defecto: {config.solver_params['diameter_range'][1]})")
    sample_parser.add_argument('--n-min', type=float, default=None,
                             help=f"Índice mínimo (por defecto: {config.solver_params['index_range'][0]})")
    sample_parser.add_argument('--n-max', type=float, default=None,
                             help=f"Índice máximo (por defecto: {config.solver_params['index_range'][1]})")
    sample_parser.add_argument('--sample-index', type=float, default=None,
                             help="Índice de refracción absoluto (n) de la muestra; se convierte a m=n/n_medium")
    sample_parser.add_argument('--dry-run', action='store_true',
                             help="Solo validar lectura y parámetros, sin ejecutar inversión")
    
    # Procesar múltiples muestras
    batch_parser = subparsers.add_parser('batch', help='Procesar múltiples muestras')
    batch_parser.add_argument('--sample-dir', '-d', type=str, required=True,
                            help="Directorio con archivos FCS")
    batch_parser.add_argument('--pattern', '-p', type=str, default='*.fcs',
                            help="Patrón para buscar archivos (glob)")
    batch_parser.add_argument('--constants', '-c', type=str, default=None,
                            help="Archivo JSON con constantes de calibración (por defecto: autodetección)")
    batch_parser.add_argument('--lookup-table', action='store_true',
                            help="Usar tabla de lookup para aceleración")
    batch_parser.add_argument('--max-events', '-m', type=int, default=None,
                            help="Número máximo de eventos a procesar por muestra (por defecto: todos)")
    batch_parser.add_argument('--d-min', type=float, default=None,
                            help=f"Diámetro mínimo [nm] (por defecto: {config.solver_params['diameter_range'][0]})")
    batch_parser.add_argument('--d-max', type=float, default=None,
                            help=f"Diámetro máximo [nm] (por defecto: {config.solver_params['diameter_range'][1]})")
    batch_parser.add_argument('--n-min', type=float, default=None,
                            help=f"Índice mínimo (por defecto: {config.solver_params['index_range'][0]})")
    batch_parser.add_argument('--n-max', type=float, default=None,
                            help=f"Índice máximo (por defecto: {config.solver_params['index_range'][1]})")
    batch_parser.add_argument('--sample-index', type=float, default=None,
                            help="Índice de refracción absoluto (n) de la muestra; se convierte a m=n/n_medium")
    batch_parser.add_argument('--compare', action='store_true', default=True,
                            help="Generar gráfico comparativo (activado por defecto)")
    batch_parser.add_argument('--no-compare', action='store_false', dest='compare',
                            help="No generar gráfico comparativo")
    batch_parser.add_argument('--dry-run', action='store_true',
                            help="Solo validar lectura y parámetros, sin ejecutar inversión")
    
    return parser

def ensure_dir(path):
    """Crea directorio si no existe"""
    os.makedirs(path, exist_ok=True)
    return path

def find_constants_file():
    """
    Busca un archivo constants.json en ubicaciones posibles
    
    Returns:
        Path al archivo si existe, None en caso contrario
    """
    possible_paths = [
        'constants.json',
        'results/constants.json',
        os.path.join(os.path.dirname(__file__), 'constants.json'),
        os.path.join(os.path.dirname(__file__), 'results', 'constants.json')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def update_config_from_args(args, config):
    """
    Actualiza la configuración con los argumentos de línea de comandos
    SIN guardar automáticamente
    
    Args:
        args: Argumentos parseados
        config: Instancia de Config a actualizar
    
    Returns:
        Tupla (config actualizada, diccionario con actualizaciones)
    """
    # Cargar archivo de configuración personalizado si se proporciona
    if args.config_file and os.path.exists(args.config_file):
        config.load_config(args.config_file)
    
    # Actualizar configuración con argumentos de línea de comandos
    updates = {}
    
    # Actualizar configuración de paralelismo
    if hasattr(args, 'parallel') and args.parallel:
        updates['parallel_enabled'] = True
    
    # Actualizar opciones de paralelismo
    if args.n_jobs is not None:
        updates['parallel_jobs'] = args.n_jobs
    
    # Actualizar límites de solver si se especifican
    if hasattr(args, 'd_min') and args.d_min is not None:
        updates['diameter_min'] = args.d_min
    if hasattr(args, 'd_max') and args.d_max is not None:
        updates['diameter_max'] = args.d_max
    if hasattr(args, 'n_min') and args.n_min is not None:
        updates['index_min'] = args.n_min
    if hasattr(args, 'n_max') and args.n_max is not None:
        updates['index_max'] = args.n_max
    
    # Actualizar parámetros de calibración
    if args.command == 'calibrate':
        if args.angle_min is not None or args.angle_max is not None:
            angle_min = args.angle_min if args.angle_min is not None else config.angle_range[0]
            angle_max = args.angle_max if args.angle_max is not None else config.angle_range[1]
            updates['angle_range'] = [angle_min, angle_max]
        if args.n_medium is not None:
            updates['n_medium'] = args.n_medium
    
    # Actualizar sin guardar automáticamente
    if updates:
        config.update(updates)
    
    return config, updates

def save_config_if_requested(config, updates, args, path=None):
    """
    Guarda la configuración en disco si se solicita con --save-config
    
    Args:
        config: Instancia de Config
        updates: Diccionario con actualizaciones realizadas
        args: Argumentos parseados
        path: Ruta donde guardar, si es None usa output_dir/constants.json
    """
    if not args.save_config or not updates:
        return
    
    if path is None:
        # Asegurarse que el directorio exista
        ensure_dir(args.output_dir)
        path = os.path.join(args.output_dir, 'constants.json')
    
    # Guardar solo si se solicitó explícitamente
    config.save(path)
    logger.info(f"Configuración guardada en: {path}")

def get_sample_name(sample_path):
    """
    Extrae el nombre de la muestra sin extensión
    
    Args:
        sample_path: Ruta al archivo de muestra
        
    Returns:
        Nombre de la muestra sin extensión
    """
    return os.path.splitext(os.path.basename(sample_path))[0]

def process_sample(sample_path, config, output_dir=None, use_parallel=None, lookup_table=None, 
                max_events=None, sample_index=None, n_jobs=None, dry_run=False,
                save_config=False):
    """
    Procesa una muestra FCS y guarda los resultados
    
    Args:
        sample_path: Ruta al archivo FCS
        config: Instancia de Config con parámetros del sistema
        output_dir: Directorio donde guardar resultados (si None, usa 'results')
        use_parallel: Si se debe usar procesamiento paralelo (None=usar config)
        lookup_table: Tabla de lookup precomputada para aceleración
        max_events: Número máximo de eventos a procesar (None para procesar todos)
        sample_index: Índice de refracción absoluto (n) de la muestra
        n_jobs: Número de procesos paralelos (None para usar el valor de config)
        dry_run: Si es True, solo valida lectura y parámetros, sin ejecutar inversión
        save_config: Si es True, guarda los parámetros de inversión en disco
        
    Returns:
        Tupla con (diámetros, índices_refracción, indicador_convergencia)
    """
    # Verificar existencia del archivo
    if not os.path.exists(sample_path):
        logger.error(f"Archivo no encontrado: {sample_path}")
        return None, None, None
    
    # Usar directorio de salida por defecto si no se especifica
    if output_dir is None:
        output_dir = 'results'
    
    # Asegurar que el directorio exista
    ensure_dir(output_dir)
    
    # Obtener nombre de la muestra
    sample_name = get_sample_name(sample_path)
    
    # Preparar constantes de calibración como diccionario
    constants = {
        'n_medium': config.n_medium,
        'angle_range': config.angle_range
    }
    
    # Verificar que las constantes de calibración existan
    if config.get('K488') is None or config.get('K405') is None:
        logger.error("Constantes de calibración no encontradas (K488 o K405). "
                   "Ejecute primero el comando 'calibrate'.")
        return None, None, None
    
    # Añadir constantes de calibración
    constants['K488'] = config.get('K488')
    constants['K405'] = config.get('K405')
    
    # Calcular índice de refracción relativo si se proporciona un valor absoluto
    if sample_index is not None:
        constants['fixed_index'] = float(sample_index)
        constants['fixed_m'] = float(sample_index) / constants['n_medium']
        logger.info(f"Usando índice de refracción fijo: n={sample_index}, m={constants['fixed_m']:.4f}")
    
    # Generar rutas de salida usando la función centralizada
    results_path = os.path.join(output_dir, config.get_file_pattern('results_csv', sample_name=sample_name))
    plot_path = os.path.join(output_dir, config.get_file_pattern('plot', sample_name=sample_name))
    config_path = os.path.join(output_dir, 'constants.json')
    
    logger.info(f"Procesando muestra: {sample_path}")
    
    # Leer señales de la muestra
    try:
        I488, I405 = read_ssc_signals(sample_path)
        logger.info(f"Leídas {len(I488)} partículas")
    except Exception as e:
        logger.error(f"Error al leer el archivo {sample_path}: {e}")
        return None, None, None
    
    # Verificar que hay suficientes datos
    if len(I488) < 5:
        logger.warning(f"Muy pocas partículas ({len(I488)}) en la muestra: {sample_path}")
        if len(I488) == 0:
            return None, None, None
    
    # Limitar el número de eventos si se especifica
    if max_events is not None and max_events < len(I488):
        logger.info(f"Limitando a {max_events} eventos")
        # Usar selección aleatoria para evitar sesgos
        indices = np.random.choice(len(I488), max_events, replace=False)
        I488 = I488[indices]
        I405 = I405[indices]
    
    # Calcular rangos de señal utilizando la función centralizada en Config
    # que aplica percentiles y es robusta ante outliers
    min_I488, max_I488 = config.get_signal_bounds(I488)
    min_I405, max_I405 = config.get_signal_bounds(I405)
    
    # Actualizamos las constantes para la inversión
    constants.update({
        'min_I488': min_I488,
        'max_I488': max_I488,
        'min_I405': min_I405,
        'max_I405': max_I405
    })
    
    # Actualizamos la configuración en memoria si se solicita guardar
    if save_config:
        config_updates = {
            'min_I488': min_I488,
            'max_I488': max_I488,
            'min_I405': min_I405,
            'max_I405': max_I405
        }
        
        # Aplicar actualizaciones a la configuración en memoria
        config.update(config_updates)
        
        # Guardar la configuración
        config.save(config_path)
        logger.info(f"Rangos de señal guardados en configuración: {config_path}")
    
    # Imprimir distribución de señales para análisis
    logger.info(f"Rangos de señal - I488: [{min_I488:.1f}, {max_I488:.1f}]")
    logger.info(f"Rangos de señal - I405: [{min_I405:.1f}, {max_I405:.1f}]")
    
    if dry_run:
        logger.info("Modo dry-run: validación completa, omitiendo inversión")
        return None, None, None
    
    # Determinar si usar paralelismo (usar configuración si no se especifica)
    if use_parallel is None:
        use_parallel = config.get('parallel_enabled', False)
    
    # Invertir señales para obtener diámetro e índice de refracción
    diameters, n_particles, converged = invert_scatter(
        I488, I405, constants, use_parallel=use_parallel, 
        lookup_table=lookup_table, n_jobs=n_jobs
    )
    
    # Calcular porcentaje de convergencia
    success_rate = float(np.sum(converged) / len(converged) * 100)
    logger.info(f"Éxito en inversión: {success_rate:.1f}% ({np.sum(converged)} de {len(converged)})")
    
    # Guardar resultados en CSV
    try:
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['diameter_nm', 'n_particle', 'converged', 'I488', 'I405'])
            for d, n, conv, i488, i405 in zip(diameters, n_particles, converged, I488, I405):
                # Convertir a tipos nativos de Python para CSV
                # Usar float('nan') para NaN para mantener la coherencia de tipos numéricos
                writer.writerow([
                    float(d) if not np.isnan(d) else float('nan'), 
                    float(n) if not np.isnan(n) else float('nan'), 
                    bool(conv), 
                    float(i488), 
                    float(i405)
                ])
        
        logger.info(f"Resultados guardados en: {results_path}")
    except Exception as e:
        logger.error(f"Error al guardar resultados: {e}")
    
    # Crear gráficas usando el módulo de plots
    try:
        fig = plot_scatter_results(
            diameters, n_particles, I488, converged, 
            output_path=plot_path, 
            title=f'Partículas en {sample_name}'
        )
    except Exception as e:
        logger.error(f"Error al generar gráfica: {e}")
    
    return diameters, n_particles, converged

def calibrate(args, config):
    """
    Ejecuta la calibración y guarda constantes
    
    Args:
        args: Argumentos de línea de comandos
        config: Instancia de Config
    
    Returns:
        Diccionario con constantes de calibración
    """
    calibration_path = args.calibration_file
    output_dir = args.output_dir
    ensure_dir(output_dir)
    
    # Verificar que el archivo de calibración exista
    if not os.path.exists(calibration_path):
        logger.error(f"Archivo de calibración no encontrado: {calibration_path}")
        return None
    
    logger.info(f"Iniciando calibración desde: {calibration_path}")
    
    try:
        # Cargar datos de calibración sin actualizar config (modificado)
        n_med, n_part_ref, diam_ref, I488_ref, I405_ref = load_calibration(calibration_path)
        
        # Calcular factores K para ambas longitudes de onda
        angle_range = config.angle_range
        
        logger.info(f"Calibrando para rango de ángulos: {angle_range}")
        
        logger.info("Ajustando factor K para 488 nm (espacio log-log con offset)...")
        K488, B488 = fit_K(diam_ref, I488_ref, sigma_sca_ssc, config.lambda_blue, n_part_ref, n_med, angle_range, show_progress=args.progress)
        
        logger.info("Ajustando factor K para 405 nm (espacio log-log con offset)...")
        K405, B405 = fit_K(diam_ref, I405_ref, sigma_sca_ssc, config.lambda_violet, n_part_ref, n_med, angle_range, show_progress=args.progress)
        
        # Evaluar calidad del ajuste incluyendo offset
        quality_488 = evaluate_calibration_quality(
            diam_ref, I488_ref, K488, B488, 
            n_particle=n_part_ref,
            wavelength=config.lambda_blue, 
            n_medium=n_med, 
            angle_range=angle_range
        )
        quality_405 = evaluate_calibration_quality(
            diam_ref, I405_ref, K405, B405, 
            n_particle=n_part_ref,
            wavelength=config.lambda_violet, 
            n_medium=n_med, 
            angle_range=angle_range
        )
        
        logger.info(f"Calidad de ajuste 488 nm: R² = {quality_488['r_squared']:.4f}, "
                  f"R²(log) = {quality_488['r_squared_log']:.4f}, "
                  f"Error relativo medio = {quality_488['mean_rel_error']:.2f}%")
        logger.info(f"Calidad de ajuste 405 nm: R² = {quality_405['r_squared']:.4f}, "
                  f"R²(log) = {quality_405['r_squared_log']:.4f}, "
                  f"Error relativo medio = {quality_405['mean_rel_error']:.2f}%")
        
        # Verificar calidad mínima del ajuste usando R² logarítmico
        min_r_squared = 0.8
        if quality_488['r_squared_log'] < min_r_squared or quality_405['r_squared_log'] < min_r_squared:
            logger.warning(f"La calidad del ajuste log-log está por debajo del umbral (R² < {min_r_squared}).")
            logger.warning("Revise los datos de calibración y parámetros.")
        
        # Calcular rangos de señales usando la función centralizada
        min_I488, max_I488 = config.get_signal_bounds(I488_ref)
        min_I405, max_I405 = config.get_signal_bounds(I405_ref)
        
        # Preparar actualizaciones sin aplicar automáticamente
        constants_updates = {
            'n_medium': float(n_med),
            'angle_range': [float(angle) for angle in angle_range],
            'K488': float(K488),
            'K405': float(K405),
            'B488': float(B488),
            'B405': float(B405),
            'min_I488': min_I488,
            'max_I488': max_I488,
            'min_I405': min_I405,
            'max_I405': max_I405,
            'n_particle_ref': float(n_part_ref),
            'mie_n_points': 300  # Aumentar la resolución angular a 300 puntos
        }
        
        # Actualizar config en memoria
        config.update(constants_updates)
        
        # Guardar en disco
        config_path = os.path.join(output_dir, 'constants.json')
        config.save(config_path)
        logger.info(f"Constantes de calibración guardadas en: {config_path}")
        
        # Precalcular curvas teóricas para gráficos con los nuevos parámetros
        # Usar el argumento de la línea de comandos para mostrar/ocultar la barra de progreso
        theory_curves = calculate_theory_curves(
            diam_ref, K488, K405, B488, B405, 
            extended_range=True, log_spacing=True,
            show_progress=args.progress
        )
        
        # Generar gráfica de calibración con eje X lineal y eje Y logarítmico (log-lin)
        calib_plot_path = get_output_path(None, 'calib_plot')
        plot_calibration(
            diam_ref, I488_ref, I405_ref, K488, K405, B488, B405,
            output_path=calib_plot_path,
            theory_curves=theory_curves,
            log_x=False  # Usar escala lineal en X
        )
        
        return constants_updates
    
    except Exception as e:
        logger.error(f"Error durante la calibración: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def process_batch(args, config):
    """
    Procesa un lote de muestras FCS
    
    Args:
        args: Argumentos de línea de comandos
        config: Instancia de Config
    
    Returns:
        Lista de resultados para cada muestra
    """
    sample_dir = args.sample_dir
    pattern = args.pattern
    output_dir = args.output_dir
    use_parallel = args.parallel
    use_lookup = args.lookup_table
    n_jobs = args.n_jobs
    dry_run = args.dry_run
    save_config = args.save_config
    
    # Verificar que el directorio existe
    if not os.path.exists(sample_dir):
        logger.error(f"Directorio no encontrado: {sample_dir}")
        return
    
    # Buscar archivos que coincidan con el patrón
    import glob
    sample_paths = glob.glob(os.path.join(sample_dir, pattern))
    
    if not sample_paths:
        logger.error(f"No se encontraron archivos que coincidan con {pattern} en {sample_dir}")
        return
    
    # Ordenar por nombre para consistencia
    sample_paths.sort()
    
    logger.info(f"Encontrados {len(sample_paths)} archivos para procesar")
    
    # Crear tabla de lookup si se solicita
    lookup_table = None
    if use_lookup and not dry_run:
        # Verificar que existen constantes de calibración
        if config.get('K488') is None or config.get('K405') is None:
            logger.error("No se encontraron constantes de calibración. Ejecute primero el comando 'calibrate'.")
            return
        
        logger.info("Creando tabla de lookup...")
        try:
            lookup_table = create_lookup_table({
                'K488': config.get('K488'),
                'K405': config.get('K405'),
                'n_medium': config.n_medium,
                'angle_range': config.angle_range
            })
        except Exception as e:
            logger.error(f"Error al crear tabla de lookup: {e}")
            lookup_table = None
    
    # Asegurar que exista el directorio de salida
    ensure_dir(output_dir)
    
    # Procesar cada muestra
    all_results = []
    sample_names = []
    
    for sample_path in sample_paths:
        sample_name = get_sample_name(sample_path)
        sample_names.append(sample_name)
        
        # Crear directorio específico para esta muestra dentro del output_dir
        sample_output_dir = output_dir
        
        # Procesar la muestra
        try:
            diameters, n_particles, converged = process_sample(
                sample_path, config, sample_output_dir, 
                use_parallel=use_parallel, lookup_table=lookup_table, 
                max_events=args.max_events, sample_index=args.sample_index,
                n_jobs=n_jobs, dry_run=dry_run, save_config=save_config
            )
            
            if not dry_run and diameters is not None:
                all_results.append((diameters, n_particles, converged))
        except Exception as e:
            logger.error(f"Error al procesar {sample_path}: {e}")
    
    # Crear gráfica comparativa
    if args.compare and len(sample_names) > 1 and len(all_results) > 1 and not dry_run:
        try:
            comparison_path = os.path.join(output_dir, 'comparison.png')
            diameters_list = [res[0] for res in all_results]
            n_particles_list = [res[1] for res in all_results]
            
            # Usar solo nombres de archivos procesados exitosamente
            valid_names = sample_names[:len(all_results)]
            
            plot_comparison(valid_names, diameters_list, n_particles_list, 
                          output_path=comparison_path)
            logger.info(f"Gráfica comparativa guardada en: {comparison_path}")
        except Exception as e:
            logger.error(f"Error al generar gráfica comparativa: {e}")
    
    return all_results

def validate_range_parameters(args, config):
    """
    Valida que los parámetros de rango especificados sean coherentes.
    (d_min < d_max, n_min < n_max)
    
    Args:
        args: Argumentos de línea de comandos parseados
        config: Configuración actual
        
    Returns:
        Tupla (es_válido, mensaje_error)
    """
    # Si no hay rangos en los argumentos, no hay nada que validar
    has_ranges = hasattr(args, 'd_min') or hasattr(args, 'd_max') or hasattr(args, 'n_min') or hasattr(args, 'n_max')
    if not has_ranges:
        return True, ""
    
    # Obtener valores actuales de la configuración
    d_min = config.solver.get('diameter_min', 50)
    d_max = config.solver.get('diameter_max', 20000)
    n_min = config.solver.get('index_min', 1.01)
    n_max = config.solver.get('index_max', 1.8)
    
    # Aplicar valores de argumentos si se proporcionan
    if hasattr(args, 'd_min') and args.d_min is not None:
        d_min = args.d_min
    if hasattr(args, 'd_max') and args.d_max is not None:
        d_max = args.d_max
    if hasattr(args, 'n_min') and args.n_min is not None:
        n_min = args.n_min
    if hasattr(args, 'n_max') and args.n_max is not None:
        n_max = args.n_max
    
    # Validar relaciones de valores
    errors = []
    
    if d_min >= d_max:
        errors.append(f"Rango de diámetros inválido: d_min ({d_min}) debe ser menor que d_max ({d_max})")
    
    if n_min >= n_max:
        errors.append(f"Rango de índices inválido: n_min ({n_min}) debe ser menor que n_max ({n_max})")
    
    # Validar rangos físicamente razonables
    if d_min <= 0:
        errors.append(f"Diámetro mínimo inválido: d_min ({d_min}) debe ser positivo")
    
    if n_min <= 1.0:
        errors.append(f"Índice mínimo inválido: n_min ({n_min}) debe ser mayor que 1.0")
    
    if n_max > 3.0:
        errors.append(f"Índice máximo inválido: n_max ({n_max}) posiblemente fuera de rango físico (>3.0)")
    
    # Devolver resultado de validación
    if errors:
        return False, "\n".join(errors)
    else:
        return True, ""

def main():
    """Función principal"""
    # Configurar logging primero antes de cualquier otra operación
    log_level = logging.DEBUG if '--verbose' in sys.argv or '-v' in sys.argv else logging.INFO
    log_file = os.path.join('results', 'cytoflex.log') if os.path.exists('results') else None
    configure_logging(level=log_level, log_file=log_file)
    
    # Silenciar loggers ruidosos de bibliotecas externas
    quiet_noisy_loggers()
    
    # Inicializar configuración global
    config = get_config()
    
    # Configurar y procesar argumentos de línea de comandos
    parser = setup_parser()
    args = parser.parse_args()
    
    # Si no hay comando, mostrar ayuda
    if not args.command:
        parser.print_help()
        return 0  # Salir con código 0 (éxito)
    
    try:
        # Actualizar configuración con argumentos (sin guardar automáticamente)
        config, updates = update_config_from_args(args, config)
        
        # Validar parámetros de rango antes de crear directorios o reservar recursos
        is_valid, validation_message = validate_range_parameters(args, config)
        if not is_valid:
            # Usar parser.error para mostrar error y salir con código de error
            parser.error(f"Errores de validación en parámetros de rango:\n{validation_message}")
            return 1  # Código de error, aunque parser.error termina el proceso
        
        # Crear directorio de salida si no existe
        ensure_dir(args.output_dir)
        
        # Guardar la configuración si se solicita explícitamente
        save_config_if_requested(config, updates, args)
        
        # Si solo se quiere validar, mostrar configuración y terminar
        if args.just_validate:
            logger.info("Modo de validación - Configuración actual:")
            logger.info(f"  Ángulos: {config.angle_range}")
            logger.info(f"  n_medium: {config.n_medium}")
            logger.info(f"  Solver params: {config.solver_params}")
            logger.info(f"  Paralelismo: {getattr(args, 'parallel', False)}, jobs: {args.n_jobs or 'auto'}")
            logger.info("Validación completada.")
            return 0  # Éxito
        
        # Verificar comando
        if args.command == 'calibrate':
            result = calibrate(args, config)
            return 0 if result is not None else 1  # Devolver código según éxito
            
        elif args.command == 'process':
            # Intentar encontrar archivo de constantes si no se especifica
            if args.constants is None:
                args.constants = find_constants_file()
                if args.constants:
                    logger.info(f"Usando constantes de: {args.constants}")
                else:
                    logger.error("No se encontró archivo de constantes. Especifique uno con --constants o ejecute 'calibrate'.")
                    return 1  # Error
            
            # Cargar configuración específica para este comando
            if os.path.exists(args.constants):
                config.load_config(args.constants)
            else:
                logger.error(f"Archivo de constantes no encontrado: {args.constants}")
                return 1  # Error
            
            # Crear tabla de lookup si se solicita
            lookup_table = None
            if args.lookup_table and not args.dry_run:
                logger.info("Creando tabla de lookup...")
                try:
                    lookup_table = create_lookup_table({
                        'K488': config.get('K488'),
                        'K405': config.get('K405'),
                        'n_medium': config.n_medium,
                        'angle_range': config.angle_range
                    })
                except Exception as e:
                    logger.error(f"Error al crear tabla de lookup: {e}")
            
            # Procesar la muestra
            result = process_sample(
                args.sample, config, args.output_dir, 
                use_parallel=args.parallel, lookup_table=lookup_table, 
                max_events=args.max_events, sample_index=args.sample_index,
                n_jobs=args.n_jobs, dry_run=args.dry_run,
                save_config=args.save_config
            )
            return 0 if result[0] is not None else 1  # Código según resultado
            
        elif args.command == 'batch':
            # Intentar encontrar archivo de constantes si no se especifica
            if args.constants is None:
                args.constants = find_constants_file()
                if args.constants:
                    logger.info(f"Usando constantes de: {args.constants}")
                else:
                    logger.error("No se encontró archivo de constantes. Especifique uno con --constants o ejecute 'calibrate'.")
                    return 1  # Error
            
            # Cargar configuración específica para este comando
            if os.path.exists(args.constants):
                config.load_config(args.constants)
            else:
                logger.error(f"Archivo de constantes no encontrado: {args.constants}")
                return 1  # Error
            
            result = process_batch(args, config)
            return 0 if result is not None and len(result) > 0 else 1  # Código según éxito
            
        else:
            # Si no se especificó un comando válido, mostrar la ayuda
            parser.print_help()
            return 0  # No es un error, solo muestra ayuda
            
    except Exception as e:
        # Capturar excepciones no manejadas
        logger.error(f"Error no manejado: {e}")
        logger.debug("Detalles del error:", exc_info=True)
        return 1  # Error general
        
# Script entry point mejorado
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)  # Propaga código de salida al sistema