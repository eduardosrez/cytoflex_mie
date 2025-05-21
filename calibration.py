import json
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.special import spherical_jn, spherical_yn
import logging
from functools import lru_cache
from logging_config import get_logger
from config import get_config  # Agregar importación global para get_config

# Importar tqdm con degradación elegante si no está disponible
try:
    from tqdm import tqdm
except ImportError:
    # Función que simula tqdm pero no hace nada
    def tqdm(iterable, **kwargs):
        return iterable

from joblib import Parallel, delayed
# Ensure get_config is already imported: from config import get_config
# Ensure logger is available

# Reemplazar la configuración básica por un logger configurado centralmente
logger = get_logger('cytoflex_calibration')

def load_calibration(path):
    """
    Carga datos de calibración desde un archivo JSON.
    Solo devuelve los datos, sin efectos secundarios.
    
    Args:
        path: Ruta al archivo JSON con datos de calibración
        
    Returns:
        n_medium: Índice de refracción del medio
        n_particle: Índice de refracción de las partículas de referencia
        diameters: Array con diámetros de partículas de referencia [nm]
        I488: Array con intensidades SSC en 488 nm
        I405: Array con intensidades SSC en 405 nm
    """
    logger.info(f"Cargando datos de calibración: {path}")
    try:
        with open(path) as f:
            calib = json.load(f)
            
        # Metadatos
        n_medium = calib['metadata']['n_medium']
        
        # Manejar ambas posibles claves para índice de refracción de partícula
        if 'n_particles' in calib['metadata']:
            n_particle = calib['metadata']['n_particles']
        elif 'n_particle' in calib['metadata']:
            n_particle = calib['metadata']['n_particle']
        elif 'n_particle_ref' in calib['metadata']:
            n_particle = calib['metadata']['n_particle_ref']
        else:
            raise KeyError("No se encontró índice de refracción de partícula en el archivo de calibración. "
                         "Use 'n_particles', 'n_particle' o 'n_particle_ref' en metadata.")
        
        # Datos de calibración
        diameters = np.array([pt['diameter_nm'] for pt in calib['data']])
        I488 = np.array([pt['Blue_SSC-H'] for pt in calib['data']])
        I405 = np.array([pt['Violet_SSC-H'] for pt in calib['data']])
        
        # Validación de datos
        if np.any(diameters <= 0):
            logger.warning("Se encontraron diámetros negativos o cero. Filtrando datos inválidos.")
            valid_idx = diameters > 0
            diameters = diameters[valid_idx]
            I488 = I488[valid_idx]
            I405 = I405[valid_idx]
            
        if np.any(I488 <= 0) or np.any(I405 <= 0):
            logger.warning("Se encontraron intensidades negativas o cero. Filtrando datos inválidos.")
            valid_idx = (I488 > 0) & (I405 > 0)
            diameters = diameters[valid_idx]
            I488 = I488[valid_idx]
            I405 = I405[valid_idx]
            
        logger.info(f"Datos de calibración válidos: {len(diameters)} puntos")
        
        return n_medium, n_particle, diameters, I488, I405
    except Exception as e:
        logger.error(f"Error al cargar datos de calibración: {e}")
        raise

# Funciones auxiliares para configuración de Mie
def setup_mie_calculation(r, n_particle, λ, n_medium):
    """
    Configura parámetros comunes para cálculos de teoría de Mie
    
    Args:
        r: Radio de la partícula [metros]
        n_particle: Índice de refracción de la partícula (real o complejo)
        λ: Longitud de onda [metros]
        n_medium: Índice de refracción del medio
        
    Returns:
        Tupla con (k, x, m, nmax) o None si los parámetros son inválidos
    """
    # Validación de entrada - modificada para aceptar índices complejos
    if r <= 0 or np.real(n_particle) <= 0 or n_medium <= 0 or λ <= 0:
        logger.debug(f"Parámetros inválidos: r={r}, n_particle={n_particle}, n_medium={n_medium}, λ={λ}")
        return None
        
    # Parámetro de tamaño
    k = 2 * np.pi * n_medium / λ
    x = k * r
    
    # Validar parámetro de tamaño
    if x <= 0:
        logger.debug(f"Parámetro de tamaño x <= 0: {x}")
        return None
        
    # Índice relativo
    m = n_particle / n_medium
    
    # Número de términos a sumar (aproximación de Wiscombe)
    nmax = calculate_mie_nmax(x)
    
    return k, x, m, nmax

@lru_cache(maxsize=1024)
def _discretize_float(value, precision=1e-10):
    """
    Discretiza un valor flotante para permitir un uso eficiente de cache.
    
    Args:
        value: Valor flotante a discretizar
        precision: Precisión para la discretización (por defecto 1e-10)
        
    Returns:
        Valor discretizado como tupla (parte entera, parte decimal discretizada)
    """
    if np.iscomplex(value):
        real_part = round(value.real / precision) * precision
        imag_part = round(value.imag / precision) * precision
        return (real_part, imag_part)
    else:
        return round(value / precision) * precision

@lru_cache(maxsize=1024)
def mie_an_cached(n, x_discrete, m_discrete):
    """
    Versión cached del coeficiente de Mie a_n con discretización de parámetros.
    
    Args:
        n: Orden del coeficiente
        x_discrete: Parámetro de tamaño discretizado
        m_discrete: Índice relativo discretizado
        
    Returns:
        Coeficiente complejo a_n
    """
    # Lógica idéntica a mie_an
    mx = m_discrete * x_discrete
    
    jmx = spherical_jn(n, mx)
    jx = spherical_jn(n, x_discrete)
    
    djmx = spherical_jn(n, mx, derivative=True)
    djx = spherical_jn(n, x_discrete, derivative=True)
    
    h1x = jx + 1j * spherical_yn(n, x_discrete)
    dh1x = djx + 1j * spherical_yn(n, x_discrete, derivative=True)
    
    numerator = m_discrete * jmx * djx - jx * djmx
    denominator = m_discrete * jmx * dh1x - h1x * djmx
    
    return numerator / denominator

def mie_an(n, x, m, config=None):
    """
    Calcula el coeficiente de Mie a_n utilizando cache inteligente con discretización.
    
    Args:
        n: Orden del coeficiente
        x: Parámetro de tamaño (= k*r)
        m: Índice relativo (= n_particle / n_medium)
        config: Instancia de config para evitar importaciones repetidas (opcional)
        
    Returns:
        Coeficiente complejo a_n
    """
    # Obtener la configuración para determinar la precisión si no se proporciona
    if config is None:
        from config import get_config
        config = get_config()
    
    # Obtener precisión de la configuración
    precision = config.get('mie_cache_precision', 1e-10)
    
    # Discretizar los valores flotantes para mejorar el uso de cache
    x_discrete = _discretize_float(x, precision)
    m_discrete = _discretize_float(m, precision)
    
    # Llamar a la versión cacheada con valores discretizados
    return mie_an_cached(n, x_discrete, m_discrete)

@lru_cache(maxsize=1024)
def mie_bn_cached(n, x_discrete, m_discrete):
    """
    Versión cached del coeficiente de Mie b_n con discretización de parámetros.
    
    Args:
        n: Orden del coeficiente
        x_discrete: Parámetro de tamaño discretizado
        m_discrete: Índice relativo discretizado
        
    Returns:
        Coeficiente complejo b_n
    """
    # Lógica idéntica a mie_bn
    mx = m_discrete * x_discrete
    
    jmx = spherical_jn(n, mx)
    jx = spherical_jn(n, x_discrete)
    
    djmx = spherical_jn(n, mx, derivative=True)
    djx = spherical_jn(n, x_discrete, derivative=True)
    
    h1x = jx + 1j * spherical_yn(n, x_discrete)
    dh1x = djx + 1j * spherical_yn(n, x_discrete, derivative=True)
    
    numerator = jmx * djx - m_discrete * jx * djmx
    denominator = jmx * dh1x - m_discrete * h1x * djmx
    
    return numerator / denominator

def mie_bn(n, x, m):
    """
    Calcula el coeficiente de Mie b_n utilizando cache inteligente con discretización.
    
    Args:
        n: Orden del coeficiente
        x: Parámetro de tamaño (= k*r)
        m: Índice relativo (= n_particle / n_medium)
        
    Returns:
        Coeficiente complejo b_n
    """
    # Obtener la configuración para determinar la precisión
    from config import get_config
    config = get_config()
    precision = config.get('mie_cache_precision', 1e-10)
    
    # Discretizar los valores flotantes para mejorar el uso de cache
    x_discrete = _discretize_float(x, precision)
    m_discrete = _discretize_float(m, precision)
    
    # Llamar a la versión cacheada con valores discretizados
    return mie_bn_cached(n, x_discrete, m_discrete)

@lru_cache(maxsize=1024)
def pi_n_cached(n, u_discrete):
    """
    Versión cached de pi_n con discretización del parámetro u.
    
    Args:
        n: Orden de la función
        u_discrete: cos(θ) discretizado
        
    Returns:
        Valor de pi_n(u)
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return ((2*n-1)/(n-1)) * u_discrete * pi_n_cached(n-1, u_discrete) - (n/(n-1)) * pi_n_cached(n-2, u_discrete)

def pi_n(n, u):
    """
    Función angular pi_n para los cálculos de Mie, con cache inteligente.
    
    Args:
        n: Orden de la función
        u: cos(θ)
        
    Returns:
        Valor de pi_n(u)
    """
    # Obtener la configuración para determinar la precisión
    from config import get_config
    config = get_config()
    precision = config.get('mie_cache_precision', 1e-10)
    
    # Discretizar el valor flotante para mejorar el uso de cache
    u_discrete = _discretize_float(u, precision)
    
    # Llamar a la versión cacheada con valor discretizado
    return pi_n_cached(n, u_discrete)

@lru_cache(maxsize=1024)
def tau_n_cached(n, u_discrete):
    """
    Versión cached de tau_n con discretización del parámetro u.
    
    Args:
        n: Orden de la función
        u_discrete: cos(θ) discretizado
        
    Returns:
        Valor de tau_n(u)
    """
    return n * u_discrete * pi_n_cached(n, u_discrete) - (n+1) * pi_n_cached(n-1, u_discrete)

def tau_n(n, u):
    """
    Función angular tau_n para los cálculos de Mie, con cache inteligente.
    
    Args:
        n: Orden de la función
        u: cos(θ)
        
    Returns:
        Valor de tau_n(u)
    """
    # Obtener la configuración para determinar la precisión
    from config import get_config
    config = get_config()
    precision = config.get('mie_cache_precision', 1e-10)
    
    # Discretizar el valor flotante para mejorar el uso de cache
    u_discrete = _discretize_float(u, precision)
    
    # Llamar a la versión cacheada con valor discretizado
    return tau_n_cached(n, u_discrete)

def calculate_mie_nmax(x, safety_factor=None):
    """
    Calcula el número de términos a sumar en la serie de Mie (aproximación de Wiscombe)
    con límite adaptativo basado en el tamaño de partícula.
    
    Args:
        x: Parámetro de tamaño (= k*r)
        safety_factor: Factor de seguridad para multiplicar la estimación básica.
                      Si es None, se usa el valor de configuración.
            
    Returns:
        Número de términos a sumar (nmax)
    """
    # Obtener configuración actual (usando importación tardía para evitar ciclos)
    from config import get_config
    config = get_config()
    mie_params = config.scientific.mie_params
    
    # Usar valores de configuración en lugar de hardcodear
    absolute_max_nmax = mie_params['absolute_max_nmax']
    default_max_nmax = mie_params['max_nmax']
    large_threshold = mie_params['large_particle_threshold']
    small_factor = mie_params['small_particle_factor']
    large_factor = mie_params['large_particle_factor']
    
    # Usar safety_factor de configuración si no se especifica
    if safety_factor is None:
        safety_factor = mie_params['safety_factor']
    
    try:
        # Fórmula de Wiscombe para estimar nmax
        nmax = int(np.round(x + 4 * x**(1/3) + 2))
        
        # Aplicar factor de seguridad
        nmax = int(nmax * safety_factor)
            
        # Para partículas grandes, usar un límite adaptativo
        if x > large_threshold:
            # Escalar el máximo permitido basado en el parámetro de tamaño
            adaptive_max = min(int(x * large_factor), absolute_max_nmax)
            logger.debug(f"Partícula grande (x={x:.1f}): usando límite adaptativo nmax={adaptive_max}")
            return min(nmax, adaptive_max)
        else:
            # Para partículas pequeñas/medianas, usar el límite configurado
            return min(nmax, default_max_nmax)
            
    except (ValueError, RuntimeWarning):
        # Si hay problemas con el cálculo de nmax, usar una aproximación segura
        if x < 1:
            nmax = 3  # Valor mínimo seguro para partículas muy pequeñas
        elif x < large_threshold:
            nmax = int(min(max(10, x * small_factor), default_max_nmax))
        else:
            # Para partículas grandes, asegurar suficientes términos
            nmax = int(min(max(100, x * large_factor), absolute_max_nmax))
            
        logger.debug(f"Usando nmax alternativo para x={x}: nmax={nmax}")
        return nmax

def calculate_mie_coefficients(nmax, x, m):
    """
    Calcula y devuelve los coeficientes de Mie a_n y b_n para todos los órdenes hasta nmax.
    
    Args:
        nmax: Número máximo de términos
        x: Parámetro de tamaño (= k*r)
        m: Índice relativo (= n_particle / n_medium)
        
    Returns:
        Tupla con (lista de a_n, lista de b_n)
    """
    # Precalcular todos los coeficientes para mayor eficiencia
    an_vals = [mie_an(n, x, m) for n in range(1, nmax + 1)]
    bn_vals = [mie_bn(n, x, m) for n in range(1, nmax + 1)]
    
    return an_vals, bn_vals

def sigma_sca_total(r, n_particle, λ, n_medium):
    """
    Calcula la sección eficaz de dispersión total
    
    Args:
        r: Radio de la partícula [metros]
        n_particle: Índice de refracción de la partícula
        λ: Longitud de onda [metros]
        n_medium: Índice de refracción del medio
        
    Returns:
        Sección eficaz de dispersión total [m²]
    """
    # Configurar parámetros comunes
    setup = setup_mie_calculation(r, n_particle, λ, n_medium)
    if setup is None:
        return 0.0
        
    k, x, m, nmax = setup
    
    # Calcular los coeficientes
    an_vals, bn_vals = calculate_mie_coefficients(nmax, x, m)
    
    # Calcular la suma de los coeficientes
    sum_term = 0
    for n in range(1, nmax + 1):
        an = an_vals[n-1]
        bn = bn_vals[n-1]
        sum_term += (2*n + 1) * (np.abs(an)**2 + np.abs(bn)**2)
    
    # Cálculo final
    sigma = (2 * np.pi / k**2) * sum_term
    
    return sigma

def sigma_sca_ssc(r, n_particle, λ, n_medium, angle_range):
    """
    Calcula la sección eficaz de dispersión para el rango de ángulos SSC
    
    Args:
        r: Radio de la partícula [metros]
        n_particle: Índice de refracción de la partícula
        λ: Longitud de onda [metros]
        n_medium: Índice de refracción del medio
        angle_range: Lista con [ángulo_mínimo, ángulo_máximo] en grados
        
    Returns:
        Sección eficaz de dispersión para el rango angular [m²], o np.nan si ocurre un error.
    """
    # Configurar parámetros comunes
    setup = setup_mie_calculation(r, n_particle, λ, n_medium)
    if setup is None:
        return 0.0
        
    k, x, m, nmax = setup
    
    # Validar el rango angular
    if not isinstance(angle_range, (list, tuple, np.ndarray)) or len(angle_range) != 2:
        logger.error(f"Rango angular inválido: {angle_range}. Debe ser una lista o tupla [min, max].")
        return 0.0
    
    if angle_range[0] >= angle_range[1]:
        logger.error(f"Rango angular inválido: [{angle_range[0]}, {angle_range[1]}]. Min debe ser menor que max.")
        return 0.0
    
    # Convertir ángulos a radianes
    theta_min, theta_max = np.deg2rad(angle_range)
    
    # Obtener parámetros de integración angular de la configuración
    config = get_config()
    n_points = config.get('mie_n_points', 100)
    
    # Definir la malla angular
    theta_vals = np.linspace(theta_min, theta_max, n_points)
    dtheta = (theta_max - theta_min) / (n_points - 1)
    
    # Precalcular cos(θ) y funciones angulares para eficiencia
    costheta_vals = np.cos(theta_vals)
    pi_n_vals = {}
    tau_n_vals = {}
    
    # Calcular pi_n y tau_n para todos los órdenes y ángulos
    for n in range(1, nmax + 1):
        pi_n_vals[n] = np.array([pi_n(n, u) for u in costheta_vals])
        tau_n_vals[n] = np.array([tau_n(n, u) for u in costheta_vals])
    
    # Precalcular coeficientes de Mie
    an_vals, bn_vals = calculate_mie_coefficients(nmax, x, m)
    
    # Calcular la sección eficaz diferencial para cada ángulo
    dsigma_values = np.zeros(n_points)
    
    for i, theta in enumerate(theta_vals):
        u = costheta_vals[i]
        S1 = 0
        S2 = 0
        
        for n in range(1, nmax + 1):
            an = an_vals[n-1]
            bn = bn_vals[n-1]
            
            pi_n_val = pi_n_vals[n][i]
            tau_n_val = tau_n_vals[n][i]
            
            term = (2*n + 1) / (n * (n + 1))
            S1 += term * (an * pi_n_val + bn * tau_n_val)
            S2 += term * (an * tau_n_val + bn * pi_n_val)
        
        dsigma = (np.abs(S1)**2 + np.abs(S2)**2) / k**2
        dsigma_values[i] = dsigma * 2 * np.pi * np.sin(theta)
    
    # Integración numérica usando regla del trapecio para mayor precisión
    sigma_ssc = np.trapz(dsigma_values, dx=dtheta)
    
    # Protección contra valores no físicos
    if not np.isfinite(sigma_ssc) or sigma_ssc < 0:
        logger.warning(
            f"Non-physical sigma_ssc encountered: {sigma_ssc:.4e}. "
            f"Inputs: r={r:.4e} m, n_particle={n_particle}, lambda={λ:.4e} m. Returning NaN."
        )
        return np.nan
    
    return sigma_ssc

def mie_2layer_coeffs(x_core, x_total, m_core, m_shell):
    """
    Calcula los coeficientes de Mie para una partícula de dos capas (núcleo-corteza)
    siguiendo las fórmulas de Bohren & Huffman, sección 4.4.
    
    Args:
        x_core: Parámetro de tamaño del núcleo (= k*r_core)
        x_total: Parámetro de tamaño de la partícula total (= k*r_total)
        m_core: Índice relativo del núcleo (= (n_core + i*k_core) / n_medium)
        m_shell: Índice relativo de la corteza (= (n_shell + i*k_shell) / n_medium)
        
    Returns:
        Tupla con (lista de a_n, lista de b_n) para la partícula núcleo-corteza
    
    Raises:
        ValueError: If numerical issues occur (e.g. non-finite intermediate values).
        ZeroDivisionError: If a division by zero is attempted.
        Exception: Other exceptions from underlying math functions re-raised after logging.
    """
    # Obtener configuración actual
    from config import get_config
    config = get_config()
    
    # Determinar nmax para la partícula total
    nmax = calculate_mie_nmax(x_total)
    
    # Listas para almacenar coeficientes
    an_vals = []
    bn_vals = []
    
    # Calcular coeficientes para cada orden n
    for n in range(1, nmax + 1):
        # Funciones de Bessel para el núcleo
        psi_mx1 = spherical_jn(n, m_core * x_core)
        dpsi_mx1 = spherical_jn(n, m_core * x_core, derivative=True)
        
        # Funciones de Bessel para la corteza (en r = r_core)
        psi_mx2_core = spherical_jn(n, m_shell * x_core)
        dpsi_mx2_core = spherical_jn(n, m_shell * x_core, derivative=True)
        xi_mx2_core = psi_mx2_core + 1j * spherical_yn(n, m_shell * x_core)
        dxi_mx2_core = dpsi_mx2_core + 1j * spherical_yn(n, m_shell * x_core, derivative=True)
        
        # Funciones de Bessel para la corteza (en r = r_total)
        psi_mx2_total = spherical_jn(n, m_shell * x_total)
        dpsi_mx2_total = spherical_jn(n, m_shell * x_total, derivative=True)
        xi_mx2_total = psi_mx2_total + 1j * spherical_yn(n, m_shell * x_total)
        dxi_mx2_total = dpsi_mx2_total + 1j * spherical_yn(n, m_shell * x_total, derivative=True)
        
        # Funciones de Bessel para el medio externo (en r = r_total)
        psi_x = spherical_jn(n, x_total)
        dpsi_x = spherical_jn(n, x_total, derivative=True)
        xi_x = psi_x + 1j * spherical_yn(n, x_total)
        dxi_x = dpsi_x + 1j * spherical_yn(n, x_total, derivative=True)
        
        # Calcular coeficientes auxiliares
        # Bohren & Huffman, Ecuaciones (4.55)-(4.60)
        try:
            m_ratio = m_core / m_shell
            if not np.isfinite(m_ratio):
                raise ValueError(f"m_ratio is {m_ratio}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during m_ratio calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}"
            )
            raise

        # Condiciones de contorno en r = r_core
        try:
            an1 = m_ratio * dpsi_mx1 * psi_mx2_core - psi_mx1 * dpsi_mx2_core
            if not np.isfinite(an1):
                raise ValueError(f"an1 is {an1}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during an1 calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: m_ratio={m_ratio}, dpsi_mx1={dpsi_mx1}, psi_mx2_core={psi_mx2_core}, psi_mx1={psi_mx1}, dpsi_mx2_core={dpsi_mx2_core}"
            )
            raise
        
        try:
            an2 = m_ratio * dpsi_mx1 * xi_mx2_core - psi_mx1 * dxi_mx2_core
            if not np.isfinite(an2):
                raise ValueError(f"an2 is {an2}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during an2 calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: m_ratio={m_ratio}, dpsi_mx1={dpsi_mx1}, xi_mx2_core={xi_mx2_core}, psi_mx1={psi_mx1}, dxi_mx2_core={dxi_mx2_core}"
            )
            raise

        try:
            bn1 = psi_mx1 * dpsi_mx2_core - m_ratio * dpsi_mx1 * psi_mx2_core
            if not np.isfinite(bn1):
                raise ValueError(f"bn1 is {bn1}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during bn1 calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: psi_mx1={psi_mx1}, dpsi_mx2_core={dpsi_mx2_core}, m_ratio={m_ratio}, dpsi_mx1={dpsi_mx1}, psi_mx2_core={psi_mx2_core}"
            )
            raise

        try:
            bn2 = psi_mx1 * dxi_mx2_core - m_ratio * dpsi_mx1 * xi_mx2_core
            if not np.isfinite(bn2):
                raise ValueError(f"bn2 is {bn2}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during bn2 calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: psi_mx1={psi_mx1}, dxi_mx2_core={dxi_mx2_core}, m_ratio={m_ratio}, dpsi_mx1={dpsi_mx1}, xi_mx2_core={xi_mx2_core}"
            )
            raise
        
        # Calcular coeficiente G_n y auxiliar para an
        try:
            term_an_G_num1 = an2 / an1
            term_an_G_num2 = psi_mx2_total / xi_mx2_total
            term_an_G_den1 = term_an_G_num1 # an2 / an1
            term_an_G_den2 = term_an_G_num2 * (dpsi_mx2_total / dxi_mx2_total) # (psi_mx2_total / xi_mx2_total) * (dpsi_mx2_total / dxi_mx2_total)

            if not np.isfinite(term_an_G_num1) or not np.isfinite(term_an_G_num2) or \
               not np.isfinite(dpsi_mx2_total) or not np.isfinite(dxi_mx2_total) or \
               not np.isfinite(term_an_G_den2):
                raise ValueError(f"Non-finite intermediate in G_n for an: an1={an1}, an2={an2}, psi_mx2_total={psi_mx2_total}, xi_mx2_total={xi_mx2_total}, dpsi_mx2_total={dpsi_mx2_total}, dxi_mx2_total={dxi_mx2_total}")

            G_n_an_num = term_an_G_num1 * term_an_G_num2 - (dpsi_mx2_total / dxi_mx2_total)
            G_n_an_den = term_an_G_den1 - term_an_G_den2
            
            if G_n_an_den == 0:
                raise ZeroDivisionError("Denominator for G_n (an) is zero")
            G_n_an = G_n_an_num / G_n_an_den
            if not np.isfinite(G_n_an):
                raise ValueError(f"G_n_an is {G_n_an}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during G_n (an) calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: an1={an1}, an2={an2}, psi_mx2_total={psi_mx2_total}, xi_mx2_total={xi_mx2_total}, dpsi_mx2_total={dpsi_mx2_total}, dxi_mx2_total={dxi_mx2_total}"
            )
            raise

        # Calcular coeficiente a_n
        try:
            num_a = m_shell * psi_x * dpsi_mx2_total - psi_mx2_total * dpsi_x + G_n_an * (m_shell * psi_x * dxi_mx2_total - xi_mx2_total * dpsi_x)
            den_a = m_shell * xi_x * dpsi_mx2_total - psi_mx2_total * dxi_x + G_n_an * (m_shell * xi_x * dxi_mx2_total - xi_mx2_total * dxi_x)
            if den_a == 0:
                raise ZeroDivisionError("Denominator for an is zero")
            an = num_a / den_a
            if not np.isfinite(an):
                raise ValueError(f"an is {an}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during an calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: G_n_an={G_n_an}, num_a={num_a}, den_a={den_a}"
            )
            raise
        
        # Calcular coeficiente G_n y auxiliar para bn
        try:
            term_bn_G_num1 = bn2 / bn1
            term_bn_G_num2 = psi_mx2_total / xi_mx2_total
            term_bn_G_den1 = term_bn_G_num1 # bn2 / bn1
            term_bn_G_den2 = term_bn_G_num2 * (dpsi_mx2_total / dxi_mx2_total) # (psi_mx2_total / xi_mx2_total) * (dpsi_mx2_total / dxi_mx2_total)

            if not np.isfinite(term_bn_G_num1) or not np.isfinite(term_bn_G_num2) or \
               not np.isfinite(dpsi_mx2_total) or not np.isfinite(dxi_mx2_total) or \
               not np.isfinite(term_bn_G_den2):
                raise ValueError(f"Non-finite intermediate in G_n for bn: bn1={bn1}, bn2={bn2}, psi_mx2_total={psi_mx2_total}, xi_mx2_total={xi_mx2_total}, dpsi_mx2_total={dpsi_mx2_total}, dxi_mx2_total={dxi_mx2_total}")

            G_n_bn_num = term_bn_G_num1 * term_bn_G_num2 - (dpsi_mx2_total / dxi_mx2_total)
            G_n_bn_den = term_bn_G_den1 - term_bn_G_den2
            if G_n_bn_den == 0:
                raise ZeroDivisionError("Denominator for G_n (bn) is zero")
            G_n_bn = G_n_bn_num / G_n_bn_den
            if not np.isfinite(G_n_bn):
                raise ValueError(f"G_n_bn is {G_n_bn}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during G_n (bn) calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: bn1={bn1}, bn2={bn2}, psi_mx2_total={psi_mx2_total}, xi_mx2_total={xi_mx2_total}, dpsi_mx2_total={dpsi_mx2_total}, dxi_mx2_total={dxi_mx2_total}"
            )
            raise
        
        # Calcular coeficiente b_n
        try:
            num_b = psi_mx2_total * dpsi_x - m_shell * psi_x * dpsi_mx2_total + G_n_bn * (xi_mx2_total * dpsi_x - m_shell * psi_x * dxi_mx2_total)
            den_b = psi_mx2_total * dxi_x - m_shell * xi_x * dpsi_mx2_total + G_n_bn * (xi_mx2_total * dxi_x - m_shell * xi_x * dxi_mx2_total)
            if den_b == 0:
                raise ZeroDivisionError("Denominator for bn is zero")
            bn = num_b / den_b
            if not np.isfinite(bn):
                raise ValueError(f"bn is {bn}")
        except Exception as e_inner:
            logger.warning(
                f"Error in mie_2layer_coeffs for n={n} during bn calculation: {e_inner}. "
                f"Inputs: x_core={x_core}, x_total={x_total}, m_core={m_core}, m_shell={m_shell}. "
                f"Intermediates: G_n_bn={G_n_bn}, num_b={num_b}, den_b={den_b}"
            )
            raise
        
        # Añadir a las listas de coeficientes
        an_vals.append(an)
        bn_vals.append(bn)
    
    return an_vals, bn_vals

def sigma_sca_ssc_coreshell(r_core, t_shell, n_core, k_core, n_shell, k_shell, λ, n_med, angle_range):
    """
    Calcula la sección eficaz de dispersión para una partícula de núcleo-corteza
    en el rango de ángulos SSC.
    
    Args:
        r_core: Radio del núcleo [metros]
        t_shell: Espesor de la corteza [metros]
        n_core: Parte real del índice de refracción del núcleo
        k_core: Parte imaginaria del índice de refracción del núcleo
        n_shell: Parte real del índice de refracción de la corteza
        k_shell: Parte imaginaria del índice de refracción de la corteza
        λ: Longitud de onda [metros]
        n_med: Índice de refracción del medio
        angle_range: Lista con [ángulo_mínimo, ángulo_máximo] en grados
        
    Returns:
        Sección eficaz de dispersión para el rango angular [m²], o np.nan si ocurre un error.
    """
    # Validación de parámetros de entrada
    if r_core <= 0 or t_shell < 0 or n_core <= 0 or n_shell <= 0 or n_med <= 0 or λ <= 0:
        logger.debug(f"Parámetros inválidos en sigma_sca_ssc_coreshell: r_core={r_core}, t_shell={t_shell}, "
                    f"n_core={n_core}, k_core={k_core}, n_shell={n_shell}, k_shell={k_shell}")
        return 0.0
    
    # Calcular el radio total
    r_total = r_core + t_shell
    
    # Índices complejos
    m_core = (n_core + 1j * k_core) / n_med
    m_shell = (n_shell + 1j * k_shell) / n_med
    
    # Calcular parámetros de tamaño
    k = 2 * np.pi * n_med / λ
    x_total = k * r_total
    x_core = k * r_core
    
    # Validar el rango angular
    if not isinstance(angle_range, (list, tuple, np.ndarray)) or len(angle_range) != 2:
        logger.error(f"Rango angular inválido: {angle_range}. Debe ser una lista o tupla [min, max].")
        return 0.0
    
    if angle_range[0] >= angle_range[1]:
        logger.error(f"Rango angular inválido: [{angle_range[0]}, {angle_range[1]}]. Min debe ser menor que max.")
        return 0.0
    
    # Convertir ángulos a radianes
    theta_min, theta_max = np.deg2rad(angle_range)
    
    # Obtener parámetros de integración angular de la configuración
    config = get_config()
    n_points = config.get('mie_n_points', 100)
    
    # Definir la malla angular
    theta_vals = np.linspace(theta_min, theta_max, n_points)
    dtheta = (theta_max - theta_min) / (n_points - 1)
    
    # Precalcular cos(θ) para todos los ángulos
    costheta_vals = np.cos(theta_vals)
    
    # Calcular coeficientes de Mie para el modelo núcleo-corteza
    try:
        an_vals, bn_vals = mie_2layer_coeffs(x_core, x_total, m_core, m_shell)
        nmax = len(an_vals)
    except Exception as e:
        logger.error(
            f"Error calculating core-shell Mie coefficients: {e}. "
            f"Inputs: r_core={r_core:.4e}, t_shell={t_shell:.4e}, "
            f"n_core={n_core}, k_core={k_core}, n_shell={n_shell}, k_shell={k_shell}, "
            f"lambda={λ:.4e}. Returning NaN."
        )
        return np.nan
    
    # Precalcular funciones angulares para todos los órdenes y ángulos
    pi_n_vals = {}
    tau_n_vals = {}
    
    for n in range(1, nmax + 1):
        pi_n_vals[n] = np.array([pi_n(n, u) for u in costheta_vals])
        tau_n_vals[n] = np.array([tau_n(n, u) for u in costheta_vals])
    
    # Calcular la sección eficaz diferencial para cada ángulo
    dsigma_values = np.zeros(n_points)
    
    for i, theta in enumerate(theta_vals):
        u = costheta_vals[i]
        S1 = 0
        S2 = 0
        
        for n in range(1, nmax + 1):
            an = an_vals[n-1]
            bn = bn_vals[n-1]
            
            pi_n_val = pi_n_vals[n][i]
            tau_n_val = tau_n_vals[n][i]
            
            term = (2*n + 1) / (n * (n + 1))
            S1 += term * (an * pi_n_val + bn * tau_n_val)
            S2 += term * (an * tau_n_val + bn * pi_n_val)
        
        dsigma = (np.abs(S1)**2 + np.abs(S2)**2) / k**2
        dsigma_values[i] = dsigma * 2 * np.pi * np.sin(theta)
    
    # Integración numérica usando regla del trapecio para mayor precisión
    sigma_ssc = np.trapz(dsigma_values, dx=dtheta)
    
    # Protección contra valores no físicos
    if not np.isfinite(sigma_ssc) or sigma_ssc < 0:
        logger.warning(
            f"Non-physical sigma_ssc_coreshell encountered: {sigma_ssc:.4e}. "
            f"Inputs: r_core={r_core:.4e}, t_shell={t_shell:.4e}, n_core={n_core}, k_core={k_core}, "
            f"n_shell={n_shell}, k_shell={k_shell}, lambda={λ:.4e} m. Returning NaN."
        )
        return np.nan
    
    return sigma_ssc

def fit_K(diameters, intensities, sigma_func, wavelength, n_particle, n_medium, angle_range, show_progress=False):
    """
    Ajusta el factor K que relaciona la sección eficaz de dispersión con las intensidades medidas.
    Utiliza un ajuste lineal con offset para capturar mejor la relación en todo el rango de tamaños.
    
    Args:
        diameters: Array con diámetros de partículas de referencia [nm]
        intensities: Array con intensidades medidas [a.u.]
        sigma_func: Función que calcula la sección eficaz (sigma_sca_ssc)
        wavelength: Longitud de onda [metros]
        n_particle: Índice de refracción de las partículas
        n_medium: Índice de refracción del medio
        angle_range: Rango de ángulos [min, max] en grados
        show_progress: Si se debe mostrar barra de progreso durante el cálculo
        
    Returns:
        K: Factor de calibración [a.u./m²]
        B: Offset de intensidad [a.u.]
    """
    logger.info(f"Ajustando factor K para λ={wavelength*1e9:.1f} nm con {len(diameters)} puntos")
    
    # Modelo de ajuste lineal con offset: I = K * σ + B
    def model_with_offset(d_nm, K, B):
        """
        d_nm: array de diámetros [nm]
        devuelve K * σ_sca_ssc(d_nm) + B
        """
        sigma_vals = calculate_sigma_array(
            diameters_nm=d_nm,
            sigma_func=sigma_func,
            n_particle=n_particle,
            wavelength=wavelength,
            n_medium=n_medium,
            angle_range=angle_range
        )
        return K * sigma_vals + B
    
    # Crear un iterador con o sin barra de progreso
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(1), desc=f"Ajustando K para λ={wavelength*1e9:.1f}nm")
        except ImportError:
            iterator = range(1)
    else:
        iterator = range(1)
    
    # Calcular secciones eficaces y realizar el ajuste
    try:
        # Valores iniciales estimados para mejor convergencia
        p0 = [1e18, 0]  # K inicial ~10^18, B inicial = 0
        
        # Límites para evitar factores K negativos
        # Permitimos offset B negativo si mejora el ajuste
        bounds = ([0, -np.inf], [np.inf, np.inf])
        
        # Ajustar con curve_fit usando el modelo con offset
        for _ in iterator:
            popt, pcov = curve_fit(model_with_offset, diameters, intensities, 
                                 p0=p0, bounds=bounds, method='trf')
        
        # Extraer parámetros ajustados
        K = popt[0]
        B = popt[1]
        
        # Calcular error estándar de los parámetros
        K_err = np.sqrt(pcov[0, 0])
        B_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0
        
        logger.info(f"Ajuste completado: K = {K:.3e} ± {K_err:.3e}, B = {B:.3e} ± {B_err:.3e}")
        
        return K, B
        
    except Exception as e:
        logger.error(f"Error en ajuste de K: {e}")
        # En caso de error, devolver valores por defecto
        return 1.0e18, 0.0

def evaluate_calibration_quality(diameters, intensities, K, B, n_particle, wavelength, n_medium, angle_range):
    """
    Evalúa la calidad del ajuste de calibración comparando mediciones con predicciones teóricas.
    
    Args:
        diameters: Array con diámetros de partículas de referencia [nm]
        intensities: Array con intensidades medidas [a.u.]
        K: Factor de calibración ajustado [a.u./m²]
        B: Offset de intensidad ajustado [a.u.]
        n_particle: Índice de refracción de las partículas
        wavelength: Longitud de onda [metros]
        n_medium: Índice de refracción del medio
        angle_range: Rango de ángulos [min, max] en grados
        
    Returns:
        Diccionario con métricas de calidad:
            r_squared: Coeficiente de determinación (espacio lineal)
            r_squared_log: Coeficiente de determinación (espacio logarítmico)
            mean_rel_error: Error relativo medio [%]
            max_rel_error: Error relativo máximo [%]
            abs_error: Array de errores absolutos
            rel_error: Array de errores relativos [%]
    """
    # Calcular intensidades teóricas
    I_theo = np.zeros_like(diameters, dtype=float)
    
    for i, d in enumerate(diameters):
        # Convertir diámetro [nm] a radio [m]
        r = d * 1e-9 / 2
        sigma = sigma_sca_ssc(r, n_particle, wavelength, n_medium, angle_range)
        I_theo[i] = K * sigma + B
    
    # Calcular errores y métricas
    abs_error = I_theo - intensities
    rel_error = abs_error / intensities * 100  # Error relativo en porcentaje
    
    # Calcular R² en espacio lineal
    mean_I = np.mean(intensities)
    total_sum_sq = np.sum((intensities - mean_I) ** 2)
    residual_sum_sq = np.sum(abs_error ** 2)
    r_squared = 1 - residual_sum_sq / total_sum_sq
    
    # Calcular R² en espacio logarítmico
    log_I = np.log10(intensities)
    log_I_theo = np.log10(I_theo)
    log_mean_I = np.mean(log_I)
    log_total_sum_sq = np.sum((log_I - log_mean_I) ** 2)
    log_residual_sum_sq = np.sum((log_I_theo - log_I) ** 2)
    r_squared_log = 1 - log_residual_sum_sq / log_total_sum_sq
    
    # Estadísticas sobre el error relativo
    mean_rel_error = np.mean(np.abs(rel_error))
    max_rel_error = np.max(np.abs(rel_error))
    
    return {
        'r_squared': r_squared,
        'r_squared_log': r_squared_log,
        'mean_rel_error': mean_rel_error,
        'max_rel_error': max_rel_error,
        'abs_error': abs_error,
        'rel_error': rel_error
    }

def calculate_sigma_array(
    diameters_nm: np.ndarray,
    sigma_func,
    n_particle: float,
    wavelength: float,
    n_medium: float,
    angle_range: list[float]
) -> np.ndarray:
    """
    Calcula un array de secciones eficaces para un conjunto de diámetros [nm].
    
    Args:
        diameters_nm: Array de diámetros en nanómetros.
        sigma_func: Función de dispersión (e.g. sigma_sca_ssc).
        n_particle: Índice de refracción de la partícula.
        wavelength: Longitud de onda [m].
        n_medium: Índice de refracción del medio.
        angle_range: [ángulo_mín, ángulo_máx] en grados.
    Returns:
        Numpy array con σ_sca para cada diámetro.
    """
    config = get_config()
    logger_cal = get_logger('cytoflex_calibration') # Ensure logger is accessible

    radii = diameters_nm * 1e-9 / 2

    parallel_enabled = config.solver.get('parallel_enabled', True)
    n_jobs = config.solver.get('parallel_jobs', -1)
    # Using a specific threshold for calibration, can be adjusted
    min_points_for_parallel = config.solver.get('min_calibration_points_for_parallel', 5) 

    if parallel_enabled and len(radii) >= min_points_for_parallel:
        logger_cal.debug(f"Calculating sigma array in parallel for {len(radii)} points (n_jobs={n_jobs}).")
        sigma_vals = Parallel(n_jobs=n_jobs)(
            delayed(sigma_func)(r, n_particle, wavelength, n_medium, angle_range) for r in radii
        )
    else:
        logger_cal.debug(f"Calculating sigma array sequentially for {len(radii)} points.")
        sigma_vals = []
        for r in radii:
            sigma = sigma_func(r, n_particle, wavelength, n_medium, angle_range)
            sigma_vals.append(sigma)
    
    return np.array(sigma_vals)