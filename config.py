import json
import os
import logging
from pathlib import Path
import numpy as np
import hashlib
import warnings
import threading

# Configuración del logger - Se reemplazará por configuración central
logger = logging.getLogger('cytoflex_config')

# Variable global para el singleton y lock para thread-safety
_CONFIG_INSTANCE = None
_CONFIG_LOCK = threading.Lock()

class BaseConfig:
    """Clase base para las configuraciones con funcionalidad común"""
    
    def __init__(self, config_dict=None):
        self._config = config_dict or {}
    
    def update(self, new_values):
        """Actualiza la configuración con nuevos valores"""
        self._config.update(new_values)
        return self
    
    def __getitem__(self, key):
        if key in self._config:
            return self._config[key]
        else:
            raise KeyError(f"Clave de configuración no encontrada: {key}")
    
    def __setitem__(self, key, value):
        self._config[key] = value
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def as_dict(self):
        return self._config.copy()


class ScientificConfig(BaseConfig):
    """Configuración de parámetros físicos y científicos"""
    
    DEFAULT_CONFIG = {
        # Parámetros físicos
        "n_medium": 1.333,  # Índice de refracción del medio (agua)
        "angle_range": [35.0, 145.0],  # Rango angular para SSC
        "n_particle_ref": 1.45,  # Índice referencia para partículas de calibración
        
        # Índices complejos y modelo núcleo-corteza
        "allow_complex_index": True,      # Si se permite usar índices complejos
        "core_shell_defaults": {          # Valores orientativos para el modelo núcleo-corteza
            "shell_thickness_nm": 20,     # Espesor de la pared celular [nm]
            "n_shell": 1.45,              # Índice de refracción real de la corteza
            "k_shell": 0.000,             # Índice de refracción imaginario de la corteza
            "n_core_prior": 1.39,         # Prior para índice real del núcleo
            "k_core_prior": 0.005         # Prior para índice imaginario del núcleo
        },
        
        # Longitudes de onda
        "lambda_blue": 488e-9,  # Longitud de onda azul [m]
        "lambda_violet": 405e-9,  # Longitud de onda violeta [m]
        
        # Parámetros Mie
        "mie_max_nmax": 100,  # Límite máximo para nmax
        "mie_n_points": 300,  # Número de puntos para integración angular (mayor -> más preciso pero más lento)
        "mie_absolute_max_nmax": 1000,  # Límite absoluto para nmax (partículas muy grandes)
        "mie_safety_factor": 2,  # Factor de seguridad para nmax (Wiscombe recomienda 2-3)
        "mie_approx_threshold": 1e-36,  # Factor para aproximación de Rayleigh [m²/nm^6]
        "mie_large_particle_threshold": 100,  # Umbral para considerar partícula grande (x > 100)
        "mie_small_particle_factor": 2,  # Factor para partículas pequeñas (nmax ≈ factor*x)
        "mie_large_particle_factor": 1.5,  # Factor para partículas grandes (nmax ≈ factor*x)
        "mie_cache_precision": 1e-10,  # Precisión para discretización en cache
    }
    
    def __init__(self, config_dict=None):
        super().__init__(config_dict or self.DEFAULT_CONFIG.copy())
    
    @property
    def n_medium(self):
        return self._config['n_medium']
    
    @property
    def angle_range(self):
        return self._config['angle_range']
    
    @property
    def lambda_blue(self):
        return self._config['lambda_blue']
    
    @property
    def lambda_violet(self):
        return self._config['lambda_violet']
    
    @property
    def n_particle_ref(self):
        return self._config['n_particle_ref']
    
    @property
    def mie_params(self):
        """Devuelve un diccionario con parámetros para cálculos Mie"""
        return {
            'max_nmax': self._config['mie_max_nmax'],
            'n_points': self._config['mie_n_points'],
            'absolute_max_nmax': self._config.get('mie_absolute_max_nmax', 1000),
            'safety_factor': self._config.get('mie_safety_factor', 2),
            'approx_threshold': self._config.get('mie_approx_threshold', 1e-36),
            'large_particle_threshold': self._config.get('mie_large_particle_threshold', 100),
            'small_particle_factor': self._config.get('mie_small_particle_factor', 2),
            'large_particle_factor': self._config.get('mie_large_particle_factor', 1.5)
        }
    
    @property
    def allow_complex_index(self):
        """Indica si se permite usar índices complejos"""
        return self._config.get('allow_complex_index', True)
    
    @property
    def core_shell_defaults(self):
        """Devuelve los valores por defecto para el modelo núcleo-corteza"""
        return self._config.get('core_shell_defaults', {
            "shell_thickness_nm": 20,
            "n_shell": 1.45,
            "k_shell": 0.000,
            "n_core_prior": 1.39,
            "k_core_prior": 0.005
        })
    
    @core_shell_defaults.setter
    def core_shell_defaults(self, value):
        """Establece los valores para el modelo núcleo-corteza"""
        self._config['core_shell_defaults'] = value
    
    @property
    def calibration_constants(self):
        """Devuelve las constantes de calibración y rangos de intensidad"""
        constants = {}
        # Factores de calibración
        calib_keys = ['K488', 'K405', 'B488', 'B405', 'min_I488', 'max_I488', 'min_I405', 'max_I405']
        for key in calib_keys:
            if key in self._config:
                constants[key] = self._config[key]
        return constants


class SolverConfig(BaseConfig):
    """Configuración para el solver y tablas de lookup"""
    
    DEFAULT_CONFIG = {
        # Parámetros de solver
        "diameter_min": 50,  # Diámetro mínimo para lookuptable [nm]
        "diameter_max": 20000,  # Diámetro máximo para lookuptable [nm]
        "diameter_step": 50,  # Paso diámetro para lookuptable [nm]
        "index_min": 1.01,  # Índice mínimo para lookuptable
        "index_max": 1.8,  # Índice máximo para lookuptable
        "index_step": 0.01,  # Paso índice para lookuptable
        
        # Parámetros para índices complejos
        "k_min": 0.000,  # Mínimo índice imaginario
        "k_max": 0.020,  # Máximo índice imaginario
        "k_step": 0.002,  # Paso de índice imaginario
        
        # Parámetros para modelo núcleo-corteza
        "shell_thickness_min": 5,    # Espesor mínimo de corteza [nm]
        "shell_thickness_max": 50,   # Espesor máximo de corteza [nm]
        "shell_thickness_step": 5,   # Paso para espesor de corteza [nm]
        
        # Parámetros de estrategia
        "solver_strategy": "sequential",  # Estrategia de inversión: "sequential" o "joint"
        
        # Parámetros iniciales para inversión
        "default_diameter_init": 500,  # Valor inicial de diámetro para inversión [nm]
        "default_index_init": 1.05,  # Valor inicial de índice para inversión
        "default_k_init": 0.001,     # Valor inicial para parte imaginaria del índice
        
        # Tolerancias y umbrales
        "solver_tolerance": 1e-6,  # Tolerancia para convergencia
        "solver_cost_threshold": 1e4,  # Umbral de costo para soluciones aceptables
        
        # Parámetros para tablas de lookup
        "lookup_table_enabled": False,  # Si usar tablas de lookup por defecto
        "lookup_table_path": "results/lookup_table.pkl",  # Ruta por defecto
        "lookup_table_compression": True,  # Si comprimir tabla con gzip
        "min_size_for_parallel": 5,  # Mínimo número de items (partículas en inversión, diámetros en lookup table) para activar paralelismo
        "min_calibration_points_for_parallel": 5, # Min number of calibration points to use parallel processing in calculate_sigma_array
        "parallel_enabled": True,  # Si usar paralelismo (activado por defecto para mayor velocidad)
        "parallel_jobs": -1,  # Número de trabajos para paralelización (-1 = todos)
    }
    
    def __init__(self, config_dict=None):
        super().__init__(config_dict or self.DEFAULT_CONFIG.copy())
    
    @property
    def solver_params(self):
        """Devuelve un diccionario con parámetros para el solver"""
        return {
            'diameter_range': (self._config['diameter_min'], self._config['diameter_max'], self._config['diameter_step']),
            'index_range': (self._config['index_min'], self._config['index_max'], self._config['index_step']),
            'k_range': (self._config.get('k_min', 0.0), self._config.get('k_max', 0.02), self._config.get('k_step', 0.002)),
            'shell_thickness_range': (
                self._config.get('shell_thickness_min', 5),
                self._config.get('shell_thickness_max', 50),
                self._config.get('shell_thickness_step', 5)
            ),
            'init_values': {
                'diameter': self._config['default_diameter_init'],
                'index': self._config['default_index_init'],
                'k': self._config.get('default_k_init', 0.001)
            },
            'strategy': self._config.get('solver_strategy', 'sequential')
        }


class IOConfig(BaseConfig):
    """Configuración para entrada/salida de archivos"""
    
    # Lista de patrones de salida requeridos
    REQUIRED_OUTPUT_PATTERNS = [
        'results_csv', 'plot', 'config', 'calib_plot'
    ]
    
    DEFAULT_CONFIG = {
        # Parámetros para FCS reader
        "fcs_header_size": 58,  # Tamaño predeterminado de cabecera FCS
        
        # Directorio base para archivos de salida
        "output_dir": "results",  # Directorio base para resultados
        
        # Patrones de nombres para salida
        "output_patterns": {
            "results_csv": "{sample_name}_results.csv",
            "plot": "{sample_name}_plot.png",
            "config": "constants.json",
            "calib_plot": "calibration_plot.png",
            "lookup_table": "lookup_table.pkl",
            "lookup_checkpoint": "lookup_checkpoint_{part}.pkl"
        },
        
        # Mapeo de canales en archivos FCS
        "channel_mappings": {
            # Configuración para diferentes tipos de archivos
            "default": {
                "blue_ssc_index": 1,  # Por defecto, índice 1 (canal 2)
                "violet_ssc_index": 2  # Por defecto, índice 2 (canal 3)
            },
            # Configuraciones específicas por tipo de archivo
            "synechococcus": {
                "blue_ssc_index": 1,
                "violet_ssc_index": 2
            },
            "prochlorococcus": {
                "blue_ssc_index": 1,
                "violet_ssc_index": 2
            },
            "picoeukaryotes": {
                "blue_ssc_index": 2,  # Índice 2 (canal 3, SSC-H)
                "violet_ssc_index": 4  # Índice 4 (canal 5, SSC_1-H)
            }
        },
        
        # Nombres de canal esperados (patrones para buscar en headers FCS)
        "channel_name_patterns": {
            "blue_ssc": ["SSC", "SSC-H", "SSC-A"],      # Patrones para canal SSC azul
            "violet_ssc": ["SSC_1", "SSC-H", "SSC_V", "SSC-V"]  # Patrones para canal SSC violeta
        }
    }
    
    def __init__(self, config_dict=None):
        super().__init__(config_dict or self.DEFAULT_CONFIG.copy())
        self._validate_output_patterns()
    
    def _validate_output_patterns(self):
        """Valida que los patrones de salida requeridos estén presentes"""
        if 'output_patterns' not in self._config:
            self._config['output_patterns'] = self.DEFAULT_CONFIG['output_patterns']
            return
            
        patterns = self._config['output_patterns']
        for pattern_key in self.REQUIRED_OUTPUT_PATTERNS:
            if pattern_key not in patterns:
                warnings.warn(f"Patrón de salida '{pattern_key}' no encontrado en configuración. "
                              f"Usando valor por defecto: '{self.DEFAULT_CONFIG['output_patterns'][pattern_key]}'")
                patterns[pattern_key] = self.DEFAULT_CONFIG['output_patterns'][pattern_key]
    
    @property
    def channel_mappings(self):
        return self._config['channel_mappings']
    
    @property
    def output_patterns(self):
        """Devuelve patrones de nombre para archivos de salida"""
        return self._config['output_patterns']
    
    @property
    def output_dir(self):
        """Directorio base para archivos de salida"""
        return self._config.get('output_dir', 'results')
    
    @property
    def fcs_header_size(self):
        """Tamaño de la cabecera FCS"""
        return self._config['fcs_header_size']
    
    @property
    def channel_name_patterns(self):
        """Patrones de nombres de canal para buscar en headers FCS"""
        return self._config.get('channel_name_patterns', self.DEFAULT_CONFIG['channel_name_patterns'])
    
    def get_file_pattern(self, pattern_key, **kwargs):
        """
        Obtiene un patrón de nombre de archivo formateado con kwargs
        
        Args:
            pattern_key: Clave del patrón a usar (ej: 'results_csv')
            **kwargs: Variables para formatear el patrón
            
        Returns:
            Patrón formateado
        """
        pattern = self.output_patterns.get(pattern_key)
        if not pattern:
            return None
        
        return pattern.format(**kwargs)
    
    def get_output_path(self, pattern_key, create_dirs=True, **kwargs):
        """
        Obtiene una ruta completa de salida (output_dir + patrón formateado)
        
        Args:
            pattern_key: Clave del patrón a usar
            create_dirs: Si se deben crear los directorios necesarios
            **kwargs: Variables para formatear el patrón
            
        Returns:
            Ruta completa de salida
        """
        filename = self.get_file_pattern(pattern_key, **kwargs)
        if not filename:
            return None
            
        # Construir ruta completa
        full_path = os.path.join(self.output_dir, filename)
        
        # Crear directorio si es necesario
        if create_dirs:
            os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
            
        return full_path
    
    def get_channel_mapping(self, filename):
        """
        Obtiene el mapeo de canales para un archivo específico.
        Busca en los patrones específicos y cae en 'default' si no encuentra coincidencia.
        
        Args:
            filename: Nombre del archivo para determinar qué mapeo usar
            
        Returns:
            Diccionario con mapeo de canales
        """
        filename_lower = filename.lower()
        mappings = self._config['channel_mappings']
        
        # Buscar mapeo específico basado en el nombre del archivo
        for pattern, mapping in mappings.items():
            if pattern != 'default' and pattern in filename_lower:
                logger.debug(f"Usando mapeo de canales para '{pattern}' en {filename}")
                return mapping
        
        # Si no hay coincidencia, usar el mapeo por defecto
        logger.debug(f"No se encontró mapeo específico para {filename}, usando 'default'")
        return mappings['default']
    
    def get_signal_bounds(self, signals, low_percentile=1, high_percentile=99, margin=1.2):
        """
        Calcula los rangos de señal usando percentiles para ser robusto ante outliers
        
        Args:
            signals: Array de señales a analizar
            low_percentile: Percentil inferior (por defecto 1%)
            high_percentile: Percentil superior (por defecto 99%)
            margin: Margen para ampliar el rango superior (por defecto 20%)
            
        Returns:
            Tupla (min_signal, max_signal)
        """
        signals = np.asarray(signals)
        positive_signals = signals[signals > 0]
        
        if len(positive_signals) > 0:
            min_signal = float(np.percentile(positive_signals, low_percentile))
        else:
            min_signal = 0
            
        max_signal = float(np.percentile(signals, high_percentile) * margin)
        
        return min_signal, max_signal


class VisualizationConfig(BaseConfig):
    """Configuración para visualización y gráficos"""
    
    DEFAULT_CONFIG = {
        # Parámetros para plots
        "plot_margins": {
            "diameter": 0.3,  # Margen para diámetros (30%)
            "intensity": 0.4,  # Margen para intensidades (40%)
            "index": 0.2      # Margen para índices (20%)
        },
        "plot_dpi": 150,      # DPI para gráficos
        "plot_figsize": [10, 8],  # Tamaño de figura por defecto
        "plot_bins": 50,      # Número de bins para histogramas
        "plot_scatter_cmap": "viridis",  # Colormap para scatter
        "plot_hist2d_cmap": "inferno",   # Colormap para histograma 2D
        "plot_color_488": "blue",        # Color para 488 nm
        "plot_color_405": "purple"       # Color para 405 nm
    }
    
    def __init__(self, config_dict=None):
        super().__init__(config_dict or self.DEFAULT_CONFIG.copy())
    
    @property
    def plot_params(self):
        """Devuelve un diccionario con parámetros para plots"""
        return {
            'margins': self._config['plot_margins'],
            'dpi': self._config['plot_dpi'],
            'figsize': self._config['plot_figsize'],
            'bins': self._config['plot_bins'],
            'scatter_cmap': self._config['plot_scatter_cmap'],
            'hist2d_cmap': self._config['plot_hist2d_cmap'],
            'color_488': self._config['plot_color_488'],
            'color_405': self._config['plot_color_405']
        }


class CalibrationConfig(BaseConfig):
    """Configuración para constantes de calibración"""
    
    DEFAULT_CONFIG = {
        # Factores de calibración
        "K488": None,  # Factor para 488 nm
        "K405": None,  # Factor para 405 nm
        "B488": 0.0,   # Offset para 488 nm
        "B405": 0.0,   # Offset para 405 nm
        
        # Rangos de intensidad válidos
        "min_I488": 0.0,   # Intensidad mínima 488 nm
        "max_I488": 1e8,   # Intensidad máxima 488 nm
        "min_I405": 0.0,   # Intensidad mínima 405 nm
        "max_I405": 1e8    # Intensidad máxima 405 nm
    }
    
    def __init__(self, config_dict=None):
        super().__init__(config_dict or self.DEFAULT_CONFIG.copy())
    
    @property
    def calibration_constants(self):
        """Devuelve todas las constantes de calibración"""
        return self.as_dict()


class Config:
    """
    Clase centralizada para gestionar la configuración de CytoFLEX Mie.
    Contiene subclases para separación de responsabilidades:
    - scientific: Parámetros físicos y científicos
    - solver: Parámetros de inversión y lookup
    - io: Configuración de entrada/salida
    - visualization: Parámetros de visualización
    """
    
    def __init__(self, config_path=None):
        """
        Inicializa la configuración desde el archivo constants.json
        
        Args:
            config_path: Ruta al archivo de configuración. Si es None,
                        intenta buscar en ubicaciones predeterminadas.
        """
        # Inicializar subconfigs con valores por defecto
        self.scientific = ScientificConfig()
        self.solver = SolverConfig()
        self.io = IOConfig()
        self.visualization = VisualizationConfig()
        self.calibration = CalibrationConfig()
        
        # Inicializar mapa de acceso rápido a claves
        self._key_map = self._build_key_map()
        
        # Si no se proporciona ruta, buscar en varios lugares
        if config_path is None:
            possible_paths = [
                'constants.json',
                'results/constants.json',
                os.path.join(os.path.dirname(__file__), 'constants.json'),
                os.path.join(os.path.dirname(__file__), 'results', 'constants.json')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            logger.warning("No se encontró archivo de configuración. Usando valores por defecto.")
    
    def _build_key_map(self):
        """
        Construye un mapa que relaciona cada clave con su subconfiguración
        para acceso eficiente
        
        Returns:
            Diccionario con mapeo {clave: nombre_subconfig}
        """
        key_map = {}
        
        # Añadir todas las claves de cada subconfig al mapa
        for subconfig_name, subconfig in [
            ('scientific', self.scientific),
            ('solver', self.solver),
            ('io', self.io),
            ('visualization', self.visualization),
            ('calibration', self.calibration)
        ]:
            for key in subconfig.as_dict().keys():
                key_map[key] = subconfig_name
                
        return key_map
    
    def load_config(self, config_path):
        """
        Carga la configuración desde un archivo JSON.
        Solo actualiza la configuración en memoria, no produce side-effects.
        
        Args:
            config_path: Ruta al archivo JSON de configuración
            
        Returns:
            Diccionario con la configuración cargada
        """
        try:
            with open(config_path) as f:
                loaded_config = json.load(f)
            
            # Distribuir valores a las subconfigs según las claves
            self._distribute_config(loaded_config)
            
            # Actualizar el mapa de claves
            self._key_map = self._build_key_map()
            
            logger.info(f"Configuración cargada de: {config_path}")
            return loaded_config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            raise
    
    def _distribute_config(self, config_dict):
        """
        Distribuye las claves de configuración a las subclases correspondientes
        
        Args:
            config_dict: Diccionario con la configuración a distribuir
        """
        # Diccionario para saber a qué subconfig corresponde cada clave
        key_mappings = {
            # ScientificConfig keys
            'n_medium': 'scientific',
            'angle_range': 'scientific',
            'lambda_blue': 'scientific',
            'lambda_violet': 'scientific',
            'n_particle_ref': 'scientific',
            'mie_max_nmax': 'scientific',
            'mie_n_points': 'scientific',
            'allow_complex_index': 'scientific',
            'core_shell_defaults': 'scientific',  # Todo el dict de parámetros de núcleo-corteza
            'mie_absolute_max_nmax': 'scientific',
            'mie_safety_factor': 'scientific',
            'mie_approx_threshold': 'scientific',
            'mie_large_particle_threshold': 'scientific',
            'mie_small_particle_factor': 'scientific',
            'mie_large_particle_factor': 'scientific',
            'mie_cache_precision': 'scientific',
            
            # SolverConfig keys
            'diameter_min': 'solver',
            'diameter_max': 'solver',
            'diameter_step': 'solver',
            'index_min': 'solver',
            'index_max': 'solver',
            'index_step': 'solver',
            'k_min': 'solver',
            'k_max': 'solver',
            'k_step': 'solver',
            'shell_thickness_min': 'solver',
            'shell_thickness_max': 'solver',
            'shell_thickness_step': 'solver',
            'solver_strategy': 'solver',
            'default_diameter_init': 'solver',
            'default_index_init': 'solver',
            'default_k_init': 'solver',
            'solver_tolerance': 'solver',
            'solver_cost_threshold': 'solver',
            'lookup_table_enabled': 'solver',
            'lookup_table_path': 'solver',
            'lookup_table_compression': 'solver',
            'min_size_for_parallel': 'solver',
            'parallel_enabled': 'solver',
            'parallel_jobs': 'solver',
            'sigma_n_prior': 'solver',
            'sigma_k_prior': 'solver',
            'd_solve_max_iterations': 'solver',
            'absolute_d_min': 'solver',
            'absolute_d_max': 'solver',
            
            # IOConfig keys
            'fcs_header_size': 'io',
            'output_dir': 'io',
            'output_patterns': 'io',
            'channel_mappings': 'io',
            'channel_name_patterns': 'io',
            'heuristic_intensity_threshold': 'io',
            'max_events_per_file': 'io',
            
            # VisualizationConfig keys
            'plot_margins': 'visualization',
            'plot_dpi': 'visualization',
            'plot_figsize': 'visualization',
            'plot_bins': 'visualization',
            'plot_scatter_cmap': 'visualization',
            'plot_hist2d_cmap': 'visualization',
            'plot_color_488': 'visualization',
            'plot_color_405': 'visualization',
            
            # CalibrationConfig keys
            'K488': 'calibration',
            'K405': 'calibration',
            'B488': 'calibration',
            'B405': 'calibration',
            'min_I488': 'calibration',
            'max_I488': 'calibration',
            'min_I405': 'calibration',
            'max_I405': 'calibration',
        }
        
        # Distribuir cada clave a la subconfig correspondiente
        for key, value in config_dict.items():
            # Casos especiales para núcleo-corteza: no distribuimos los subcampos
            # ya que queremos mantener toda la estructura intacta
            if key == 'core_shell_defaults':
                self.scientific.core_shell_defaults = value
                continue
                
            # Para otras claves, distribuir como normalmente
            subconfig_name = key_mappings.get(key)
            if subconfig_name:
                subcfg = getattr(self, subconfig_name)
                subcfg[key] = value
            else:
                logger.warning(f"Clave desconocida en configuración: {key}")
    
    def update(self, new_values):
        """
        Actualiza la configuración con nuevos valores solo en memoria,
        sin guardar a disco automáticamente.
        
        Args:
            new_values: Diccionario con nuevos valores a actualizar
            
        Returns:
            Self para encadenamiento
        """
        self._distribute_config(new_values)
        # Actualizar el mapa de claves después de cambios
        self._key_map = self._build_key_map()
        return self
    
    def save(self, output_path):
        """
        Guarda la configuración actual a disco.
        Esta función está separada de update para evitar side-effects no deseados.
        
        Args:
            output_path: Ruta donde guardar la configuración
            
        Returns:
            Ruta donde se guardó la configuración
        """
        # Recopilar configuración completa
        export_config = {}
        export_config.update(self.scientific.as_dict())
        export_config.update(self.solver.as_dict())
        export_config.update(self.io.as_dict())
        export_config.update(self.visualization.as_dict())
        export_config.update(self.calibration.as_dict())
        
        # Convertir valores a tipos nativos de Python para JSON
        for k, v in list(export_config.items()):
            if isinstance(v, np.ndarray):
                export_config[k] = v.tolist()
            elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                export_config[k] = v.item()
        
        # Guardar a disco
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(export_config, f, indent=4)
            
            logger.info(f"Configuración guardada en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error al guardar configuración: {e}")
            raise
    
    def update_and_save(self, new_values, output_path):
        """
        Método legacy para mantener compatibilidad.
        Actualiza la configuración y luego la guarda.
        Preferir usar update() y save() por separado.
        
        Args:
            new_values: Diccionario con nuevos valores a actualizar
            output_path: Ruta donde guardar la configuración actualizada
            
        Returns:
            Ruta donde se guardó la configuración
        """
        warnings.warn(
            "El método update_and_save() está obsoleto y será eliminado en versiones futuras. "
            "Use update() y save() por separado.", 
            DeprecationWarning, stacklevel=2
        )
        self.update(new_values)
        return self.save(output_path)
    
    def __getitem__(self, key):
        """
        Accede a los valores de configuración como si fuera un diccionario.
        Usa el mapa de claves para buscar eficientemente en la subconfig correcta.
        
        Args:
            key: Clave a buscar
            
        Returns:
            Valor de la configuración para esa clave
        """
        # Buscar en el mapa de claves, más eficiente que recorrer todas las subconfigs
        if key in self._key_map:
            subconfig_name = self._key_map[key]
            return getattr(self, subconfig_name)[key]
        
        # Si no está en el mapa, podría ser una clave nueva o no encontrada
        for subconfig in [self.scientific, self.solver, self.io, self.visualization, self.calibration]:
            try:
                return subconfig[key]
            except KeyError:
                continue
        
        # Si no se encuentra en ninguna subconfig
        raise KeyError(f"Clave de configuración no encontrada: {key}")
    
    def __setitem__(self, key, value):
        """
        Establece valores de configuración como si fuera un diccionario
        Usa _distribute_config para asignar a la subconfig correcta
        
        Args:
            key: Clave a modificar
            value: Nuevo valor
        """
        self._distribute_config({key: value})
        # Actualizar el mapa de claves si es una clave nueva
        if key not in self._key_map:
            self._key_map = self._build_key_map()
    
    def get(self, key, default=None):
        """
        Obtiene un valor de configuración con valor por defecto
        
        Args:
            key: Clave a buscar
            default: Valor por defecto si la clave no existe
            
        Returns:
            Valor de la configuración o default
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def as_dict(self):
        """
        Obtiene una copia completa de la configuración como diccionario
        
        Returns:
            Diccionario con todos los valores de configuración actuales
        """
        result = {}
        result.update(self.scientific.as_dict())
        result.update(self.solver.as_dict())
        result.update(self.io.as_dict())
        result.update(self.visualization.as_dict())
        result.update(self.calibration.as_dict())
        return result
    
    def calculate_config_hash(self, include_calibration=True):
        """
        Genera un hash SHA-256 de la configuración relevante para las tablas de lookup
        
        Args:
            include_calibration: Si se deben incluir constantes de calibración en el hash
            
        Returns:
            Hash hexadecimal SHA-256
        """
        # Siempre incluir los parámetros de solver
        hash_dict = {
            'solver': {
                'diameter_min': self.solver.get('diameter_min'),
                'diameter_max': self.solver.get('diameter_max'),
                'diameter_step': self.solver.get('diameter_step'),
                'index_min': self.solver.get('index_min'),
                'index_max': self.solver.get('index_max'),
                'index_step': self.solver.get('index_step'),
                # Removed: default_diameter_init, default_index_init, solver_tolerance, solver_cost_threshold
            }
        }
        
        # Incluir parámetros científicos relevantes
        hash_dict['scientific'] = {
            'n_medium': self.scientific.get('n_medium'),
            'angle_range': self.scientific.get('angle_range'),
            'lambda_blue': self.scientific.get('lambda_blue'),
            'lambda_violet': self.scientific.get('lambda_violet'),
            'mie_n_points': self.scientific.get('mie_n_points'),
            'mie_cache_precision': self.scientific.get('mie_cache_precision'),
            'mie_max_nmax': self.scientific.get('mie_max_nmax'),
            'mie_absolute_max_nmax': self.scientific.get('mie_absolute_max_nmax'),
            'mie_safety_factor': self.scientific.get('mie_safety_factor'),
            'mie_large_particle_threshold': self.scientific.get('mie_large_particle_threshold'),
            'mie_small_particle_factor': self.scientific.get('mie_small_particle_factor'),
            'mie_large_particle_factor': self.scientific.get('mie_large_particle_factor'),
        }
        
        # Incluir constantes de calibración si se solicita
        if include_calibration:
            hash_dict['calibration'] = self.scientific.calibration_constants
        
        # Convertir a string y calcular hash
        # Usar sort_keys para que el orden no afecte el hash
        hash_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    # Propiedades para acceso directo a parámetros comunes para compatibilidad
    
    @property
    def n_medium(self):
        return self.scientific.n_medium
    
    @property
    def angle_range(self):
        return self.scientific.angle_range
    
    @property
    def lambda_blue(self):
        return self.scientific.lambda_blue
    
    @property
    def lambda_violet(self):
        return self.scientific.lambda_violet
    
    @property
    def channel_mappings(self):
        return self.io.channel_mappings
    
    @property
    def mie_params(self):
        return self.scientific.mie_params
    
    @property
    def solver_params(self):
        return self.solver.solver_params
    
    @property
    def plot_params(self):
        return self.visualization.plot_params
    
    @property
    def output_patterns(self):
        return self.io.output_patterns
    
    # Métodos delegados a las subconfigs para compatibilidad y conveniencia
    
    def get_file_pattern(self, pattern_key, **kwargs):
        return self.io.get_file_pattern(pattern_key, **kwargs)
    
    def get_channel_mapping(self, filename):
        return self.io.get_channel_mapping(filename)
    
    def get_signal_bounds(self, signals, low_percentile=1, high_percentile=99, margin=1.2):
        return self.io.get_signal_bounds(signals, low_percentile, high_percentile, margin)


# Instancia global para fácil importación desde otros módulos
# Se reemplaza por una función get_config thread-safe
# CONFIG = Config()

def get_config(config_path=None, force_reload=False):
    """
    Obtiene la instancia global de Config, implementando un patrón singleton thread-safe.
    
    Args:
        config_path: Ruta opcional al archivo de configuración para cargar/recargar
        force_reload: Si es True, fuerza la recarga de la configuración aunque ya exista
        
    Returns:
        Instancia singleton de Config
    """
    global _CONFIG_INSTANCE
    
    # Si ya existe una instancia y no se solicita recarga, devolverla directamente
    if _CONFIG_INSTANCE is not None and not force_reload and config_path is None:
        return _CONFIG_INSTANCE
    
    # Usar lock para thread-safety
    with _CONFIG_LOCK:
        # Verificar de nuevo dentro del lock (para evitar condiciones de carrera)
        if _CONFIG_INSTANCE is None or force_reload or config_path is not None:
            if force_reload and _CONFIG_INSTANCE is not None:
                logger.info("Recargando configuración por solicitud explícita")
            
            # Crear nueva instancia con el path proporcionado o None
            _CONFIG_INSTANCE = Config(config_path)
            logger.debug("Nueva instancia de configuración creada")
        
        return _CONFIG_INSTANCE

# Función compatibilidad hacia atrás para configuración antigua
def initialize_config(config_path=None):
    """
    Inicializa o reinicializa la configuración global.
    Útil para scripts que necesitan cambiar la configuración.
    
    Args:
        config_path: Ruta al archivo de configuración para cargar
        
    Returns:
        Instancia de configuración
    """
    return get_config(config_path=config_path, force_reload=True)

# Ejemplo de uso:
if __name__ == "__main__":
    # Imprimir la configuración cargada para validación
    config = get_config()
    print(f"n_medium: {config.n_medium}")
    print(f"angle_range: {config.angle_range}")
    print(f"Solver params: {config.solver_params}")
    print(f"Plot params: {config.plot_params}")
    
    # Ejemplos de nuevas funcionalidades
    print(f"Nombre para CSV de resultados: {config.get_file_pattern('results_csv', sample_name='synechococcus')}")
    print(f"Hash de configuración: {config.calculate_config_hash()}")
    
    # Ejemplo de señales para probar signal_range_percentiles
    test_signals = np.array([10, 20, 30, 40, 50, 5000])  # Con outlier
    min_signal, max_signal = config.get_signal_bounds(test_signals)
    print(f"Rango de señales: [{min_signal}, {max_signal}]")
    
    # Demostración de separación update/save
    config.update({'n_medium': 1.34})
    # No se guarda hasta llamar explícitamente a save()
    # config.save('test_config.json')