"""
Módulo para configuración centralizada de logging.
Este módulo debe ser importado en el punto de entrada de la aplicación (main.py)
antes de importar otros módulos.
"""

import logging
import os
import sys
from pathlib import Path

# Formato por defecto para los logs
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Nivel de log por defecto
DEFAULT_LEVEL = logging.INFO

# Configuración global ha sido aplicada
_LOGGING_CONFIGURED = False

def configure_logging(level=None, log_format=None, log_file=None, force_reconfigure=False):
    """
    Configura el sistema de logging para toda la aplicación.
    
    Args:
        level: Nivel de logging (logging.DEBUG, logging.INFO, etc.)
        log_format: Formato de los mensajes de log
        log_file: Si se proporciona, ruta donde guardar los logs
        force_reconfigure: Si es True, reconfigura aunque ya esté configurado
    """
    global _LOGGING_CONFIGURED
    
    # Verificar si ya está configurado
    if _LOGGING_CONFIGURED and not force_reconfigure:
        logging.warning("La configuración de logging ya ha sido aplicada. Se ignorará esta llamada.")
        return
    
    # Si ya está configurado pero queremos forzar la reconfiguración
    if _LOGGING_CONFIGURED and force_reconfigure:
        # En este caso solo actualizamos el nivel de los handlers existentes
        if level is not None:
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            for handler in root_logger.handlers:
                handler.setLevel(level)
            logging.debug(f"Nivel de logging actualizado a {logging.getLevelName(level)}")
        return
    
    # Usar valores por defecto si no se proporcionan
    level = level or DEFAULT_LEVEL
    log_format = log_format or DEFAULT_FORMAT
    
    # Configuración base
    handlers = []
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # Handler para archivo si se especifica
    if log_file:
        try:
            # Crear directorio si no existe
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file, 'a')
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Error al configurar log en archivo {log_file}: {e}")
    
    # Configuración del logger raíz
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # Forzar configuración incluso si ya existe
    )
    
    # Registrar módulo como configurado
    _LOGGING_CONFIGURED = True
    
    # Crear logger para este módulo
    logger = logging.getLogger(__name__)
    logger.debug("Sistema de logging configurado")

def get_logger(name):
    """
    Obtiene un logger con el nombre especificado.
    Si el sistema de logging aún no ha sido configurado, lo configura con valores por defecto.
    
    Args:
        name: Nombre del logger (normalmente __name__)
        
    Returns:
        Logger configurado
    """
    if not _LOGGING_CONFIGURED:
        configure_logging()
        
    return logging.getLogger(name)

# Configurar niveles para librerías externas ruidosas
def quiet_noisy_loggers():
    """Reduce el nivel de log para algunas librerías externas ruidosas"""
    for logger_name in ['matplotlib', 'PIL', 'joblib', 'numba']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)