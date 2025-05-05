import numpy as np
import struct
import re
import os
import logging
from pathlib import Path
from config import get_config
from logging_config import get_logger

# Crear logger usando el sistema centralizado
logger = get_logger('cytoflex_reader')

# Pre-compilar expresiones regulares para mejor rendimiento
RE_PAR_COUNT = re.compile(r'\$PAR\|(\d+)\|')
RE_TOT_COUNT = re.compile(r'\$TOT\|(\d+)\|')
RE_CHANNEL_NAME = re.compile(r'\$P(\d+)N\|([^|]+)\|')
RE_CHANNEL_INDEX = re.compile(r'\$P(\d+)I\|(\d+)\|')
RE_DATATYPE = re.compile(r'\$DATATYPE\|([^|]+)\|')
RE_BYTEORD = re.compile(r'\$BYTEORD\|([^|]+)\|')

# Constantes del formato FCS
FCS_SIGNATURE = "FCS"

try:
    from fcsparser import parse
except ImportError:
    logger.warning("Instalando fcsparser...")
    import subprocess
    subprocess.check_call(["pip", "install", "fcsparser"])
    from fcsparser import parse

def extract_fcs_header(f):
    """
    Extrae y valida el encabezado del archivo FCS
    
    Args:
        f: Objeto de archivo abierto en modo binario
        
    Returns:
        Diccionario con las posiciones de los segmentos TEXT y DATA
    """
    # Obtener la configuración dentro de la función
    config = get_config()
    
    # Obtener el tamaño del header de la configuración
    FCS_HEADER_SIZE = config.io.fcs_header_size
    
    f.seek(0)
    header = f.read(FCS_HEADER_SIZE).decode('ascii')
    
    # Verificar que es un archivo FCS
    if not header.startswith(FCS_SIGNATURE):
        raise ValueError(f"El archivo no es un archivo FCS válido (falta firma '{FCS_SIGNATURE}')")
    
    # Extraer posiciones de los segmentos
    text_start = int(header[10:18].strip())
    text_end = int(header[18:26].strip())
    data_start = int(header[26:34].strip())
    data_end = int(header[34:42].strip())
    
    logger.info(f"Segmentos FCS: TEXT[{text_start}-{text_end}], DATA[{data_start}-{data_end}]")
    
    return {
        'text_start': text_start,
        'text_end': text_end,
        'data_start': data_start,
        'data_end': data_end
    }

def extract_text_segment(f, header_info):
    """
    Extrae y parsea el segmento TEXT del archivo FCS
    
    Args:
        f: Objeto de archivo abierto en modo binario
        header_info: Diccionario con información del header
        
    Returns:
        Tupla con (texto completo, diccionario de metadatos)
    """
    # Posicionarse en el inicio del segmento TEXT
    f.seek(header_info['text_start'])
    text_len = header_info['text_end'] - header_info['text_start'] + 1
    text_segment = f.read(text_len).decode('ascii', errors='replace')
    
    # Imprimir los primeros 200 caracteres del segmento TEXT para depuración
    logger.info(f"Primeros 200 caracteres del segmento TEXT: {text_segment[:200]}")
    
    # Escribir todo el segmento TEXT a un archivo para análisis posterior si estamos en modo debug
    if logger.level <= logging.DEBUG:
        debug_path = Path(f.name).parent / 'debug_text_segment.txt'
        with open(debug_path, 'w') as debug_file:
            debug_file.write(text_segment)
        logger.debug(f"Segmento TEXT completo escrito a {debug_path}")
    
    # Extraer parámetros del segmento TEXT
    metadata = {}
    delimiter = text_segment[0]
    logger.info(f"Delimitador detectado: '{delimiter}'")
    
    # Extraer metadatos clave usando regex precompilados
    for key_pattern, regex in [
        ('$PAR', RE_PAR_COUNT),
        ('$TOT', RE_TOT_COUNT),
        ('$P\\dN', RE_CHANNEL_NAME),
        ('$P\\dI', RE_CHANNEL_INDEX),
        ('$DATATYPE', RE_DATATYPE),
        ('$BYTEORD', RE_BYTEORD)
    ]:
        for match in regex.finditer(text_segment):
            if key_pattern == '$PAR':
                metadata['$PAR'] = int(match.group(1))
            elif key_pattern == '$TOT':
                metadata['$TOT'] = int(match.group(1))
            elif key_pattern == '$P\\dN':
                channel_num = match.group(1)
                channel_name = match.group(2)
                metadata[f'$P{channel_num}N'] = channel_name
                if 'SSC' in channel_name:
                    logger.info(f"Canal SSC encontrado: P{channel_num}={channel_name}")
            elif key_pattern == '$P\\dI':
                channel_num = match.group(1)
                index = match.group(2)
                metadata[f'$P{channel_num}I'] = index
            elif key_pattern == '$DATATYPE':
                metadata['$DATATYPE'] = match.group(1)
            elif key_pattern == '$BYTEORD':
                metadata['$BYTEORD'] = match.group(1)
    
    return text_segment, metadata

def determine_parameter_count(text_segment, metadata):
    """
    Determina el número de parámetros (canales) de forma robusta
    
    Args:
        text_segment: Texto completo del segmento TEXT
        metadata: Diccionario de metadatos ya extraídos
        
    Returns:
        Número de parámetros (int)
    """
    # Verificar si ya tenemos $PAR en los metadatos
    if '$PAR' in metadata:
        par_count = metadata['$PAR']
        logger.info(f"$PAR encontrado en metadatos: {par_count}")
        return par_count
    
    # Buscar a través de expresiones regulares (ya debería estar en metadata, pero por si acaso)
    match = RE_PAR_COUNT.search(text_segment)
    if match:
        par_count = int(match.group(1))
        logger.info(f"$PAR encontrado directamente: {par_count}")
        return par_count
    
    # Método alternativo: buscar en líneas que contengan $PAR
    for line in text_segment.split("\n"):
        if "$PAR" in line:
            numbers = re.findall(r'\d+', line)
            if numbers:
                par_count = int(numbers[0])
                logger.info(f"$PAR extraído de la línea: {par_count}")
                return par_count
    
    # Si aún no lo encontramos, estimar por el número máximo de canales definidos
    p_numbers = []
    for i in range(1, 50):  # Buscar hasta 50 canales potenciales
        if f"$P{i}N" in text_segment:
            p_numbers.append(i)
    
    if p_numbers:
        par_count = max(p_numbers)
        logger.info(f"$PAR estimado a partir de canales encontrados: {par_count}")
        return par_count
    
    # Si todo lo anterior falla, no podemos continuar
    raise ValueError("No se pudo determinar el número de parámetros ($PAR)")

def determine_event_count(text_segment, metadata, header_info, par_count, data_type='F'):
    """
    Determina el número de eventos (filas) de forma robusta
    
    Args:
        text_segment: Texto completo del segmento TEXT
        metadata: Diccionario de metadatos ya extraídos
        header_info: Información del header FCS
        par_count: Número de parámetros ya determinado
        data_type: Tipo de datos ('F' para float, 'I' para int, etc.)
        
    Returns:
        Número de eventos (int)
    """
    # Verificar si ya tenemos $TOT en los metadatos
    if '$TOT' in metadata:
        event_count = metadata['$TOT']
        logger.info(f"$TOT encontrado en metadatos: {event_count}")
        return event_count
    
    # Buscar a través de expresiones regulares (ya debería estar en metadata, pero por si acaso)
    match = RE_TOT_COUNT.search(text_segment)
    if match:
        event_count = int(match.group(1))
        logger.info(f"$TOT encontrado directamente: {event_count}")
        return event_count
    
    # Si no lo encontramos, estimar basado en el tamaño del segmento DATA
    data_size = (header_info['data_end'] - header_info['data_start'] + 1)
    
    # Determinar el tamaño de cada tipo de dato
    if data_type == 'I':
        item_size = 4 * par_count  # 4 bytes por entero
    elif data_type == 'F':
        item_size = 4 * par_count  # 4 bytes por float
    elif data_type == 'D':
        item_size = 8 * par_count  # 8 bytes por double
    else:
        item_size = 4 * par_count  # Default: 4 bytes por tipo
    
    event_count = data_size // item_size
    logger.info(f"$TOT estimado del tamaño del segmento DATA: {event_count}")
    return event_count

def determine_data_format(metadata, text_segment):
    """
    Determina el formato de los datos (tipo y endianness)
    
    Args:
        metadata: Diccionario de metadatos extraídos
        text_segment: Texto completo del segmento TEXT
        
    Returns:
        Diccionario con claves 'dtype', 'byte_order', 'data_size'
    """
    # Determinar tipo de datos
    data_type = metadata.get('$DATATYPE', 'F')  # Por defecto, float
    
    # Determinar endianness (orden de bytes)
    endian = metadata.get('$BYTEORD', '4,3,2,1')  # Por defecto, big-endian
    
    # Mapear al formato para struct.unpack
    if data_type == 'I':
        # Integer data
        data_size = 4
        dtype = 'i'
    elif data_type == 'F':
        # Float data
        data_size = 4
        dtype = 'f'
    elif data_type == 'D':
        # Double data
        data_size = 8
        dtype = 'd'
    else:
        raise ValueError(f"Tipo de datos FCS desconocido: {data_type}")
    
    # Determinar endianness
    if endian.startswith('1,2'):
        byte_order = '<'  # little-endian
    else:
        byte_order = '>'  # big-endian
    
    logger.info(f"Tipo de datos: {data_type}, Endianness: {endian}, Formato: {byte_order}{dtype}")
    
    return {
        'dtype': dtype,
        'byte_order': byte_order,
        'data_size': data_size
    }

def find_ssc_channel_indices(metadata, filename, par_count):
    """
    Identifica los índices de los canales SSC para azul y violeta
    
    Args:
        metadata: Diccionario de metadatos extraídos
        filename: Nombre del archivo para usar configuración específica
        par_count: Número total de parámetros
        
    Returns:
        Tupla con (blue_ssc_idx, violet_ssc_idx)
    """
    config = get_config()
    
    # Obtener la configuración predeterminada para este archivo
    channel_config = config.get_channel_mapping(filename)
    blue_ssc_idx = channel_config.get('blue_ssc_index', 1)  # Default: canal 2 (índice 1)
    violet_ssc_idx = channel_config.get('violet_ssc_index', 2)  # Default: canal 3 (índice 2)
    
    # Buscar nombres de canales específicos SSC en los metadatos (prioridad más alta)
    for key, value in metadata.items():
        if key.startswith('$P') and key.endswith('N'):
            channel_num = int(key[2:-1])
            value_lower = value.lower()
            
            if 'ssc' in value_lower:
                # Buscar específicamente canales de área (SSC-A) en lugar de altura
                if value_lower == 'ssc-a':
                    # Verificar si hay un índice explícito
                    index_key = f'$P{channel_num}I'
                    if index_key in metadata:
                        blue_ssc_idx = int(metadata[index_key]) - 1
                    else:
                        blue_ssc_idx = channel_num - 1
                        
                elif value_lower in ('ssc_1-a', 'ssc-1-a'):
                    # Verificar si hay un índice explícito
                    index_key = f'$P{channel_num}I'
                    if index_key in metadata:
                        violet_ssc_idx = int(metadata[index_key]) - 1
                    else:
                        violet_ssc_idx = channel_num - 1
                # Buscar canales por tecnología si no se encuentra por nombre específico
                elif any(term in value_lower for term in ['blue', '488']) and '-a' in value_lower:
                    index_key = f'$P{channel_num}I'
                    if index_key in metadata:
                        blue_ssc_idx = int(metadata[index_key]) - 1
                    else:
                        blue_ssc_idx = channel_num - 1
                        
                elif any(term in value_lower for term in ['violet', '405']) and '-a' in value_lower:
                    index_key = f'$P{channel_num}I'
                    if index_key in metadata:
                        violet_ssc_idx = int(metadata[index_key]) - 1
                    else:
                        violet_ssc_idx = channel_num - 1
    
    # Verificar que los índices están dentro del rango válido
    if blue_ssc_idx >= par_count:
        logger.warning(f"Índice de Blue SSC ({blue_ssc_idx}) fuera de rango, usando valor predeterminado (1)")
        blue_ssc_idx = 1
        
    if violet_ssc_idx >= par_count:
        logger.warning(f"Índice de Violet SSC ({violet_ssc_idx}) fuera de rango, usando valor predeterminado (2)")
        violet_ssc_idx = 2
    
    logger.info(f"Índices de canales SSC: Blue SSC = {blue_ssc_idx}, Violet SSC = {violet_ssc_idx}")
    return blue_ssc_idx, violet_ssc_idx

def detect_ssc_channels(metadata, filename):
    """
    Detecta los índices de los canales SSC azul y violeta
    utilizando patrones configurables desde el archivo de configuración.
    
    Args:
        metadata: Diccionario con metadatos extraídos del archivo FCS
        filename: Nombre del archivo para utilizar en la configuración específica
        
    Returns:
        Tupla con (blue_ssc_idx, violet_ssc_idx)
    """
    # Importar dentro de la función para asegurar recarga dinámica
    from config import get_config
    config = get_config()
    io_config = config.io
    
    logger.debug(f"Detectando canales SSC para: {filename}")
    
    # 1) Extraer todos los nombres de canal del metadata:
    # Formato: {0: 'FSC-A', 1: 'SSC-H', 2: 'FL1-A', ...}
    channel_names = {}
    
    for key, value in metadata.items():
        if key.startswith('$P') and key.endswith('N'):
            # Extraer número de canal: $P1N -> 1
            channel_num = int(key[2:-1])
            
            # Determinar el índice real (0-based) para este canal
            # Primero ver si hay un índice explícito definido
            index_key = f'$P{channel_num}I'
            if index_key in metadata:
                channel_idx = int(metadata[index_key]) - 1
            else:
                # Por defecto, usar número de canal - 1
                channel_idx = channel_num - 1
                
            # Guardar en el mapeo de canales
            channel_names[channel_idx] = value
            logger.debug(f"Canal {channel_idx} ({key}): {value}")
    
    # 2) Obtener patrones de nombres de canal de la configuración
    patterns = {}
    try:
        config_patterns = io_config.channel_name_patterns
        # Garantizar que existan las claves necesarias
        if (hasattr(config_patterns, 'get') and 
            isinstance(config_patterns.get('blue_ssc', None), list) and 
            isinstance(config_patterns.get('violet_ssc', None), list)):
            patterns = {
                'blue_ssc': config_patterns.get('blue_ssc', []),
                'violet_ssc': config_patterns.get('violet_ssc', [])
            }
        else:
            logger.warning("Patrones de canal en configuración mal formateados")
    except (AttributeError, KeyError) as e:
        logger.warning(f"Error accediendo a los patrones de canal: {e}")
    
    # Si no hay patrones o están vacíos, usar valores por defecto
    if not patterns or not patterns.get('blue_ssc') or not patterns.get('violet_ssc'):
        logger.info("Usando patrones de canal por defecto")
        patterns = {
            'blue_ssc': ["SSC", "SSC-H", "SSC-A", "Blue_SSC", "488_SSC"],
            'violet_ssc': ["SSC_1", "SSC-V", "SSC_V", "Violet_SSC", "405_SSC", "V-SSC"]
        }
    
    # 3) Buscar coincidencias por patrones configurados
    found = {}
    for color_type, color_patterns in patterns.items():
        logger.debug(f"Buscando canal {color_type} con patrones: {color_patterns}")
        
        # Lista para almacenar coincidencias (índice, nombre, puntuación)
        matches = []
        
        # Precalcular longitudes de patrones para mayor eficiencia
        pattern_lengths = {pattern.lower(): len(pattern) for pattern in color_patterns}
        
        for idx, name in channel_names.items():
            name_lower = name.lower()
            name_len = len(name_lower)
            
            # Buscar coincidencias exactas primero (máxima prioridad)
            for pattern in color_patterns:
                pattern_lower = pattern.lower()
                pattern_len = pattern_lengths[pattern_lower]
                
                # Coincidencia exacta
                if name_lower == pattern_lower:
                    matches.append((idx, name, 100))  # Puntuación máxima
                    break
                    
                # Coincidencia como subcadena
                elif pattern_lower in name_lower:
                    # Mayor puntuación si el patrón es una parte sustancial del nombre
                    score = int((pattern_len / name_len) * 90)
                    matches.append((idx, name, score))
                    
                # Coincidencia parcial de palabras
                elif any(token in name_lower for token in pattern_lower.split()):
                    # Menor puntuación para coincidencias parciales
                    score = 50
                    matches.append((idx, name, score))
        
        # Ordenar coincidencias por puntuación (descendente)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Usar la mejor coincidencia si existe
        if matches:
            best_idx, best_name, score = matches[0]
            found[color_type] = best_idx
            logger.debug(f"Coincidencia para {color_type}: canal {best_idx} ({best_name}) con puntuación {score}")
    
    # 4) Si falta alguno, usar el mapeo por defecto según nombre de archivo
    # Obtener configuración para este archivo sin usar variables globales
    channel_config = io_config.get_channel_mapping(filename)
    
    blue_idx = found.get('blue_ssc', channel_config.get('blue_ssc_index', 1))
    violet_idx = found.get('violet_ssc', channel_config.get('violet_ssc_index', 2))
    
    logger.info(f"Canales SSC detectados → Azul: {blue_idx}, Violeta: {violet_idx}")
    return blue_idx, violet_idx

def binary_read_fcs_file(path):
    """
    Función de respaldo para leer archivos FCS binarios
    cuando fcsparser falla. Implementación más robusta y modular.
    
    Args:
        path: Ruta al archivo FCS
        
    Returns:
        Tupla con (I488, I405, metadata)
    """
    logger.info(f"Usando lectura binaria para archivo: {path}")
    filename = os.path.basename(path)
    
    try:
        with open(path, 'rb') as f:
            # Extraer y validar el encabezado FCS
            header_info = extract_fcs_header(f)
            
            # Extraer y parsear el segmento TEXT
            text_segment, metadata = extract_text_segment(f, header_info)
            
            # Determinar parámetros fundamentales
            par_count = determine_parameter_count(text_segment, metadata)
            data_format = determine_data_format(metadata, text_segment)
            event_count = determine_event_count(
                text_segment, metadata, header_info, 
                par_count, data_format['dtype']
            )
            
            # Verificar parámetros válidos
            if par_count <= 0:
                raise ValueError(f"Número de parámetros inválido: {par_count}")
            
            item_size = data_format['data_size'] * par_count
            if item_size <= 0:
                raise ValueError(f"Tamaño de ítem inválido: {item_size}")
            
            # Leer segmento DATA
            f.seek(header_info['data_start'])
            data_len = header_info['data_end'] - header_info['data_start'] + 1
            raw_data = f.read(data_len)
            
            # Verificar tamaño de datos esperado
            expected_size = item_size * event_count
            if len(raw_data) != expected_size:
                logger.warning(f"Tamaño de datos incorrecto: esperado {expected_size}, recibido {len(raw_data)}")
                # Recalcular el número real de eventos
                event_count = len(raw_data) // item_size
                logger.info(f"Número estimado de eventos: {event_count}")
            
            # Parsear datos binarios
            format_string = data_format['byte_order'] + data_format['dtype'] * par_count
            
            # Decodificar los datos
            data = []
            for i in range(0, len(raw_data), item_size):
                if i + item_size <= len(raw_data):
                    values = struct.unpack(format_string, raw_data[i:i+item_size])
                    data.append(values)
            
            if not data:
                raise ValueError("No se pudieron extraer datos del archivo FCS")
            
            # Identificar canales SSC usando el detector automático
            blue_ssc_idx, violet_ssc_idx = detect_ssc_channels(metadata, filename)
            
            # Usar la función helper unificada para validar y extraer señales
            _, _, I488, I405 = validate_ssc_indices(
                blue_ssc_idx, violet_ssc_idx, par_count, data, 
                channel_names=None, filename=filename
            )
            
            logger.info(f"Extracción binaria completada: {len(I488)} eventos")
            
            return I488, I405, metadata
            
    except Exception as e:
        logger.error(f"Error en lectura binaria: {e}")
        raise

def read_ssc_signals(path):
    """
    Lee las señales SSC de archivos FCS.
    Intenta primero con fcsparser y si falla usa lectura binaria
    
    Args:
        path: Ruta al archivo FCS
        
    Returns:
        Tupla con (I488, I405) - Señales SSC para 488nm y 405nm
    """
    config = get_config()
    filename = os.path.basename(path)
    
    try:
        logger.info(f"Leyendo archivo FCS: {path}")
        try:
            # Intentar con fcsparser, capturando específicamente el error de NumPy 2.0
            meta, df = parse(path, reformat_meta=True)
            
            # Extraer información para el detector de canales
            metadata = {}
            for key, value in meta.items():
                if key.startswith('$P') and key.endswith('N'):
                    metadata[key] = value
                elif key.startswith('$P') and key.endswith('I'):
                    metadata[key] = value
            
            # Usar la función de detección automática de canales
            blue_ssc_idx, violet_ssc_idx = detect_ssc_channels(metadata, filename)
            
            # Definir número total de parámetros
            par_count = len(df.columns)
            
            # Usar la función helper unificada para validar y extraer señales
            _, _, I488, I405 = validate_ssc_indices(
                blue_ssc_idx, violet_ssc_idx, par_count, df, 
                channel_names=None, filename=filename
            )
            
            # Imprimir primeras intensidades para validación
            logger.info(f"Primeras intensidades I488 (fcsparser): {I488[:5]}")
            logger.info(f"Primeras intensidades I405 (fcsparser): {I405[:5]}")
            
            return np.array(I488), np.array(I405)
            
        except AttributeError as e:
            if 'newbyteorder' in str(e):
                logger.warning("Detectado NumPy 2.0+, fcsparser necesita actualización. Usando método binario.")
                raise ValueError("Incompatibilidad con NumPy 2.0")
            else:
                raise
    
    except Exception as e:
        logger.warning(f"Error al parsear con fcsparser: {e}")
        try:
            I488, I405, _ = binary_read_fcs_file(path)
            return I488, I405
        except Exception as binary_error:
            logger.error(f"Error al leer archivo binario: {binary_error}")
            raise ValueError(f"No se pudo leer el archivo {path}: {binary_error}")

def validate_ssc_indices(blue_ssc_idx, violet_ssc_idx, par_count, data, channel_names=None, filename=None):
    """
    Función helper unificada para validar y corregir índices de canales SSC.
    Implementa estrategias de fallback cuando los índices detectados son inválidos.
    
    Args:
        blue_ssc_idx: Índice del canal SSC azul
        violet_ssc_idx: Índice del canal SSC violeta
        par_count: Número total de parámetros/canales disponibles
        data: Datos para usar en estrategias heurísticas (lista de filas o DataFrame)
        channel_names: Diccionario opcional con nombres de canales por índice
        filename: Nombre del archivo para usar en configuración específica
        
    Returns:
        Tupla con (blue_ssc_idx, violet_ssc_idx, I488, I405)
    """
    # Obtener configuración centralizada
    config = get_config()
    I488 = None
    I405 = None
    
    # Función helper para extraer las señales con los índices actuales
    def extract_signals(blue_idx, violet_idx):
        """Helper para extraer señales de los datos usando los índices especificados"""
        local_I488, local_I405 = None, None
        
        if isinstance(data, list):  # Caso de binary_read_fcs_file
            local_I488 = np.array([row[blue_idx] for row in data])
            local_I405 = np.array([row[violet_idx] for row in data])
        else:  # Caso de DataFrame de fcsparser
            try:
                column_mapping = {}
                for idx, col_name in enumerate(data.columns):
                    column_mapping[idx] = col_name
                    
                blue_ssc_col = column_mapping.get(blue_idx)
                violet_ssc_col = column_mapping.get(violet_idx)
                
                if blue_ssc_col is not None:
                    local_I488 = data[blue_ssc_col].values
                
                if violet_ssc_col is not None:
                    local_I405 = data[violet_ssc_col].values
            except Exception as e:
                logger.error(f"Error al extraer valores de DataFrame: {e}")
        
        return local_I488, local_I405
    
    # Si tenemos un nombre de archivo, obtener la configuración específica
    sample_name = None
    if filename:
        # Extraer nombre base sin extensión para buscar en mappings
        sample_name = os.path.splitext(os.path.basename(filename))[0].lower()
        
        # Verificar si el nombre coincide con alguna clave en channel_mappings
        # Ordenar claves por longitud descendente para priorizar las más específicas
        channel_mappings = config.io.channel_mappings if hasattr(config.io, 'channel_mappings') else {}
        sorted_keys = sorted(channel_mappings.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key != 'default' and key in sample_name:
                sample_name = key
                logger.info(f"Tipo de muestra detectado: '{key}' para archivo: {filename}")
                break
    
    # Obtener configuración específica para el archivo o usar default
    channel_config = None
    if sample_name and hasattr(config.io, 'channel_mappings'):
        channel_config = config.io.channel_mappings.get(sample_name, 
                                                      config.io.channel_mappings.get('default', {}))
    else:
        # Fallback por si no encuentra mappings
        channel_config = {'blue_ssc_index': 1, 'violet_ssc_index': 2}
    
    default_blue_idx = min(channel_config.get('blue_ssc_index', 1), par_count - 1)
    default_violet_idx = min(channel_config.get('violet_ssc_index', 2), par_count - 1)
    
    # Obtener umbral para detección heurística desde configuración o usar valor por defecto
    heuristic_threshold = config.io.get('heuristic_intensity_threshold', 100)
    
    # Siempre usar los índices de la configuración si están disponibles
    if channel_config:
        logger.info(f"Usando índices de configuración para '{sample_name or 'default'}': "
                   f"Blue={default_blue_idx}, Violet={default_violet_idx}")
        blue_ssc_idx = default_blue_idx
        violet_ssc_idx = default_violet_idx
    
    # Verificar que los índices están dentro del rango válido
    indices_valid = True
    if blue_ssc_idx >= par_count:
        logger.warning(f"Índice de Blue SSC ({blue_ssc_idx}) fuera de rango (máx {par_count-1})")
        indices_valid = False
    
    if violet_ssc_idx >= par_count:
        logger.warning(f"Índice de Violet SSC ({violet_ssc_idx}) fuera de rango (máx {par_count-1})")
        indices_valid = False
    
    # Si los índices están fuera de rango, ajustar al máximo disponible
    if not indices_valid:
        logger.info(f"Ajustando índices fuera de rango: Azul={min(blue_ssc_idx, par_count-1)}, Violeta={min(violet_ssc_idx, par_count-1)}")
        blue_ssc_idx = min(blue_ssc_idx, par_count - 1)
        violet_ssc_idx = min(violet_ssc_idx, par_count - 1)
    
    # Verificar que los índices son diferentes (este es el problema principal)
    if blue_ssc_idx == violet_ssc_idx:
        logger.warning(f"¡CRÍTICO! Los índices para Blue y Violet SSC son idénticos: {blue_ssc_idx}")
        logger.warning(f"Forzando uso de índices diferentes de la configuración")
        # Asignar valores diferentes de configuración
        blue_ssc_idx = default_blue_idx
        # Si por casualidad los valores de configuración son iguales, forzar diferencia
        if default_blue_idx == default_violet_idx:
            violet_ssc_idx = (default_violet_idx + 1) % par_count
        else:
            violet_ssc_idx = default_violet_idx
        logger.info(f"Índices corregidos: Blue={blue_ssc_idx}, Violet={violet_ssc_idx}")
    
    # Extraer las señales SSC con los índices actuales
    I488, I405 = extract_signals(blue_ssc_idx, violet_ssc_idx)
    
    # Validar que las señales extraídas tienen valores razonables
    signal_valid = True
    if I488 is None or np.mean(I488) < 1:
        logger.warning(f"Señal Blue SSC inválida o muy baja (media={np.mean(I488) if I488 is not None else 'None'})")
        signal_valid = False
    
    if I405 is None or np.mean(I405) < 1:
        logger.warning(f"Señal Violet SSC inválida o muy baja (media={np.mean(I405) if I405 is not None else 'None'})")
        signal_valid = False
    
    # Si las señales son inválidas, intentar identificación heurística
    if not signal_valid:
        logger.info("Intentando identificación heurística de canales SSC")
        if isinstance(data, list):  # Caso de binary_read_fcs_file
            # Calcular medias por canal
            channel_means = [np.mean([row[i] for row in data]) for i in range(par_count)]
            
            # Ordenar canales por intensidad media (descendente)
            sorted_channels = sorted(
                [(i, mean) for i, mean in enumerate(channel_means) if mean > heuristic_threshold],
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Si encontramos al menos 2 canales con valores significativos
            if len(sorted_channels) >= 2:
                if I488 is None or np.mean(I488) < 1:
                    blue_ssc_idx = sorted_channels[0][0]
                    logger.info(f"Cambiando Blue SSC al canal {blue_ssc_idx+1} con media {sorted_channels[0][1]:.1f}")
                
                if I405 is None or np.mean(I405) < 1:
                    # Usar el segundo canal más intenso o el primero si no se usó para Blue
                    violet_idx = 1 if blue_ssc_idx == sorted_channels[0][0] else 0
                    if violet_idx < len(sorted_channels):
                        violet_ssc_idx = sorted_channels[violet_idx][0]
                        logger.info(f"Cambiando Violet SSC al canal {violet_ssc_idx+1} con media {sorted_channels[violet_idx][1]:.1f}")
                
                # Re-extraer señales con los nuevos índices
                if blue_ssc_idx != default_blue_idx or violet_ssc_idx != default_violet_idx:
                    I488, I405 = extract_signals(blue_ssc_idx, violet_ssc_idx)
        
        else:  # Caso de DataFrame de fcsparser
            # Buscar columnas por contenido de valores
            means = {col: np.mean(data[col]) for col in data.columns}
            
            # Ordenar columnas por media (descendente)
            sorted_cols = sorted(
                [(col, mean) for col, mean in means.items() if mean > heuristic_threshold],
                key=lambda x: x[1],
                reverse=True
            )
            
            column_mapping = {idx: col_name for idx, col_name in enumerate(data.columns)}
            
            if len(sorted_cols) >= 2:
                # Para blue SSC, usar la columna con mayor media
                if I488 is None or np.mean(I488) < 1:
                    blue_ssc_col = sorted_cols[0][0]
                    # Encontrar el índice correspondiente
                    blue_ssc_idx = next((idx for idx, col in column_mapping.items() if col == blue_ssc_col), blue_ssc_idx)
                    logger.info(f"Cambiando Blue SSC a la columna {blue_ssc_col} con media {sorted_cols[0][1]:.1f}")
                
                # Para violet SSC, usar la segunda mayor media
                if I405 is None or np.mean(I405) < 1:
                    # Usar una columna diferente a la de Blue SSC
                    violet_col_idx = 1 if blue_ssc_col != sorted_cols[0][0] else 0
                    if violet_col_idx < len(sorted_cols):
                        violet_ssc_col = sorted_cols[violet_col_idx][0]
                        # Encontrar el índice correspondiente
                        violet_ssc_idx = next((idx for idx, col in column_mapping.items() if col == violet_ssc_col), violet_ssc_idx)
                        logger.info(f"Cambiando Violet SSC a la columna {violet_ssc_col} con media {sorted_cols[violet_col_idx][1]:.1f}")
                
                # Re-extraer señales con los nuevos índices si cambiaron
                if blue_ssc_idx != default_blue_idx or violet_ssc_idx != default_violet_idx:
                    I488, I405 = extract_signals(blue_ssc_idx, violet_ssc_idx)
    
    # Si todavía no tenemos señales válidas, usar índices por defecto
    if I488 is None:
        if isinstance(data, list) and 0 <= default_blue_idx < par_count:
            logger.warning(f"Usando canal {default_blue_idx+1} como fallback para Blue SSC")
            blue_ssc_idx = default_blue_idx
            I488, _ = extract_signals(blue_ssc_idx, violet_ssc_idx)
    
    if I405 is None:
        if isinstance(data, list) and 0 <= default_violet_idx < par_count:
            logger.warning(f"Usando canal {default_violet_idx+1} como fallback para Violet SSC")
            violet_ssc_idx = default_violet_idx
            _, I405 = extract_signals(blue_ssc_idx, violet_ssc_idx)
    
    # Validar y recortar los datos
    max_events = config.io.get('max_events_per_file', 5000)
    
    if I488 is not None and I405 is not None:
        # Recortar para asegurar mismo tamaño
        min_length = min(len(I488), len(I405))
        I488 = I488[:min_length]
        I405 = I405[:min_length]
        
        # Limitar número máximo de eventos
        if len(I488) > max_events:
            logger.info(f"Limitando eventos a {max_events} (de {len(I488)})")
            indices = np.random.choice(len(I488), max_events, replace=False)
            I488 = I488[indices]
            I405 = I405[indices]
    
    # Validar y devolver resultados
    if I488 is None or I405 is None:
        raise ValueError("No se pudieron obtener señales SSC válidas después de todos los intentos")

    # Registrar estadísticas finales
    logger.info(f"Índices finales: Blue SSC = {blue_ssc_idx}, Violet SSC = {violet_ssc_idx}")
    logger.info(f"Estadísticas Blue SSC: rango [{np.min(I488):.1f} - {np.max(I488):.1f}], media={np.mean(I488):.1f}")
    logger.info(f"Estadísticas Violet SSC: rango [{np.min(I405):.1f} - {np.max(I405):.1f}], media={np.mean(I405):.1f}")
    
    return blue_ssc_idx, violet_ssc_idx, I488, I405