import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings
import sys
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.ticker as ticker
from calibration import sigma_sca_ssc, calculate_sigma_array
from config import get_config

# Importar tqdm con degradación elegante si no está disponible
try:
    from tqdm import tqdm
except ImportError:
    # Función que simula tqdm pero no hace nada
    def tqdm(iterable, **kwargs):
        return iterable

# Configuración del logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cytoflex_plots')

def plot_calibration(diameters, I488, I405, K488, K405, B488=0, B405=0, output_path=None, theory_curves=None, log_x=False):
    """
    Genera un gráfico simplificado de la calibración
    
    Args:
        diameters: Array con diámetros [nm]
        I488: Array con intensidades SSC en 488 nm
        I405: Array con intensidades SSC en 405 nm
        K488: Factor de calibración para 488 nm
        K405: Factor de calibración para 405 nm
        B488: Offset de fondo para 488 nm (por defecto 0)
        B405: Offset de fondo para 405 nm (por defecto 0)
        output_path: Ruta para guardar el gráfico (opcional)
        theory_curves: Diccionario con curvas teóricas precalculadas:
                      {'d_fine': array, 'I488_theo': array, 'I405_theo': array}
                      Si no se proporciona, generará una advertencia.
        log_x: Si usar escala logarítmica en el eje X (por defecto False)
    """
    config = get_config()
    
    # Comprobar que se proporcionan curvas teóricas
    if theory_curves is None:
        warnings.warn(
            "No se proporcionaron curvas teóricas precalculadas. "
            "Debe usar calculate_theory_curves() y pasar el resultado para evitar "
            "recálculos innecesarios del modelo Mie.",
            UserWarning, stacklevel=2
        )
        # Importamos aquí para evitar importación circular
        from main import calculate_theory_curves
        logger.warning("Generando curvas teóricas dentro de plot_calibration (ineficiente)")
        theory_curves = calculate_theory_curves(diameters, K488, K405, B488, B405)
    
    # Obtener parámetros de visualización desde la configuración centralizada
    plot_params = config.plot_params
    figsize = plot_params.get('figsize', (10, 7))
    dpi = plot_params.get('dpi', 150)
    margins = plot_params.get('margins', {})
    
    # Definir márgenes relativos para los ejes
    y_margin_bottom = margins.get('intensity', 0.4)  # Margen para intensidades
    y_margin_top = y_margin_bottom
    x_margin_left = margins.get('diameter', 0.3)     # Margen para diámetros
    x_margin_right = x_margin_left
    
    # Crear figura con un solo panel
    plt.figure(figsize=figsize)
    
    # Usar curvas teóricas precalculadas
    d_fine = theory_curves['d_fine']
    I488_theo = theory_curves['I488_theo']
    I405_theo = theory_curves['I405_theo']
    
    # Obtener colores desde configuración
    color_488 = plot_params.get('color_488', 'blue')
    color_405 = plot_params.get('color_405', 'purple')
    
    # Dibujar puntos de datos observados
    plt.scatter(diameters, I488, c=color_488, marker='o', s=60, 
              label='Datos 488 nm', alpha=0.8)
    plt.scatter(diameters, I405, c=color_405, marker='^', s=60, 
              label='Datos 405 nm', alpha=0.8)
    
    # Añadir líneas de modelo Mie - Usar color y linestyle como argumentos separados
    # en lugar de la notación string que causa problemas con nombres de colores largos
    plt.plot(d_fine, I488_theo, color=color_488, linestyle='-', alpha=0.7, lw=2, 
           label='Modelo 488 nm')
    plt.plot(d_fine, I405_theo, color=color_405, linestyle='-', alpha=0.7, lw=2, 
           label='Modelo 405 nm')
    
    # Ajustar ejes y límites
    plt.yscale('log')  # Escala logarítmica para eje Y siempre
    
    # Escala logarítmica para eje X si se solicita (mejor para ver comportamiento Rayleigh)
    if log_x:
        plt.xscale('log')
        
        # Personalizar etiquetas en escala log
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
    
    # Determinar límites de intensidad - Ajustados para mostrar el valor máximo con margen
    y_min = min(np.min(I488), np.min(I405), np.min(I488_theo), np.min(I405_theo)) * (1 - y_margin_bottom)
    y_max = max(np.max(I488), np.max(I405), np.max(I488_theo), np.max(I405_theo)) * (1 + y_margin_top)
    plt.ylim(y_min, y_max)
    
    # Usar el rango que cubra tanto datos observados como teóricos con margen adecuado
    x_min = min(np.min(diameters), np.min(d_fine)) * (1 - x_margin_left)
    x_max = max(np.max(diameters), np.max(d_fine)) * (1 + x_margin_right)
    plt.xlim(x_min, x_max)
    
    # Etiquetas y título
    plt.xlabel('Diámetro [nm]')
    plt.ylabel('Intensidad [canales]')
    plt.title('Calibración de intensidades SSC')
    plt.grid(True, alpha=0.3)
    
    # Crear una única leyenda combinada y posicionada correctamente
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)
    
    # Añadir anotación con valores de K y B
    if B488 != 0 or B405 != 0:
        annotation_text = (f'K₄₈₈ = {K488:.2e} ch/m²\nK₄₀₅ = {K405:.2e} ch/m²\n'
                          f'B₄₈₈ = {B488:.2e} ch\nB₄₀₅ = {B405:.2e} ch')
    else:
        annotation_text = f'K₄₈₈ = {K488:.2e} ch/m²\nK₄₀₅ = {K405:.2e} ch/m²'
    
    plt.annotate(annotation_text, 
                xy=(0.03, 0.97), xycoords='axes fraction',
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        logger.info(f"Gráfico de calibración guardado en: {output_path}")
    
    return plt.gcf()

def plot_scatter_results(diameters, n_particles, I488, converged, output_path=None, 
                       add_colorbar=True, title=None, bins=None):
    """
    Genera un gráfico de dispersión de diámetro vs. índice de refracción,
    con las partículas coloreadas según su intensidad de dispersión
    
    Args:
        diameters: Array de diámetros [nm]
        n_particles: Array de índices de refracción (pueden ser complejos)
        I488: Array de intensidades SSC en 488 nm
        converged: Array booleano que indica si la inversión convergió para cada partícula
        output_path: Ruta para guardar el gráfico (opcional)
        add_colorbar: Si se debe añadir barra de color con valores de intensidad
        title: Título del gráfico (opcional)
        bins: Número de bins para el histograma (opcional)
    """
    config = get_config()
    
    # Obtener parámetros de visualización desde la configuración centralizada
    plot_params = config.plot_params
    figsize = plot_params.get('figsize', (10, 8))
    dpi = plot_params.get('dpi', 150)
    margins = plot_params.get('margins', {})
    
    # Verificar que haya datos con conversión exitosa
    valid_idx = np.logical_and(converged, ~np.isnan(diameters))
    if not np.any(valid_idx):
        logger.warning("No hay datos válidos para generar el gráfico")
        return None
    
    # Filtrar datos válidos
    diameters_valid = diameters[valid_idx]
    n_particles_valid = n_particles[valid_idx]
    I488_valid = I488[valid_idx]
    
    # Si los índices son complejos, extraer la parte real para la visualización
    if np.iscomplexobj(n_particles_valid):
        n_plot = np.real(n_particles_valid)
        has_imag = True
        # Extraer también parte imaginaria para posible análisis
        k_plot = np.imag(n_particles_valid)
    else:
        n_plot = n_particles_valid
        has_imag = False
    
    # Márgenes para los límites de los ejes
    d_margin = margins.get('diameter', 0.3)
    n_margin = margins.get('index', 0.2)
    
    # Crear una figura con subplots en formato 2x2
    fig = plt.figure(figsize=figsize)
    
    # Definir la disposición de la figura: 3 paneles
    gs = plt.GridSpec(4, 4)
    
    # Scatter plot principal: índice vs diámetro (panel grande inferior izquierdo)
    ax_scatter = fig.add_subplot(gs[1:, :3])
    
    # Histograma de diámetros (panel superior)
    ax_hist_d = fig.add_subplot(gs[0, :3], sharex=ax_scatter)
    
    # Histograma de índices (panel derecho)
    ax_hist_n = fig.add_subplot(gs[1:, 3], sharey=ax_scatter)
    
    # Número de bins para los histogramas
    if bins is None:
        bins = plot_params.get('bins', 50)
    
    # Colormap para el scatter plot desde configuración centralizada
    scatter_cmap = plot_params.get('scatter_cmap', 'viridis')
    hist2d_cmap = plot_params.get('hist2d_cmap', 'inferno')
    
    # Plot principal: scatter plot de índice vs diámetro
    scatter = ax_scatter.scatter(diameters_valid, n_plot, 
                              c=I488_valid, s=10, alpha=0.7,
                              cmap=scatter_cmap, norm=LogNorm())
    
    # Calcular límites para los ejes
    d_min = np.min(diameters_valid) * (1 - d_margin)
    d_max = np.max(diameters_valid) * (1 + d_margin)
    n_min = np.min(n_plot) * (1 - n_margin)
    n_max = np.max(n_plot) * (1 + n_margin)
    
    # Establecer límites para diámetro con margen adicional
    ax_scatter.set_xlim(d_min, d_max)
    
    # Establecer límites para índice con margen adicional
    ax_scatter.set_ylim(n_min, n_max)
    
    # Etiquetas para el scatter plot
    ax_scatter.set_xlabel('Diámetro [nm]')
    ax_scatter.set_ylabel('Índice de refracción (parte real)' if has_imag else 'Índice de refracción')
    ax_scatter.grid(True, alpha=0.3)
    
    # Histograma de diámetros (arriba)
    ax_hist_d.hist(diameters_valid, bins=bins, color='skyblue', alpha=0.7)
    ax_hist_d.set_ylabel('Frecuencia')
    ax_hist_d.grid(True, alpha=0.3)
    ax_hist_d.set_axisbelow(True)
    
    # Histograma de índices (derecha)
    ax_hist_n.hist(n_plot, bins=bins, orientation='horizontal', 
                 color='skyblue', alpha=0.7)
    ax_hist_n.set_xlabel('Frecuencia')
    ax_hist_n.grid(True, alpha=0.3)
    ax_hist_n.set_axisbelow(True)
    
    # Ajustar los histogramas para no mostrar etiquetas duplicadas
    plt.setp(ax_hist_d.get_xticklabels(), visible=False)
    plt.setp(ax_hist_n.get_yticklabels(), visible=False)
    
    # Añadir colorbar si se especifica
    if add_colorbar:
        cb = plt.colorbar(scatter, ax=[ax_scatter, ax_hist_n], pad=0.01)
        cb.set_label('Intensidad SSC 488 nm [canales]')
    
    # Si los índices son complejos, añadir una anotación con estadísticas de k
    if has_imag:
        k_mean = np.mean(k_plot)
        k_std = np.std(k_plot)
        k_median = np.median(k_plot)
        annotation_text = (f"Parte imaginaria (k):\n"
                          f"Media: {k_mean:.4f}\n"
                          f"Mediana: {k_median:.4f}\n"
                          f"Desv. est.: {k_std:.4f}")
        
        # Añadir anotación en la esquina superior derecha
        ax_scatter.annotate(annotation_text, 
                          xy=(0.97, 0.97), xycoords='axes fraction',
                          ha='right', va='top', fontsize=9,
                          bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Añadir título si se proporciona
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Ajustes finales de diseño
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        logger.info(f"Gráfico de resultados guardado en: {output_path}")
    
    return fig

def plot_comparison(sample_names, diameters_list, n_particles_list, output_path=None, 
                   colors=None, figsize=None):
    """
    Genera un gráfico comparativo de histogramas para múltiples muestras
    
    Args:
        sample_names: Lista de nombres de muestras
        diameters_list: Lista de arrays de diámetros para cada muestra
        n_particles_list: Lista de arrays de índices para cada muestra
        output_path: Ruta para guardar el gráfico (opcional)
        colors: Lista de colores para cada muestra (opcional)
        figsize: Tamaño de la figura (opcional)
    """
    config = get_config()
    
    # Obtener parámetros desde configuración
    plot_params = config.plot_params
    if figsize is None:
        figsize = plot_params.get('figsize', (10, 8))
    dpi = plot_params.get('dpi', 150)
    
    # Verificar que hay datos
    if not sample_names or not diameters_list or not n_particles_list:
        logger.warning("No hay datos para generar la comparación")
        return None
    
    n_samples = len(sample_names)
    
    # Generar colores si no se proporcionan
    if colors is None:
        # Usar un colormap para generar colores distintos
        cmap = cm.get_cmap('tab10', n_samples)
        colors = [cmap(i) for i in range(n_samples)]
    
    # Crear figura con dos paneles (diámetro e índice)
    fig, (ax_d, ax_n) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogramas de diámetros
    bins = plot_params.get('bins', 50)
    alpha = 0.7
    
    # Calcular rango global para los histogramas
    all_diameters = np.concatenate([d[~np.isnan(d)] for d in diameters_list])
    all_indices = np.concatenate([n[~np.isnan(n)] for n in n_particles_list])
    
    # Definir bins globales basados en rango de datos
    d_bins = np.linspace(np.min(all_diameters), np.max(all_diameters), bins)
    n_bins = np.linspace(np.min(all_indices), np.max(all_indices), bins)
    
    # Dibujar histogramas para cada muestra
    for i, (name, diameters, n_particles, color) in enumerate(zip(sample_names, diameters_list, n_particles_list, colors)):
        # Filtrar valores NaN
        valid_d = ~np.isnan(diameters)
        valid_n = ~np.isnan(n_particles)
        
        # Histograma de diámetros
        ax_d.hist(diameters[valid_d], bins=d_bins, alpha=alpha, 
                color=color, label=name, density=True)
        
        # Histograma de índices
        ax_n.hist(n_particles[valid_n], bins=n_bins, alpha=alpha, 
                color=color, label=name, density=True)
    
    # Configuración de gráficos
    ax_d.set_xlabel('Diámetro [nm]')
    ax_d.set_ylabel('Densidad')
    ax_d.set_title('Distribución de diámetros')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)
    
    ax_n.set_xlabel('Índice de refracción')
    ax_n.set_title('Distribución de índices')
    ax_n.legend()
    ax_n.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        logger.info(f"Gráfico comparativo guardado en: {output_path}")
    
    return fig

def get_output_path(sample_name, plot_type='plot'):
    """
    Genera una ruta para guardar gráficos basada en nombre de muestra y tipo
    
    Args:
        sample_name: Nombre de la muestra (o None para calibración)
        plot_type: Tipo de gráfico ('plot', 'calib_plot')
        
    Returns:
        Ruta completa para el gráfico
    """
    config = get_config()
    
    # Directorio de salida
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Patrones para diferentes tipos de gráficos
    if plot_type == 'calib_plot':
        # No se usa el nombre de la muestra para calibración
        path = os.path.join(output_dir, config.get_file_pattern('calib_plot'))
    else:
        # Para gráficos de muestras específicas
        if sample_name:
            path = os.path.join(output_dir, 
                             config.get_file_pattern('plot', sample_name=sample_name))
        else:
            # Fallback para plot genérico
            path = os.path.join(output_dir, 'plot.png')
    
    return path

def calculate_theory_curves(diameters, K488, K405, B488=0, B405=0, extended_range=True, log_spacing=True, show_progress=True):
    """
    Genera curvas teóricas (I488 e I405) a partir de diámetros de calibración.
    Esta función debe usarse para precalcular curvas teóricas y pasarlas a plot_calibration
    para evitar recálculos innecesarios.
    
    Args:
        diameters: Array de diámetros utilizados en calibración [nm]
        K488: Factor de calibración para longitud de onda 488 nm
        K405: Factor de calibración para longitud de onda 405 nm
        B488: Offset de fondo para 488 nm (por defecto 0)
        B405: Offset de fondo para 405 nm (por defecto 0)
        extended_range: Si se debe extender el rango para visualización (por defecto True)
        log_spacing: Si se debe usar espaciado logarítmico para mejor visualización (por defecto True)
        show_progress: Si se debe mostrar barra de progreso (por defecto True)
        
    Returns:
        Diccionario con:
        - 'd_fine': array de diámetros finos para la curva
        - 'I488_theo': intensidades teóricas a 488 nm
        - 'I405_theo': intensidades teóricas a 405 nm
    """
    config = get_config()
    
    # Obtener parámetros de configuración
    n_part_ref = config.scientific.n_particle_ref
    n_med = config.n_medium
    angle_range = config.angle_range
    λ488 = config.lambda_blue
    λ405 = config.lambda_violet
    
    # Crear rango fino de diámetros con más puntos y rango extendido
    d_min, d_max = np.min(diameters), np.max(diameters)
    
    # Número de puntos configurable para la curva
    n_points = config.plot_params.get('curve_points', 500)  # Aumentado a 500 por defecto
    
    # Extender el rango si se solicita (mejor visualización en escala log-log)
    if extended_range:
        # Extender hacia valores más pequeños para ver comportamiento Rayleigh
        # Asegurar al menos un punto en 200 nm para anclaje
        d_min_ext = min(d_min * 0.7, 200)
        # Extender hacia valores más grandes para ver oscilaciones Mie
        d_max_ext = d_max * 1.5
        logger.info(f"Extendiendo rango de diámetros: {d_min} → {d_min_ext} nm, {d_max} → {d_max_ext} nm")
        d_min, d_max = d_min_ext, d_max_ext
    
    # Crear array de diámetros optimizado para visualización log-log
    if log_spacing:
        # Escala logarítmica para mejor resolución en valores pequeños
        d_fine = np.logspace(np.log10(d_min), np.log10(d_max), n_points)
    else:
        # Escala lineal tradicional
        d_fine = np.linspace(d_min, d_max, n_points)
    
    # Determinar si mostrar la barra de progreso (solo en TTY y si está habilitada)
    disable_progress = not show_progress or not sys.stderr.isatty()
    
    # Versión optimizada: calcula sigma para diámetros individuales con barra de progreso
    sigma488 = []
    sigma405 = []
    
    # Usar tqdm para mostrar progreso durante los cálculos intensivos
    for d in tqdm(d_fine, desc='σ_sca Mie', disable=disable_progress):
        r = d / 2e9  # nm → m
        sigma488.append(sigma_sca_ssc(r, n_part_ref, λ488, n_med, angle_range))
        sigma405.append(sigma_sca_ssc(r, n_part_ref, λ405, n_med, angle_range))
    
    # Convertir a arrays NumPy
    sigma488 = np.array(sigma488)
    sigma405 = np.array(sigma405)
    
    # Convertir a intensidades usando factores de calibración e incorporando offset
    I488_theo = K488 * sigma488 + B488
    I405_theo = K405 * sigma405 + B405
    
    logger.info(f"Curvas teóricas calculadas con {n_points} puntos en rango [{d_min:.1f}, {d_max:.1f}] nm")
    
    return {
        'd_fine': d_fine, 
        'I488_theo': I488_theo, 
        'I405_theo': I405_theo
    }