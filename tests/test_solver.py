import numpy as np
import time

# Adjust imports for relative paths when running as part of a package
try:
    from ..config import get_config
    from ..solver import create_lookup_table
except ImportError:
    # Fallback for running the script directly
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import get_config
    from solver import create_lookup_table

# Function de utilidad para probar la regeneración o reutilización de tablas
def test_lookup_table_regeneration(constants, modified_config=None):
    """
    Prueba si la tabla de lookup se regenera correctamente cuando cambia la configuración
    
    Args:
        constants: Diccionario con constantes de calibración
        modified_config: Diccionario con cambios a aplicar a la configuración
        
    Returns:
        Tupla con (tabla original, tabla nueva, tiempo de carga original, tiempo de regeneración)
    """
    config = get_config()
    
    # Guardar configuración original
    original_config_dict = config.as_dict() # Use a different name to avoid conflict
    
    # Medir tiempo de creación/carga original
    start_time = time.time()
    # Ensure create_lookup_table uses the potentially modified config
    # If modified_config is None, it will use the global config state
    if modified_config:
        # Temporarily update config for this test run
        config.update(modified_config)
        table1 = create_lookup_table(constants, save_to_disk=True)
        # Restore original config after creating table1 with modified settings
        config.update(original_config_dict)
    else:
        table1 = create_lookup_table(constants, save_to_disk=True)
        
    original_load_time = time.time() - start_time
    
    # Aplicar modificaciones para la segunda tabla si es necesario
    if modified_config:
        config.update(modified_config) # Apply modification again for table2
        
    # Medir tiempo de carga/regeneración con posibles cambios
    start_time = time.time()
    table2 = create_lookup_table(constants, save_to_disk=True) # This will use the modified config if set
    modified_load_time = time.time() - start_time
    
    # Restaurar configuración original globalmente
    config.update(original_config_dict)
    
    return table1, table2, original_load_time, modified_load_time

if __name__ == "__main__":
    # Ejemplo de prueba de la tabla de lookup
    # Ensure config is loaded for the test execution context
    config = get_config() 
    
    # Cargar constantes mínimas para la prueba
    # Access config values directly, e.g., config.n_medium
    test_constants = {
        'K488': 2.8e18,
        'K405': 1.4e19,
        'n_medium': config.n_medium, # Use loaded config
        'angle_range': config.angle_range # Use loaded config
    }
    
    # Probar carga/regeneración sin cambios (debería ser rápido)
    print("\nPrueba 1 - Sin cambios:")
    # Pass a copy of current config to avoid modification issues if test_lookup_table_regeneration modifies it internally
    table1_orig, table2_orig, time1_orig, time2_orig = test_lookup_table_regeneration(test_constants)
    print(f"  Tiempo original: {time1_orig:.3f} s")
    print(f"  Tiempo recarga: {time2_orig:.3f} s")
    # Check if the tables' components are identical
    # table1_orig and table2_orig are tuples: (I488_table, I405_table, diameters, n_particles)
    if table1_orig and table2_orig:
        print(f"  ¿Tablas I488 idénticas? {np.array_equal(table1_orig[0], table2_orig[0])}")
        print(f"  ¿Tablas I405 idénticas? {np.array_equal(table1_orig[1], table2_orig[1])}")
        print(f"  ¿Diámetros idénticos? {np.array_equal(table1_orig[2], table2_orig[2])}")
        print(f"  ¿Índices idénticos? {np.array_equal(table1_orig[3], table2_orig[3])}")
    else:
        print("  Una o ambas tablas no se generaron correctamente.")

    # Probar con cambio en diámetro mínimo (debería regenerar)
    # Need to get the current diameter_min from the config to modify it
    # Assuming diameter_min is under solver_params in the config structure
    # This part needs careful handling of how config is structured and accessed
    # For now, let's assume a default or fetch it if possible.
    # This might require a more robust way to access nested config values.
    # For the purpose of this refactoring, we'll keep the logic similar to original.
    # The key is that `test_lookup_table_regeneration` now handles config changes internally.
    
    # Get current diameter_min from config to base the modification on
    # Fallback if solver_params or diameter_min is not in config
    current_solver_params = config.solver_params
    current_diameter_min = current_solver_params.get('diameter_range', [100, 5000, 10])[0]

    print(f"\nPrueba 2 - Cambio en diameter_min (original: {current_diameter_min}):")
    # Pass the modification directly
    _, table3_mod, _, time3_mod = test_lookup_table_regeneration(
        test_constants, 
        modified_config={'solver_params': {**current_solver_params, 'diameter_range': [current_diameter_min + 10, current_solver_params.get('diameter_range', [100,5000,10])[1], current_solver_params.get('diameter_range', [100,5000,10])[2]]}}
    )
    print(f"  Tiempo regeneración: {time3_mod:.3f} s")
    if table1_orig and table3_mod:
        print(f"  ¿Tablas I488 idénticas? {np.array_equal(table1_orig[0], table3_mod[0])}") # Expect False
    else:
        print("  Una o ambas tablas no se generaron correctamente para la comparación.")

    # Probar con cambio en constante de calibración (debería regenerar)
    print(f"\nPrueba 3 - Cambio en K488 (original: {test_constants['K488']}):")
    # The constants are passed directly, so create a modified version of test_constants
    modified_constants_k488 = {**test_constants, 'K488': test_constants['K488'] * 1.01}
    _, table4_mod_const, _, time4_mod_const = test_lookup_table_regeneration(modified_constants_k488)
    print(f"  Tiempo regeneración: {time4_mod_const:.3f} s")
    if table1_orig and table4_mod_const:
        print(f"  ¿Tablas I488 idénticas? {np.array_equal(table1_orig[0], table4_mod_const[0])}") # Expect False
    else:
        print("  Una o ambas tablas no se generaron correctamente para la comparación.")
