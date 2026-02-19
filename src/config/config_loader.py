import yaml
from pathlib import Path

def confing_loader(config_path: str, overrides: dict) -> dict:
    """
    Carga la configuración del archivo YAML y aplica algun overrides entregada en un diccionario.

    Args:
        config_path (str): Path
    """

    path = Path(config_path or Path(__file__).parent / "config.yaml")

    # Lectura del archivo YAML
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Aplica los overrides acuerdo a la ubicación correspondiente en el diccionario
    for key, value in overrides.items():
        # setdefault() se utilzia para asegurar que la key exista en el diccionario, si no existe, se crea un valor predeterminado (un diccionario vacio) y luego se aplica un update() con el valor proporcionado en el override
        config.setdefault(key, {}).update(value)       
    
    return config