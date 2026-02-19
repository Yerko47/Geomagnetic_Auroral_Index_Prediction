from pathlib import Path

def path_file():
    """
    Crea la estructura de carpetas del proyecto y devuelve un diccionario
    """
    base = Path(Path(__file__).resolve().parent.parent.parent.parent)
    
    PROJECT_PATHS = {
        "raw_file": ("data", "raw"),
        "process_file": ("data", "process"),
        "model_result_file": ("results", "model"),
        "prediction_result_file": ("results", "prediction"),
        "metric_result_file": ("results", "metrics_result"),

    }

    paths = {key: base.joinpath(*parts) for key, parts in PROJECT_PATHS.items()}
    

    for path in paths.values():
        path.mkdir(parents = True, exist_ok = True)

    return paths
