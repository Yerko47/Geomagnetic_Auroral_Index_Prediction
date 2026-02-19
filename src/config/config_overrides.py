import argparse

def config_overrides():
    """
    Función para modificar las variables de configuración en el diccionario del archivo YAML en la linea de comandos. Para esto, se debe anteponer en todas las variables que se quiera colocar "--set ", siendo el comando main.py --set key.value=<value>

    Casos:
        Normalmente se debe utilizar como key.value=<value> (ej: dataset.scaler_type=<name scaler>)
        En el caso de que se quiera modificar una matriz, se debe colocar como key.value=<v1>,<v2>,<v3>,... (ej: dataset.omni_variables=<V1>,<V2>,<V3>,...)
        En el caso de que se quiera modificar un diccionario dentro del diccionario, se debe colocar como key.value.subvalue=<value> (ej: hyparameter.model.type=<model type>)
    """

    parser = argparse.ArgumentParser(
        description = "Permite modificar las variables de configuración en el diccionario del archivo YAML en la línea de comandos"
    )

    # Anotación estandar para modificar la variable
    parser.add_argument("--set", action = "append", help = "Override de config en formato key.value = value")

    args = parser.parse_args()

    # Conversor a diccionario
    def build_nested_overrides(pairs):
        """
        Función para convertir overrides a diccionario
        """

        overrides = {}

        for item in pairs:
            key, value = item.split("=", 1)
            keys = key.split(".")

            current = overrides
            for k in keys[:-1]:
                current = current.setdefault(k, {})

            current[keys[-1]] = parse_value(value)

        return overrides

    # Evaluador de tipos de datos
    def parse_value(value):
        """
        Revisa los tipos de datos que se añaden como override.
        """

        if value.lower() == "none": return None
        if value.lower() in ["true", "false"]: return value.lower() == "true"

        try: return int(value)
        except ValueError: pass

        try: return float(value)
        except ValueError: pass

        if "," in value: return value.split(",")

        return value
    
    if args.set is None:
        return {}
    else:
        return build_nested_overrides(args.set)
