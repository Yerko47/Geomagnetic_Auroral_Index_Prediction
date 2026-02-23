import cdflib
import pandas as pd
import numpy as np
from pathlib import Path


def dataset(config: dict, paths: dict) -> pd.DataFrame:
    """
    Función encargada de procesar los archivos CDF de OMNI y convertirlos a un DataFrame. Si el archivo .feather ya existe, se carga directamente desde ese archivo. Si no existe, se procesan los archivos CDF, se limpian y se guardan en un archvio .feather. La función devuelve un DataFrame con los tipos correspondiente de datos.

    Args:
        - config (dict): Diccionario de configuración del proyecto, que incluye información sobre las variables.
        - paths (dict): Diccionario con las rutas de los archivos y carpetas del proyecto.
    
    Returns:
        - df (pd.DataFrame): DataFrame con los datos de los archivos CDF procesados, limpios y con los tipos de datos correspondientes.
    """
    omni_path = Path("data/omni/hro_1min")
    
    start_time = pd.Timestamp(config["dataset"]["time_range"]["start"])
    end_time = pd.Timestamp(config["dataset"]["time_range"]["end"])

    save_feather_file = paths["raw_file"] / Path(f"data_{start_time.year}_to_{end_time.year}.feather")
    
    if not save_feather_file.exists():
        print(f"\n Procesando OMNI data desde la fecha {start_time.strftime('%Y-%m-%d')} a {end_time.strftime('%Y-%m-%d')}\n"
              f"{'=' * 61}\n")

        date_array = pd.date_range(start = start_time, end = end_time, freq = "MS")
        
        date_frame = []

        for date in date_array:
            cdf_file_name = f"omni_hro_1min_{date.strftime('%Y%m011')}_v01.cdf"
            cdf_file_path = omni_path / str(date.year) / cdf_file_name
            if cdf_file_path.exists():
                cdf_df = cdf_read(cdf_file_path)
                if not cdf_df.empty:
                    date_frame.append(cdf_df)
            else:
                print(f"Este archivo CDF no existe: {cdf_file_path}")

        df = pd.concat(date_frame, axis = 0, ignore_index = True)
        df = df[[col for col in ["Epoch"] + config["dataset"]["omni_variables"] + config["dataset"]["auroral_variables"]]]

        df = bad_data(df)

        df.reset_index(drop = True, inplace = True)
        df.to_feather(save_feather_file)
        print(f" Archivo feather ha sido procesado con shape de {df.shape} y guardado en {save_feather_file}")

        return df
    
    elif save_feather_file.exists():
        try:
            print(f" Cargando los datos procesados desde el archivo feather en {save_feather_file}")
            return pd.read_feather(save_feather_file)
        except FileNotFoundError:
            print(f"El archivo feather no existe en {save_feather_file} y no se pudieron procesar los archivos CDF")


    def cdf_read(cdf_file_path: Path) -> pd.DataFrame:
        """
        Función encargada de leer el archivo CDF y convertirlo a un DataFrame. Además, se encarga de convertir la columna "Epoch" a formato datetime y renombrar las columnas 'E' a 'E_Field' y 'F' a 'B_Total'
        """
        try:
            cdf = cdflib.CDF(cdf_file_path)
        except FileNotFoundError or Exception as e:
            print(f"Error al abrir o leer el archivo CDF {e}")

        cdf_dict = {}
        
        for key in cdf.cdf_info().zVariables:
            cdf_dict[key] = cdf[key][...]

        cdf_df = pd.DataFrame(cdf_dict)
        cdf_df["Epoch"] = pd.to_datetime(cdflib.cdfepoch.encode(cdf_df["Epoch"].values))
        cdf_df.rename(columns = {"E": "E_field", "F": "B_Total"})

        return cdf_df
    
    def bad_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Función encargada de procesar los datos erroneos del DataFrame entregado. Para cada columna, se establece un umbral máximo basado en el valor máximo de la columna, redondeado hacia abajo a dos decimales. Los valores que excedan este umbral se reemplazan por NaN. Luego, se realiza una interpolación lineal para llenar los valores faltantes, seguido de un backfill y un forwardfill con un límite de 2 para cada uno. Finalmente, devuelve el DataFrame limpiado.
        """
        if df.empty: return df

        processed_df = df.copy()

        for col in df.columns:
            if col == "Epoch": continue

            max_threshold = np.floor(processed_df[col].max() * 100) / 100
            df.loc[col >= max_threshold, col] = np.nan

            df[col] = df[col].interpolate(method = "linear")
            df[col] = df[col].bfill(limit = 2)
            df[col] = df[col].ffill(limit = 2)

            return df
        
