import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import torch
from torch.utils.data import Dataset, DataLoader

#* SELECICCIÓN DE TORMENTAS GEOMAGNÉTICAS
def storm_selection(df: pd.DataFrame, config: dict, paths: dict) -> pd.DataFrame:
    """
    """

    storm_list_path = paths["raw_file"] / "storm_list.csv"
    
    if not storm_list_path.exists():
        return df
    else:
        storm_list_df = pd.read_csv(storm_list_path, header = None, names = ["Epoch"], parse_dates = ["Epoch"])
    

    segment = []
    for storm_time in storm_list_df["Epoch"]:
        start_time = storm_time - pd.Timedelta(hours = 36)
        end_time = storm_time + pd.Timedelta(hours = 36)

        storm_segment = df[df["Epoch"].between(start_time, end_time)]
        segment.append(storm_segment)

    df = pd.concat(segment, ignore_index = True)

    return df.reset_index(drop = True)


#* ESCALADO DE DATOS
def scaler_fit(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    """
    
    scaler_type = config["dataset"]["scaler_type"]
    omni_variables = df[config["dataset"]["omni_variables"]]

    if scaler_type.lower() == "standard":
        scaler = StandardScaler().fit(omni_variables)
    elif scaler_type.lower() == "robust":
        scaler = RobustScaler().fit(omni_variables)
    elif scaler_type.lower() == "minmax":
        scaler = MinMaxScaler().fit(omni_variables)
    else:
        raise ValueError(f"Scaler tipo '{scaler_type}' no reconocido.")

    auroral_variables = df[config["dataset"]["auroral_variables"]]
    epoch = df["Epoch"]

    omni_scaled = scaler.transform(omni_variables)
    omni_scaled_variables = pd.DataFrame(omni_scaled, columns = omni_variables.columns, index = omni_variables.index)

    df = pd.concat([epoch, omni_scaled_variables, auroral_variables], axis=1)

    return df, scaler