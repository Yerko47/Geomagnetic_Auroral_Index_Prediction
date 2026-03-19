import numpy as np
import torch.nn as nn

def linear(input_size: int, output_size: int, bias = True, dropout: float = None):
    """
    Crea una capa fully-connected, con la opción de agregar dropout antes de la capa
    Args:
        input_size (int):
        outout_size (int):
        bias (Bool):
        dropout (float):
    """

    lin = nn.Linear(input_size, output_size, bias = bias)
    if dropout:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin
    
def linspace(backcast_length: int, forescast_length: int, centered: bool = False):
    """
    Define el eje temporal para reconstruir señales, donde genera un eje temporal normalizado para:
    - backcast: Representa la parte de la señal histórica que el bloque intenta "explicar" o recostruir
    - forecast: Contribución de ese bloque a la predicción futura
    Args:
        backcast_length (int):
        forecast_length (int):
        centered (bool): Para centered = False -> tiempo va de 0 hacia el futuro. Mientras que para centered = True -> El pasado es negativo y el futuro es positivo
    """

    if centered:
        norm = max(backcast_length, forescast_length)
        start = -backcast_length
        stop = forescast_length - 1
    else:
        norm = backcast_length + forescast_length
        start = 0
        stop = backcast_length + forescast_length - 1
    
    lin_space = np.linspace(start = (start/norm), stop = (stop/norm), num = backcast_length + forescast_length, dtype = np.float32)

    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]

    return b_ls, f_ls

