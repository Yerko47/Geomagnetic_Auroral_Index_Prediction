import math
import numpy as np
import torch
import torch.nn as nn

from .utils import *


class KANLayer(nn.Module):
    """
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 k: int,
                 num: int, 
                 noises_scale: float, 
                 base_fun = nn.SiLU(), 
                 grid_range: list = [-1, 1], 
                 grid_eps: float = 0.02, 
                 scale_sp: float = 1.0, 
                 sp_trainable: bool = True):
        """
        Args:
            - in_dim (int): Dimensión de entrada
            - out_dim (int): Dimensión de salida
            - k (int): Orden del polinomio por partes de los spline
            - num (int): Número de intervalos del grid
            - noises_scale (float): Cantidad de ruido aleatorio para iniciar las funciones Spline
            - base_fun (nn.function):
            - grid_range (list): Define el rango del grid
            - grid_eps (float): Controla cómo se contruye el grid de los nodos cuando se adapta a los datos. Decide cuánto del grid es uniforme y cuanto depende de los datos
            - sclae_sp (float): Peso de cuanto influe la parte Spline en cada conexión de la capa KAN
            - sp_trainable (bool): Controla si se entrena, es decir, si es True, entonces scale_sp aprende durante el entrenamiento. Sino queda fijo
        """
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.num = num
        self.base_fun = base_fun

        # Crear grid inicial
        grid = torch.linspace(grid_range[0], grid_range[1], steps = num + 1)
        grid = extend_grid(grid, k_extend = k)
        self.grid = nn.Parameter(grid).requires_grad_(False)

        # Crear ruido inicial
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1/2) * (noises_scale / num)
        self.coef = nn.Parameter(curve2coef(self.grid[:, k, -k].permute(1, 0), noises, self.grid, k))

        # Inicializar máscaras de conexión
        self.mask = nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)

        # Escala de spline
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / math.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)

        self.grid_eps = grid_eps

    def forward(self, x):
        """
        """
        
        