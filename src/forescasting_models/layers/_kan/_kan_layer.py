import numpy as np
import torch
import torch.nn as nn

from ._utils import *


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
                 scale_base_mu: float = 0.0,
                 scale_base_sigma: float = 1.0,                
                 sb_trainable: bool = True,
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
            - scale_sp (float): Peso de cuanto influe la parte Spline en cada conexión de la capa KAN
            - scale_base_mu (float): Controla el valor promedio inicial de los parámetros. Si es 0, significa que los pesos se inician alrededor de 0
            - scale_base_sigma (float): Controla la cantidad de ruido que se agrega. Si se aproxima a 0, los pesos quedan cerca de scale_base_mu
            - sb_trainable (bool): Controla si se entrena la función base, es decir, si es True, entonces scale_sb aprende durante el entrenamiento. Sino queda fijo
            - sp_trainable (bool): Controla si se entrenan los Spline, es decir, si es True, entonces scale_sp aprende durante el entrenamiento. Sino queda fijo
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

        # Peso de la función base
        self.scale_base = torch.nn.Parameter(scale_base_mu * (1 / np.sqrt(in_dim)) + scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * (1/np.sqrt(in_dim))).requires_grad_(sb_trainable)

        # Peso del Spline
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)

        self.grid_eps = grid_eps


    def forward(self, x):
        """
        """
        base = self.base_fun(x)

        y = coef2curve(x = x, grid = self.grid, coef = self.coef, k = self.k)
        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y

        y = torch.sum(y, dim = 1)

        return y
    

    def update_grid_from_samples(self, x, mode = "sample"):
        """
        Reajusta el grid de los B-spline según los datos

        Args:
            - mode (str): 
                -> "sample": Modo por defecto y usa los datos reales que se le pasa a la función
                -> "grid": Añade el uso del grid como puntos de muestreo, no los datos
        """

        def get_grid(num_interval):
            """
            Crea un nuevo grid
            """
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            margin = 0.00
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
            grid_uniform = (grid_adaptive[:, [-1]] - margin + h * torch.arange(num_interval + 1,)[None, :])
            grid = (self.grid_eps * grid_uniform) + (1 - self.grid_eps) * grid_adaptive

            return grid


        batch = x.shape[0]

        x_pos = torch.sort(x, dim = 0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - (2 * self.k)

        
        
        grid = get_grid(num_interval)

        if mode == "grid":
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)

        self.grid.data = extend_grid(grid, k_extend = self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)


        