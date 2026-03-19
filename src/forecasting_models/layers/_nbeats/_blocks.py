import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import *
from forecasting_models.layers._kan._kan_layer import KANLayer

class SeasonalMixin(nn.Module):
    """
    Mixin que proporciona funcionalidad de componente estacional para redes neuronales N-BEATS.
    Implementa funciones basis estacionales usando funcioniones trigonométricas (coseno y seno).
    """
    def _init_seasonal(self, backcast_length: int, forecast_length: int, thetas_dim: int, min_period: int, centered: bool = False):
        """
        Inicializa los parámetros y buffers para el componente estacional.
        Crea funciones basis estacionales (coseno y seno) para backcasting y forecasting.
        Args:
            backcast_length (int): Longitud del período de backcasting.
            forecast_length (int): Longitud del período de forecasting.
            thetas_dim (int): Dimensión de los parámetros theta.
            min_period (int): Período mínimo para calcular frecuencias.
            centered (bool): Si es True, centra las linspace. Por defecto False.
        Returns:
            None (registra buffers internos S_backcast y S_forecast).
        """
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.min_period = min_period

        backcast_linspace, forecast_linspace = linspace(self.backcast_length, self.forecast_length, centered)

        p_dim = (
            (thetas_dim // 2, thetas_dim // 2) if thetas_dim % 2 == 0 else (thetas_dim // 2, thetas_dim // 2 + 1)
        )
        osci_fun = [np.cos, np.sin]

        s_b = [
            torch.tensor(osci_fun[i](2 * np.pi * self.get_frequencies(p_i)[:, None] * backcast_linspace), dtype = torch.float32) for i, p_i in enumerate(p_dim)
        ]

        s_f = [
            torch.tensor(osci_fun[i](2 * np.pi * self.get_frequencies(p_i)[:, None] * forecast_linspace), dtype = torch.float32) for i, p_i in enumerate(p_dim)
        ]

        self.register_buffer("S_backcast", torch.cat(s_b, dim = 0))
        self.register_buffer("S_forecast", torch.cat(s_f, dim = 0))


    def seasonal_forward(self, x: torch.Tensor, theta_b_layer: nn.Module, theta_f_layer: nn.Module):
        """
        Realiza el paso forward del componente estacional.
        Calcula backcasting y forecasting multiplicando amplitudes por las bases estacionales.
        Args:
            x (torch.Tensor): Tensor de entrada.
            theta_b_layer (nn.Module): Capa para calcular amplitudes de backcasting.
            theta_f_layer (nn.Module): Capa para calcular amplitudes de forecasting.
        Returns:
            tuple: (backcast, forecast) con los valores de backcasting y forecasting.
        """
        amplitudes_backward = theta_b_layer(x)
        amplitudes_forward = theta_f_layer(x)

        backcast = amplitudes_backward.mm(self.S_backcast)
        forecast = amplitudes_forward.mm(self.S_forecast)

        return backcast, forecast
    

    def get_frequencies(self, n: int) -> np.ndarray:
        """
        Calcula frecuencias estacionales para el componente seasonal.
        Args:
            n (int): Número de frecuencias a generar.
        Returns:
            np.ndarray: Array de frecuencias lineales.
        """
        return np.linspace(0, (self.backcast_length + self.forecast_length) / self.min_period, n)
    

class TrendMixin:
    """
    Mixin que proporciona funcionalidad de componente de tendencia para redes neuronales N-BEATS.
    Implementa funciones basis polinomiales para modelar tendencias a través del tiempo.
    """
    def _init_trend(self, backcast_length: int, forecast_length: int, thetas_dim: int, centered: bool = False):
        """
        Inicializa los parámetros y buffers para el componente de tendencia.
        Crea funciones basis polinomiales para backcasting y forecasting.
        Args:
            backcast_length (int): Longitud del período de backcasting.
            forecast_length (int): Longitud del período de forecasting.
            thetas_dim (int): Grado del polinomio (número de términos).
            centered (bool): Si es True, centra las linspace. Por defecto False.
        Returns:
            None (registra buffers internos T_backcast y T_forecast).
        """
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        all_linspace = linspace(self.backcast_length, self.forecast_length, centered)

        norm = np.sqrt(forecast_length / thetas_dim)

        thetas_dims_range = np.array(range(thetas_dim))

        coefficients = [torch.tensor(type_linspace ** thetas_dims_range[:, None], dtype = torch.float32) for type_linspace in all_linspace]

        self.register_buffer = ("T_backcast", coefficients[0] * norm)
        self.register_buffer = ("T_forecast", coefficients[1] * norm)
        
    def trend_forward(self, x, theta_b_layer, theta_f_layer):
        """
        Realiza el paso forward del componente de tendencia.
        Calcula backcasting y forecasting multiplicando coeficientes por las bases polinomiales.
        Args:
            x (torch.Tensor): Tensor de entrada.
            theta_b_layer (nn.Module): Capa para calcular coeficientes de backcasting.
            theta_f_layer (nn.Module): Capa para calcular coeficientes de forecasting.
        Returns:
            tuple: (backcast, forecast) con los valores de backcasting y forecasting.
        """        
        backcast = theta_b_layer(x).mm(self.T_backcast)
        forecast = theta_f_layer(x).mm(self.T_forecast)

        return backcast, forecast






