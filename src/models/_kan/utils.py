import torch

def BSpline(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calcula la base B-Spline para un tensor de entrada x, una cuadrícula dada y el grado k.
    Args:
        x (torch.Tensor): Tensor de valores donde se evalúa la base.
        grid (torch.Tensor): Tensor que define la cuadrícula de nodos.
        k (int): Grado de la B-Spline.
    Returns:
        torch.Tensor: Valores de la base B-Spline evaluados.
    """
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_ik1 = BSpline(x[:, :, 0], grid = grid[0], k = k - 1)

        value = (( (x - grid[:, :, : -(k + 1)]) / (grid[:, :, k: -1] - grid[:, :, : -(k + 1)]) ) * B_ik1[:, :, :-1]) + (
                 ( (grid[:, :, (k + 1):] - x) / (grid[:, :, (k + 1):] - grid[:, :, 1: -k]) ) * B_ik1[:, :, 1:])
        
     
    value = torch.nan_to_num(value)

    return value



def coef2curve(x: torch.Tensor, grid: torch.Tensor, coef: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evalúa una curva a partir de coeficientes de B-Spline.
    Args:
        x (torch.Tensor): Puntos donde se evalúa la curva.
        grid (torch.Tensor): Cuadrícula de nodos.
        coef (torch.Tensor): Coeficientes de la B-Spline.
        k (int): Grado de la B-Spline.
    Returns:
        torch.Tensor: Valores de la curva evaluada.
    """
    b_spline = BSpline(x, grid, k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_spline, coef.to(b_spline))

    return y_eval



def curve2coef(x: torch.Tensor, y: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calcula los coeficientes de B-Spline a partir de una curva dada.
    Args:
        x (torch.Tensor): Puntos donde se evalúa la curva.
        y (torch.Tensor): Valores de la curva.
        grid (torch.Tensor): Cuadrícula de nodos.
        k (int): Grado de la B-Spline.
    Returns:
        torch.Tensor: Coeficientes de la B-Spline.
    """
    batch = x.shape[0]
    in_dim = x.shape[1]
    out_dim = y.shape[2]
    n_coef = grid.shape[1] - k - 1

    mat = BSpline(x, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)

    y = y.permute(1, 2, 0).unsqueeze(dim = 3)

    try:
        coef = torch.linalg.lstsq(mat, y).solution[:, :, :, 0]
    except:
        print("Falló lstq")

    return coef



def extend_grid(grid: torch.tensor, k_extend: int) -> torch.Tensor:
    """
    """
    h = (grid[:,[-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - 1, grid], dim = 1)
        grid = torch.cat([grid, grid[:, [-1]] + 1], dim = 1)

    return grid



def sparse_mask(in_dim: int, out_dim: int):
    """
    """
    in_coords = torch.arange(in_dim) * (1/in_dim) + (1/(2 * in_dim))
    out_coords = torch.arange(out_dim) * (1/out_dim) + (1/(2 * out_dim))

    mask = torch.zeros(in_dim, out_dim)

    dist_mat = torch.abs(out_coords[:, None] - in_coords[None, :])

    in_nearest = torch.argmin(dist_mat, dim = 0)
    out_nearest = torch.argmin(dist_mat, dim = 1)

    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1,0)
    out_connection = torch.stack([torch.arange(out_dim), out_nearest]).permute(1,0)

    all_connection = torch.cat([in_connection, out_connection], dim = 0)

    mask[all_connection[:, 0], all_connection[:, 1]] = 1

    return mask