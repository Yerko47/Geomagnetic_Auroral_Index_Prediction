import torch

def b_spline(x, grid, k = 0):

    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_ik1 = b_spline(x[:, :, 0], grid = grid[0], k = k - 1)

        value = (( (x - grid[:, :, : -(k + 1)]) / (grid[:, :, k: -1] - grid[:, :, : -(k + 1)]) ) * B_ik1[:, :, :-1]) + (
                 ( (grid[:, :, (k + 1):] - x) / (grid[:, :, (k + 1):] - grid[:, :, 1: -k]) ) * B_ik1[:, :, 1:])
        
    # En caso de que el grid esté tenga algunos nodos iguales y pueda provocar divisiones por cero. 
    value = torch.nan_to_num(value)

    return value

