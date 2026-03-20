from src.utils import *
from src.pipelines import *
from src.forecasting_models.layers import *


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import time

def main():
    overrides = config_overrides()
    config = config_loader(config_path = None, overrides = overrides)

    paths = path_file()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = set_seed(42)

    ts = pd.Timestamp(config["dataset"]["time_range"]["end"]).year

    omni = config["dataset"]["omni_variables"]
    auroral = config["dataset"]["auroral_variables"]


    df = dataset(config = config, paths = paths)


    df = storm_selection(df = df, config = config, paths = paths)
    

    df, scaler = scaler_fit(df = df, config = config)

    delay_length = config["hyparameter"]["constant"]["delay_length"]

    for delay in delay_length:

        train_datatorch = OMNIDataset(df, config, delay = delay, split = "train")
        valid_datatorch = OMNIDataset(df, config, delay = delay, split = "valid")
        test_datatorch = OMNIDataset(df, config, delay = delay, split = "test")
    
        test_epoch = test_datatorch.epoch

        train_loader = DataLoader(train_datatorch, batch_size = 1024, shuffle = True)
        valid_loader = DataLoader(valid_datatorch, batch_size = 1024, shuffle = False)
        test_loader = DataLoader(test_datatorch, batch_size = 512, shuffle = False)

        del train_datatorch, valid_datatorch, test_datatorch
        



if __name__ == "__main__":
    main()
