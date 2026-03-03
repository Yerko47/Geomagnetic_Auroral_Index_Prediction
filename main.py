from src.utils import *
from src.pipelines import *


import torch
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

        datatorch_train = OMNIDataset(df, config, delay = delay, split = "train")
        datatorch_valid = OMNIDataset(df, config, delay = delay, split = "valid")
        datatorch_test = OMNIDataset(df, config, delay = delay, split = "test")
    

    

    #train_torch = OMNI_Dataset(df = df, config = config, delay = 30, split = "train")
    #val_torch = OMNI_Dataset(df = df, config = config, delay = 30, split = "val")
    #test = OMNI_Dataset(df = df, config = config, delay = 30, split = "test")
    #print(test.X.shape)
    #print(test.epoch)
    #test_torch, test_epoch = (test.X, test.y), test.epoch




if __name__ == "__main__":
    main()
