from src.utils import *
from src.pipelines import *


import torch
import pandas as pd
import numpy as np

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



if __name__ == "__main__":
    main()
