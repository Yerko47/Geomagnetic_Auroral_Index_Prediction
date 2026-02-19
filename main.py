from src.config import *

def main():

    overrides = config_overrides()
    print(overrides)
    config = config_loader(config_path = None, overrides = overrides)

    print(config)
    



if __name__ == "__main__":
    main()
