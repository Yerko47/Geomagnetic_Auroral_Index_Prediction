from src.utils import *

def main():

    overrides = config_overrides()
    config = config_loader(config_path = None, overrides = overrides)

    paths = path_file()

    print(config)
    



if __name__ == "__main__":
    main()
