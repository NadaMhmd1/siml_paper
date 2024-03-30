from simulations import run_simulations
from reg_model import *


def main():
    run_simulations()
    train_first_round_models()
    extract_metadata()
    train_second_round_models()

if __name__ == "__main__":
    main()
