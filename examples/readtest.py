import numpy as np


def main():
    path = "./mock_data/noise0.05_peakposition2.0_width5.txt"

    print(np.loadtxt(path))


if __name__ == "__main__":
    main()
