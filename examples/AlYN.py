from glob import glob

import numpy as np
from larch import Group
from larch.io import merge_groups

from ibr_xas import IbrXas


def read_and_merge_spectra(file_paths: list[str]):
    merged_spectra: list[Group] = []

    energy_grid: np.ndarray | None = None

    for file_path in file_paths:
        files = glob(file_path)
        files.sort()

        group_list: list[Group] = []
        for file in files:
            data = np.loadtxt(file)

            iff = data[:, 4]
            i0 = data[:, 1]
            energy = data[:, 0]

            mu = iff / i0
            if energy_grid is None:
                energy_grid = energy

            else:
                mu = np.interp(energy_grid, energy, mu)

            group_list.append(Group(energy=energy_grid, mu=mu))
        print(file)
        print(group_list)
        group = merge_groups(group_list)

        merged_spectra.append(group)

    return merged_spectra


def main():
    angles = [25, 30, 35, 40, 45, 50]

    file_paths = [f"./data/AlYN/AlYN-R{angle}*.dat" for angle in angles]

    labels = [f"AlYN-R{angle}.dat" for angle in angles]
    merged_spectra = read_and_merge_spectra(file_paths)

    ix = IbrXas(group_list=merged_spectra, file_list=labels)

    ix.calc_bragg_iter().save_dat()


if __name__ == "__main__":
    main()
