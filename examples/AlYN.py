import os
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
from larch import Group
from larch.io import merge_groups
from larch.xafs import autobk, pre_edge, xftf

from ibr_xas import IbrXas

plt.style.use(["science", "nature", "bright"])
font_size = 8
plt.rcParams.update({"font.size": font_size})
plt.rcParams.update({"axes.labelsize": font_size})
plt.rcParams.update({"xtick.labelsize": font_size})
plt.rcParams.update({"ytick.labelsize": font_size})
plt.rcParams.update({"legend.fontsize": font_size})


def plot_group_list(
    group_list: list[Group], label_list: list[str], save_dir: str = "./output/"
):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(
            group.energy,
            group.mu + 0.05 * (len(group_list) - i - 1),
            label=label,
            linewidth=0.5,
            color=f"C{i}",
        )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Raw absorption coefficient (offset = 0.05)")

    save_path = os.path.join(save_dir, "energy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(group.k, group.k**2 * group.chi, label=label, color=f"C{i}")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("k ($Å^{-1}$)")
    ax.set_ylabel("$k^2\chi(\mathrm{k})$ (Å$^{-2}$)")

    ax.set_xlim(0, 15)
    save_path = os.path.join(save_dir, "chi.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)


def read_and_merge_spectra(file_paths: list[str]) -> list[Group]:
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
        if len(group_list) > 0:
            group = merge_groups(group_list)

            merged_spectra.append(group)

    return merged_spectra


def generate_larch_group_list(ix: IbrXas) -> list[Group]:
    energy_list = ix.energy_list
    mu_list = ix.mu_list
    min_mu_list = ix.min_mu_list
    file_list = ix.file_list

    if file_list is None:
        file_list = [""] * len(energy_list)

    group_list: list[Group] = []
    for energy, mu, min_mu, file in zip(energy_list, mu_list, min_mu_list, file_list):
        group = Group(energy=energy, mu=mu + min_mu, filename=file)
        group_list.append(group)

    return group_list


def main():
    angles = [25, 30, 35, 40, 45, 50]

    file_paths = [f"./data/AlYN/AlYN-R{angle}*.dat" for angle in angles]

    labels = [f"AlYN {angle}deg.dat" for angle in angles]
    merged_spectra = read_and_merge_spectra(file_paths)

    # Remove the bragg peak with IbrXas
    ix = IbrXas(group_list=merged_spectra, file_list=labels)

    ix.calc_bragg_iter().save_dat()

    group_list = generate_larch_group_list(ix)

    # Merge spectra
    merged_bragg_peak_removed_spectrum = merge_groups(group_list)

    merged_spectra.append(merged_bragg_peak_removed_spectrum)
    labels.append("AlYN Bragg Peak Removed")

    pre_edge_kws: dict = {}
    autobk_kws: dict = {}

    for group in merged_spectra:
        pre_edge(group, **pre_edge_kws)
        autobk(group, **autobk_kws)

    plot_group_list(merged_spectra, labels)


if __name__ == "__main__":
    main()
