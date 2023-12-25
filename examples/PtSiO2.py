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
plt.rcParams.update({"legend.fontsize": font_size - 2})

plt.rcParams.update({"legend.frameon": True})
plt.rcParams.update({"legend.framealpha": 1.0})
plt.rcParams.update({"legend.fancybox": True})
plt.rcParams.update({"legend.numpoints": 1})
plt.rcParams.update({"patch.linewidth": 0.5})
plt.rcParams.update({"patch.edgecolor": "black"})


def plot_group_list(
    group_list: list[Group],
    label_list: list[str],
    save_dir: str = "./output/",
    save_prefix: str = "",
):
    e0 = group_list[-1].e0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(
            group.energy,
            group.mu + 0.1 * (len(group_list) - i - 1),
            label=label,
            linewidth=0.5,
            color=f"C{i}",
        )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scaled raw absorption coefficient (offset = 0.1)")

    save_path = os.path.join(save_dir, f"{save_prefix}energy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    ax.set_xlim(e0 - 20, e0 + 80)
    save_path = os.path.join(save_dir, f"{save_prefix}energy_xanes.png")

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    group = group_list[-1]
    label = label_list[-1]
    ax.plot(group.k, group.k**2 * group.chi, label=label, color="C0")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("k ($Å^{-1}$)")
    ax.set_ylabel("$k^2\chi(\mathrm{k})$ (Å$^{-2}$)")

    ax.set_xlim(0, 15)
    save_path = os.path.join(save_dir, f"{save_prefix}chi.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)


def plot_group_list_comparison(
    group_list: list[Group],
    label_list: list[str],
    save_dir: str = "./output/",
    save_prefix: str = "",
):
    e0 = group_list[-1].e0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(
            group.energy,
            group.flat,
            label=label,
            linewidth=0.5,
            color=f"C{i}",
        )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scaled raw absorption coefficient (offset = 0.1)")

    save_path = os.path.join(save_dir, f"{save_prefix}energy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    ax.set_xlim(e0 - 20, e0 + 80)
    save_path = os.path.join(save_dir, f"{save_prefix}energy_xanes.png")

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
    save_path = os.path.join(save_dir, f"{save_prefix}chi.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(group.r, group.chir_mag, label=label, color=f"C{i}")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("R ($Å$)")
    # tobe fixed
    ax.set_ylabel("$|R|\chi(\mathrm{R})$ (Å$^{-3}$)")

    ax.set_xlim(0, 6)
    save_path = os.path.join(save_dir, f"{save_prefix}chir_mag.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)


def read_and_merge_spectra(
    file_paths: list[str], fluorescence: bool = True
) -> list[Group]:
    merged_spectra: list[Group] = []

    energy_grid: np.ndarray | None = None

    for file_path in file_paths:
        files = glob(file_path)
        files.sort()

        group_list: list[Group] = []
        for file in files:
            data = np.loadtxt(file)

            iff = data[:, 4]
            it = data[:, 2]
            i0 = data[:, 1]

            energy = data[:, 0]

            if fluorescence:
                mu = iff / i0
            else:
                mu = np.log(i0 / it)

            if energy_grid is None:
                energy_grid = energy

            else:
                mu = np.interp(energy_grid, energy, mu)

            group_list.append(Group(energy=energy_grid, mu=mu))
        if len(group_list) > 1:
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
    angles = [30, 35, 40, 45]

    experiments: list[dict] = [
        {"temp": "350", "gas": "H2"},
        {"temp": "350", "gas": "N2"},
        {"temp": "RT", "gas": "CO"},
        {"temp": "RT", "gas": "N2"},
    ]

    for experiment in experiments:
        temp = experiment["temp"]
        gas = experiment["gas"]

        file_paths = [
            f"./data/PtSiO2/PtSiO2_Al_Plate_{temp}_{gas}_{angle}*.dat"
            for angle in angles
        ]

        file_list = [f"PtSiO2_{temp}_{gas}_{angle}.dat" for angle in angles]

        if temp == "RT":
            temp_label: str = "RT"
        else:
            temp_label: str = f"{temp}$^\circ$C"

        labels = [f"Pt/SiO$_2$ {temp_label} {gas} {angle}$^\circ$" for angle in angles]
        merged_spectra = read_and_merge_spectra(file_paths)

        print(len(merged_spectra))

        # Remove the bragg peak with IbrXas
        ix = IbrXas(group_list=merged_spectra, file_list=file_list)

        ix.calc_bragg_iter().save_dat()

        group_list = generate_larch_group_list(ix)

        # Merge spectra
        merged_bragg_peak_removed_spectrum = merge_groups(group_list)

        merged_spectra.append(merged_bragg_peak_removed_spectrum)
        labels.append(f"Pt/SiO$_2$ {temp_label} {gas} IBR")

        ix_scale = IbrXas(group_list=merged_spectra)

        scale = ix_scale.loss_spectrum(ix_scale.mu_list, ix_scale.mu_list[-1], -1)

        e0 = 11564

        pre_edge_kws: dict = {
            "nnorm": 3,
            "pre1": -180,
            "pre2": -50,
            "norm1": 150,
            "norm2": 970,
        }

        pre_edge(merged_bragg_peak_removed_spectrum, **pre_edge_kws)

        edge_step = merged_bragg_peak_removed_spectrum.edge_step

        for i, merged_spectrum in enumerate(merged_spectra):
            merged_spectrum.mu = merged_spectrum.mu * scale[i] / edge_step

        autobk_kws: dict = {"rbkg": 1.6, "kmin": 0, "kmax": None}

        xftf_kws: dict = {
            "kmin": 2,
            "kmax": 8,
            "dk": 2,
            "kweight": 2,
            "window": "Hanning",
        }

        for group in merged_spectra:
            group.e0 = e0
            pre_edge(group, **pre_edge_kws)
            autobk(group, **autobk_kws)
            xftf(group, **xftf_kws)

        plot_group_list(merged_spectra, labels, save_prefix=f"PtSiO2_{temp}_{gas}")

        # Comprison with the reference spectrum

        ref_file_paths: list[str] = [f"./data/PtSiO2_pellet/PtSiO2_{temp}_{gas}*.dat"]

        ref_group: list[Group] = read_and_merge_spectra(
            ref_file_paths, fluorescence=False
        )
        comparison_group_list: list[Group] = [
            merged_bragg_peak_removed_spectrum
        ] + ref_group

        comparison_labels = [
            f"Pt/SiO$_2$ {temp_label} {gas} IBR",
            f"Pt/SiO$_2$ {temp_label} {gas} ref",
        ]

        for group in comparison_group_list:
            group.e0 = 0
            pre_edge(group, **pre_edge_kws)
            autobk(group, **autobk_kws)
            xftf(group, **xftf_kws)

        plot_group_list_comparison(
            comparison_group_list,
            comparison_labels,
            save_prefix=f"PtSiO2_{temp}_{gas}_comparison",
        )


if __name__ == "__main__":
    main()
