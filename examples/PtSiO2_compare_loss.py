import os
from glob import glob
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
from larch import Group
from larch.io import merge_groups
from larch.xafs import autobk, pre_edge, xftf

from ibr_aic import IbrAic

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
            group.mu,
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


def generate_larch_group_list(ix: IbrAic) -> list[Group]:
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

        # Remove the bragg peak with IbrAic
        ix = IbrAic(group_list=merged_spectra, file_list=file_list)

        ix.calc_bragg_iter().save_dat()

        group_list = generate_larch_group_list(ix)

        # Merge spectra
        merged_bragg_peak_removed_spectrum = merge_groups(group_list)

        merged_spectra.append(merged_bragg_peak_removed_spectrum)
        labels.append(f"Pt/SiO$_2$ {temp_label} {gas} IBR")

        ia_scale = IbrAic(group_list=merged_spectra)

        scale_dict = {}

        for weight in ["MSRE", "MAE", "MSE"]:
            scale_dict[weight] = ia_scale.loss_spectrum(
                ia_scale.mu_list, ia_scale.mu_list[-1], -1, weight=weight
            )

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

        merged_spectra = merged_spectra[:-1]
        labels = labels[:-1]

        for key, scale in scale_dict.items():
            merged_spectra_tmp = deepcopy(merged_spectra)
            for i, merged_spectrum in enumerate(merged_spectra_tmp):
                merged_spectrum.e0 = e0
                merged_spectrum.mu = merged_spectrum.mu * scale[i] / edge_step

            plot_group_list(
                merged_spectra_tmp,
                labels,
                save_prefix=f"PtSiO2_{temp}_{gas}_{key}_comparison",
            )


if __name__ == "__main__":
    main()
