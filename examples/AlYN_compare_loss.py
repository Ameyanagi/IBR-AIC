import os
from copy import deepcopy
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
font_size = 12
plt.rcParams.update({"font.size": font_size})
plt.rcParams.update({"axes.labelsize": font_size})
plt.rcParams.update({"xtick.labelsize": font_size})
plt.rcParams.update({"ytick.labelsize": font_size})
plt.rcParams.update({"legend.fontsize": font_size - 2})

plt.rcParams.update({"legend.frameon": True})
plt.rcParams.update({"legend.framealpha": 1.0})
plt.rcParams.update({"legend.fancybox": True})
plt.rcParams.update({"legend.numpoints": 1}) plt.rcParams.update({"patch.linewidth": 0.5}) plt.rcParams.update({"patch.edgecolor": "black"}) def plot_group_list( group_list: list[Group], label_list: list[str],
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
    ax.legend(loc="lower right")

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scaled raw absorption coefficient")

    save_path = os.path.join(save_dir, f"{save_prefix}energy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    ax.set_ylim(1.0, 1.5)
    save_path = os.path.join(save_dir, f"{save_prefix}energy_expand_y.png")
    fig.savefig(save_path, dpi=300)

    ax.set_xlim(e0 - 20, e0 + 80)
    ax.set_ylim(0, 2.0)
    save_path = os.path.join(save_dir, f"{save_prefix}energy_xanes.png")

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)


def plot_group_list_publication(
    group_list: list[Group],
    label_list: list[str],
    save_dir: str = "./output/",
    save_prefix: str = "",
    scale_dict: dict | None = None,
    edge_step: float | None = None,
    plot_dict_list: list[str] | None = None,
    e0: float | None = None,
):
    if scale_dict is None:
        raise ValueError("scale_dict must be given")

    if edge_step is None:
        raise ValueError("edge_step must be given")

    if plot_dict_list is None:
        raise ValueError("plot_dict must be given")

    if e0 is None:
        raise ValueError("e0 must be given")

    cols = 2
    rows = len(plot_dict_list)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for plot_dict_key, ax_row in zip(plot_dict_list, ax):
        merged_spectra_tmp = deepcopy(group_list)
        for i, merged_spectrum in enumerate(merged_spectra_tmp):
            merged_spectrum.e0 = e0
            merged_spectrum.mu = (
                merged_spectrum.mu
                * scale_dict[plot_dict_key][i]
                / edge_step
                / scale_dict[plot_dict_key][-1]
            )
            print(scale_dict[plot_dict_key][i])

        for i, group, label in zip(
            range(len(merged_spectra_tmp)), merged_spectra_tmp, label_list
        ):
            ax_row[0].plot(
                group.energy,
                group.mu,
                label=label,
                linewidth=0.5,
                color=f"C{i}",
            )

            ax_row[1].plot(
                group.energy,
                group.mu,
                label=label,
                linewidth=0.5,
                color=f"C{i}",
            )

        ax_row[0].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_row[0].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax_row[1].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_row[1].xaxis.set_minor_locator(ticker.MaxNLocator(20))

        # set labels
        ax_row[0].set_xlabel("Energy (eV)")
        ax_row[0].set_ylabel("Scaled raw absorption coefficient")

        save_path = os.path.join(save_dir, f"{save_prefix}energy.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ax_row[1].set_ylim(1.0, 1.5)
        ax_row[1].set_xlabel("Energy (eV)")
        ax_row[1].set_ylabel("Scaled raw absorption coefficient")

    ax[0, 0].legend(loc="lower right")

    figure_labels = ["(a) MSRE", "(b)", "(c) MAE", "(d)", "(e) MSE", "(f)"]

    for ax_item, label in zip(ax.flatten(), figure_labels):
        ax_item.text(
            x=0.02,
            y=0.93,
            s=f"{label}",
            transform=ax_item.transAxes,
            fontsize=font_size,
        )

    save_path = os.path.join(save_dir, f"{save_prefix}.png")
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
    angles = [25, 30, 50]
    file_paths = [f"./data/AlYN/AlYN-R{angle}*.dat" for angle in angles]

    file_list = [f"AlYN {angle}deg.dat" for angle in angles]
    labels = [f"AlYN {angle}$^\circ$" for angle in angles]
    merged_spectra = read_and_merge_spectra(file_paths)

    # Remove the bragg peak with IbrXas

    ix = IbrXas(group_list=merged_spectra, file_list=file_list)

    ix.calc_bragg_iter()

    group_list = generate_larch_group_list(ix)

    # Merge spectra
    merged_bragg_peak_removed_spectrum = merge_groups(group_list)

    merged_spectra.append(merged_bragg_peak_removed_spectrum)
    labels.append("AlYN IBR")

    ix_scale = IbrXas(group_list=merged_spectra)

    scale_dict = {}

    for weight in ["MSRE", "MAE", "MSE"]:
        scale_dict[weight] = ix_scale.loss_spectrum(
            ix_scale.mu_list, ix_scale.mu_list[0], 0, weight=weight
        )

    print(scale_dict)

    e0 = 17041.900

    pre_edge_kws: dict = {
        "nnorm": 3,
        "pre1": -152.2,
        "pre2": -35.90,
        "norm1": 150,
        "norm2": 1398.848,
    }
    pre_edge(merged_bragg_peak_removed_spectrum, **pre_edge_kws)

    edge_step = merged_bragg_peak_removed_spectrum.edge_step

    merged_spectra = merged_spectra[:-1]
    labels = labels[:-1]

    for key, scale in scale_dict.items():
        merged_spectra_tmp = deepcopy(merged_spectra)
        for i, merged_spectrum in enumerate(merged_spectra_tmp):
            merged_spectrum.e0 = e0
            merged_spectrum.mu = merged_spectrum.mu * scale[i] / edge_step / scale[-1]

        plot_group_list(
            merged_spectra_tmp,
            labels,
            save_prefix=f"AlYN_{key}_comparison",
        )

    plot_group_list_publication(
        group_list=merged_spectra,
        label_list=labels,
        save_prefix="publication_AlYN_comparison",
        scale_dict=scale_dict,
        edge_step=edge_step,
        plot_dict_list=["MSRE", "MAE", "MSE"],
        e0=e0,
    )


if __name__ == "__main__":
    main()
