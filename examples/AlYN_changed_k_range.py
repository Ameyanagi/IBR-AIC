import os
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
from larch import Group
from larch.io import merge_groups
from larch.xafs import autobk, pre_edge, xftf

from ibr_aic import IbrAic

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

    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
    ax.legend()

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scaled raw absorption coefficient\n(offset = 0.1)")

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
    ax.set_xlabel("$k$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel("$k^2\chi(\mathrm{k})$ ($\mathrm{\AA}^{-2}$)")

    ax.set_xlim(0, 15)
    save_path = os.path.join(save_dir, f"{save_prefix}chi.png")
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


def plot_group_list_comparison(
    group_list: list[Group],
    label_list: list[str],
    save_dir: str = "./output/",
    save_prefix: str = "",
    ax_ext=None,
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

        if ax_ext is not None:
            ax_ext[0].plot(
                group.energy,
                group.flat,
                label=label,
                linewidth=0.5,
                color=f"C{i}",
            )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
    ax.legend()

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Normalized absorption coefficient")

    save_path = os.path.join(save_dir, f"{save_prefix}energy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    if ax_ext is not None:
        ax_ext[0].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_ext[0].xaxis.set_minor_locator(ticker.MaxNLocator(20))

        ax_ext[0].set_xlabel("Energy (eV)")
        ax_ext[0].set_ylabel("Normalized absorption coefficient")

    ax.set_xlim(e0 - 20, e0 + 80)
    save_path = os.path.join(save_dir, f"{save_prefix}energy_xanes.png")

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(group.k, group.k**2 * group.chi, label=label, color=f"C{i}")

        if ax_ext is not None:
            ax_ext[1].plot(
                group.k, group.k**2 * group.chi, label=label, color=f"C{i}"
            )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("$k$ ($\AA^{-1}$)")
    ax.set_ylabel("$k^2\chi(\mathrm{k})$ ($\mathrm{\AA}^{-2}$)")

    ax.set_xlim(0, 15)
    save_path = os.path.join(save_dir, f"{save_prefix}chi.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    if ax_ext is not None:
        ax_ext[1].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_ext[1].xaxis.set_minor_locator(ticker.MaxNLocator(20))

        ax_ext[1].set_xlabel("$k$ ($\mathrm{\AA}^{-1}$)")
        ax_ext[1].set_ylabel("$k^2\chi(\mathrm{k})$ ($\mathrm{\AA}^{-2}$)")

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(group.r, group.chir_mag, label=label, color=f"C{i}")

        if ax_ext is not None:
            ax_ext[2].plot(group.r, group.chir_mag, label=label, color=f"C{i}")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("$R$ ($\mathrm{\AA}$)")
    # tobe fixed
    ax.set_ylabel("$|\chi(R)|$ ($\mathrm{\AA}^{-3}$)")

    ax.set_xlim(0, 6)
    save_path = os.path.join(save_dir, f"{save_prefix}chir_mag.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    if ax_ext is not None:
        ax_ext[2].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_ext[2].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax_ext[2].set_xlabel("$R$ ($\mathrm{\AA}$)")
        ax_ext[2].set_ylabel("$|\chi(R)|$ ($\mathrm{\AA}^{-3}$)")


def read_reference_spectra(file_path: str) -> Group:
    data = np.loadtxt(file_path)
    return Group(energy=data[:, 0], mu=data[:, 1])


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
    # angles = [25, 30, 35, 40, 45, 50]
    angles = [25, 30, 50]
    # angles = [40, 45, 50]
    file_paths = [f"./data/AlYN/AlYN-R{angle}*.dat" for angle in angles]

    file_list = [f"AlYN {angle}deg.dat" for angle in angles]
    labels = [f"AlYN {angle}$^\circ$" for angle in angles]
    merged_spectra = read_and_merge_spectra(file_paths)

    # Remove the bragg peak with IbrAic
    ix = IbrAic(group_list=merged_spectra, file_list=file_list)

    ix.calc_bragg_iter().save_dat()

    group_list = generate_larch_group_list(ix)

    # Merge spectra
    merged_bragg_peak_removed_spectrum = merge_groups(group_list)

    merged_spectra.append(merged_bragg_peak_removed_spectrum)
    labels.append("AlYN IBR-AIC")

    ia_scale = IbrAic(group_list=merged_spectra)

    scale = ia_scale.loss_spectrum(ia_scale.mu_list, ia_scale.mu_list[-1], -1)

    e0 = 17041.900

    pre_edge_kws: dict = {
        "nnorm": 3,
        "pre1": -152.2,
        "pre2": -35.90,
        "norm1": 150,
        "norm2": 1398.848,
    }

    autobk_kws: dict = {"rbkg": 1.3}

    xftf_kws: dict = {
        "kmin": 2,
        "kmax": 12,
        "dk": 2,
        "kweight": 2,
        "window": "Hanning",
    }

    pre_edge(merged_bragg_peak_removed_spectrum, **pre_edge_kws)

    edge_step = merged_bragg_peak_removed_spectrum.edge_step

    for i, merged_spectrum in enumerate(merged_spectra):
        merged_spectrum.mu = merged_spectrum.mu * scale[i] / edge_step

    for group in merged_spectra:
        group.e0 = e0
        pre_edge(group, **pre_edge_kws)
        autobk(group, **autobk_kws)
        xftf(group, **xftf_kws)

    plot_group_list(merged_spectra, labels, save_prefix="publication_AlYN_")

    # Comprison with the reference spectrum

    ref_file_path = "./data/AlYN/AlYN-manual-deglitch.xmu"

    ref_group: Group = read_reference_spectra(ref_file_path)
    comparison_group_list: list[Group] = [merged_bragg_peak_removed_spectrum, ref_group]

    comparison_labels = [
        f"AlYN IBR-AIC",
        f"AlYN manual\ndeglitch",
    ]

    for group in comparison_group_list:
        group.e0 = e0
        pre_edge(group, **pre_edge_kws)
        autobk(group, **autobk_kws)
        xftf(group, **xftf_kws)

    fig, ax = plt.subplots(1, 3, figsize=(3 * 3, 3))
    plot_group_list_comparison(
        comparison_group_list,
        comparison_labels,
        save_prefix=f"AlYN_comparison_different_k",
        ax_ext=ax,
    )

    legend = ax[0].legend()

    for t in legend.get_texts():
        t.set_ha("right")

    figure_labels = ["(a)", "(b)", "(c)"]

    for ax_item, label in zip(ax.flatten(), figure_labels):
        ax_item.text(
            x=0.05,
            y=0.90,
            s=f"{label}",
            transform=ax_item.transAxes,
            fontsize=font_size * 1.2,
        )

    fig.tight_layout(pad=0.5)
    fig.savefig("./output/publication_AlYN_comparison_different_k.png", dpi=300)


if __name__ == "__main__":
    main()
