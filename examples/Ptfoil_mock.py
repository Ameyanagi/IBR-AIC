import os
from glob import glob
import random
from copy import deepcopy
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
from larch import Group
from larch.io import merge_groups
from larch.xafs import autobk, pre_edge, xftf

from ibr_aic import IbrAic


plt.style.use(["science", "nature", "bright"])
font_size = 11
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
    ax_ext=None,
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

        if ax_ext is not None:
            ax_ext.plot(
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

    if ax_ext is not None:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax_ext.set_xlabel("Energy (eV)")
        ax_ext.set_ylabel("Scaled raw absorption coefficient\n(offset = 0.1)")

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
    ax.set_xlabel("$k$ ($Å^{-1}$)")
    ax.set_ylabel("$k^2\chi(\mathrm{k})$ (Å$^{-2}$)")

    ax.set_xlim(0, 15)
    save_path = os.path.join(save_dir, f"{save_prefix}chi.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)


def plot_group_list_comparison(
    group_list: list[Group],
    label_list: list[str] | None,
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

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
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
    ax.set_xlabel("$k$ ($Å^{-1}$)")
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
        ax_ext[1].set_xlim(0, 15)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(group.r, group.chir_mag, label=label, color=f"C{i}")

        if ax_ext is not None:
            ax_ext[2].plot(group.r, group.chir_mag, label=label, color=f"C{i}")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("$R$ ($Å$)")
    # tobe fixed
    ax.set_ylabel("$|\chi(R)$ ($\mathrm{\AA}^{-3}$)")

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
        ax_ext[2].set_xlim(0, 6)


def read_xmu_spectrum(file_path: str) -> Group:
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


def k2e(k: float | np.ndarray, E0: float) -> float | np.ndarray:
    """
    Convert from k-space to energy in eV

    Parameters
    ----------
    k : float
        k value
    E0 : float
        Edge energy in eV

    Returns
    -------
    out : float
        Energy value

    See Also
    --------
    :func:`isstools.conversions.xray.e2k`
    """
    return ((1000 / (16.2009**2)) * (k**2)) + E0


def e2k(E: float | np.ndarray, E0: float) -> float | np.ndarray:
    """
    Convert from energy in eV to k-space

    Parameters
    ----------
    E : float
        Current energy in eV
    E0 : float
        Edge energy in eV

    Returns
    -------
    out : float
        k-space value

    See Also
    --------
    :func:`isstools.conversions.xray.k2e`
    """
    return 16.2009 * (((E - E0) / 1000) ** 0.5)


def gaussian(x: float | np.ndarray, mu: float, sigma: float) -> float:
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def add_bagg_peak_to_groups(
    group: Group,
    e0: float,
    k_position: list[float],
    intensity: float,
    width: float,
) -> list[Group]:
    groups = [deepcopy(group) for _ in k_position]

    for group, position in zip(groups, k_position):
        k = k2e(position, e0)
        group.mu += intensity * gaussian(group.energy, k, width)

    return groups


def add_noise_to_groups(groups: list[Group], intensity: float, seed=42) -> list[Group]:
    random.seed(seed)
    np.random.seed(seed)

    for group in groups:
        noise = np.random.normal(scale=intensity, size=len(group.mu))
        group.mu = group.mu + noise

    return groups


def change_intensity(
    groups: list[Group], max_intensity: float = 1.5, min_intensity: float = 0.5, seed=42
) -> list[Group]:
    random.seed(seed)
    np.random.seed(seed)

    intensity = np.random.uniform(min_intensity, max_intensity, len(groups))

    for group, inten in zip(groups, intensity):
        group.mu = group.mu * inten

    return groups


def generate_mock_data_from_Pt_foil(bragg_width: float = 5):
    Pt_foil_path: str = "./data/Pt_foil/Pt_foil.xmu"
    Pt_foil_group: Group = read_xmu_spectrum(Pt_foil_path)

    bragg_peakposition: list[float] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    bragg_intensity: float = 5

    noise_intensity: list[float] = [
        0,
        0.001,
        0.01,
        0.02,
        0.05,
    ]

    e0 = 11564

    group_dict: dict = {}

    for intensity in noise_intensity:
        group_list = add_bagg_peak_to_groups(
            Pt_foil_group,
            e0,
            bragg_peakposition,
            bragg_intensity,
            bragg_width,
        )
        group_list = add_noise_to_groups(group_list, intensity)

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot(Pt_foil_group.energy, Pt_foil_group.mu, label="Pt foil")

        for i, group in enumerate(group_list):
            ax.plot(
                group.energy,
                group.mu,
                label=f"Noise {intensity} {bragg_peakposition[i]}",
            )
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Absorption coefficient")
        ax.legend()

        fig.tight_layout(pad=0.5)
        fig.savefig(
            f"./output/Pt_foil_noise_{intensity}_width{bragg_width}.png", dpi=300
        )

        group_dict[intensity] = {}
        for i, group in enumerate(group_list):
            # group_dict[intensity][bragg_peakposition[i]] = group
            np.savetxt(
                f"./mock_data/noise{intensity}_peakposition{bragg_peakposition[i]}_width{bragg_width}.txt",
                np.array([group.energy, group.mu]).T,
            )


def process_IBR_AIC_mock_data():
    Pt_foil_path: str = "./data/Pt_foil/Pt_foil.xmu"
    Pt_foil_group: Group = read_xmu_spectrum(Pt_foil_path)

    bragg_peakposition: list[float] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    bragg_intensity: float = 5
    bragg_width: float = 5

    noise_intensity: list[float] = [
        0,
        0.001,
        0.01,
        0.02,
        0.05,
    ]

    # Pt foil
    e0 = 11564

    pre_edge_kws: dict = {
        "nnorm": 3,
        "pre1": -180,
        "pre2": -50,
        "norm1": 150,
        "norm2": 970,
    }

    xftf_kws: dict = {
        "kmin": 2,
        "kmax": 8,
        "dk": 2,
        "kweight": 2,
        "window": "Hanning",
    }
    autobk_kws: dict = {"rbkg": 1.6, "kmin": 0, "kmax": None}

    pre_edge(Pt_foil_group, **pre_edge_kws)
    autobk(Pt_foil_group, **autobk_kws)
    xftf(Pt_foil_group, **xftf_kws)

    # IBR-AIC with combination of 2 independent sharp bragg peaks
    for intensity in noise_intensity:
        experiments: list[dict] = [
            {"intensity": intensity, "peakposition": "2.0", "width": "5"},
            {"intensity": intensity, "peakposition": "6.0", "width": "5"},
            {"intensity": intensity, "peakposition": "14.0", "width": "5"},
        ]

        combination_2_conditions = list(itertools.combinations(experiments, 2))

        # plots
        ncols_compare = 3
        nrows_compare = 1
        fig_compare, ax_compare = plt.subplots(
            nrows=nrows_compare,
            ncols=ncols_compare,
            figsize=(3 * ncols_compare, 3 * nrows_compare),
        )

        ax_compare = ax_compare.flatten()

        comparison_group_list = [Pt_foil_group]
        comparison_labels = ["Pt foil"]

        for combination in combination_2_conditions:
            file_path = []
            for condition in combination:
                file_path.append(
                    f"./mock_data/noise{condition['intensity']}_peakposition{condition['peakposition']}_width{condition['width']}.txt"
                )

            group_list = [read_xmu_spectrum(path) for path in file_path]

            ia = IbrAic(group_list=group_list, file_list=file_path)
            ia = ia.calc_bragg_iter().save_dat(output_dir=f"./mock_data/IBR_AIC/")

            deglitched_grouplist = generate_larch_group_list(ia)

            merged_spectra = merge_groups(deglitched_grouplist)

            pre_edge(merged_spectra, **pre_edge_kws)
            autobk(merged_spectra, **autobk_kws)
            xftf(merged_spectra, **xftf_kws)

            label = f"$k$:({combination[0]['peakposition']} {combination[1]['peakposition']})"

            comparison_group_list.append(merged_spectra)
            comparison_labels.append(label)

        plot_group_list_comparison(
            comparison_group_list,
            # [None for _ in comparison_labels],
            comparison_labels,
            save_prefix=f"publication_Pt_foil_IBR_AIC_comparison_combination_of_2_indenpendent_bragg_peaks_noise{intensity}",
            ax_ext=ax_compare,
        )


def main():
    # generate_mock_data_from_Pt_foil(bragg_width=5)
    # generate_mock_data_from_Pt_foil(bragg_width=20)

    process_IBR_AIC_mock_data()

    # experiments: list[dict] = [
    #     {"temp": "350", "gas": "N2"},
    #     {"temp": "350", "gas": "H2"},
    #     {"temp": "RT", "gas": "N2"},
    #     {"temp": "RT", "gas": "CO"},
    # ]
    #
    # label_dict: dict = {
    #     "H2": "$\mathrm{H_2}$",
    #     "N2": "$\mathrm{N_2}$",
    #     "CO": "CO",
    # }
    #
    # ncols_plot_all = 2
    # nrows_plot_all = 2
    # fig_plot_all, ax_plot_all = plt.subplots(
    #     nrows=nrows_plot_all,
    #     ncols=ncols_plot_all,
    #     figsize=(3 * ncols_plot_all, 3 * nrows_plot_all),
    # )
    #
    # ncols_compare = 3
    # nrows_compare = 4
    # fig_compare, ax_compare = plt.subplots(
    #     nrows=nrows_compare,
    #     ncols=ncols_compare,
    #     figsize=(3 * ncols_compare, 3 * nrows_compare),
    # )
    #
    # ax_plot_all = ax_plot_all.flatten()
    #
    # for j, experiment in enumerate(experiments):
    #     temp = experiment["temp"]
    #     gas = experiment["gas"]
    #
    #     file_paths = [
    #         f"./data/PtSiO2/PtSiO2_Al_Plate_{temp}_{gas}_{angle}*.dat"
    #         for angle in angles
    #     ]
    #
    #     file_list = [f"PtSiO2_{temp}_{gas}_{angle}.dat" for angle in angles]
    #
    #     if temp == "RT":
    #         temp_label: str = "RT"
    #     else:
    #         temp_label: str = f"{temp}$^\circ$C"
    #
    #     labels = [
    #         # f"Pt/SiO$_2$ {temp_label} {label_dict[gas]} {angle}$^\circ$"
    #         f"{angle}$^\circ$"
    #         for angle in angles
    #     ]
    #     merged_spectra = read_and_merge_spectra(file_paths)
    #
    #     print(len(merged_spectra))
    #
    #     # Remove the bragg peak with IbrAic
    #     ix = IbrAic(group_list=merged_spectra, file_list=file_list)
    #
    #     ix.calc_bragg_iter().save_dat()
    #
    #     group_list = generate_larch_group_list(ix)
    #
    #     # Merge spectra
    #     merged_bragg_peak_removed_spectrum = merge_groups(group_list)
    #
    #     merged_spectra.append(merged_bragg_peak_removed_spectrum)
    #     labels.append(f"IBR-AIC")
    #     # labels.append(f"Pt/SiO$_2$ {temp_label} {label_dict[gas]}\nIBR-AIC")
    #
    #     ia_scale = IbrAic(group_list=merged_spectra)
    #
    #     scale = ia_scale.loss_spectrum(ia_scale.mu_list, ia_scale.mu_list[-1], -1)
    #
    #     e0 = 11564
    #
    #     pre_edge_kws: dict = {
    #         "nnorm": 3,
    #         "pre1": -180,
    #         "pre2": -50,
    #         "norm1": 150,
    #         "norm2": 970,
    #     }
    #
    #     pre_edge(merged_bragg_peak_removed_spectrum, **pre_edge_kws)
    #
    #     edge_step = merged_bragg_peak_removed_spectrum.edge_step
    #
    #     for i, merged_spectrum in enumerate(merged_spectra):
    #         merged_spectrum.mu = merged_spectrum.mu * scale[i] / edge_step
    #
    #     autobk_kws: dict = {"rbkg": 1.6, "kmin": 0, "kmax": None}
    #
    #     xftf_kws: dict = {
    #         "kmin": 2,
    #         "kmax": 8,
    #         "dk": 2,
    #         "kweight": 2,
    #         "window": "Hanning",
    #     }
    #
    #     for group in merged_spectra:
    #         group.e0 = e0
    #         pre_edge(group, **pre_edge_kws)
    #         autobk(group, **autobk_kws)
    #         xftf(group, **xftf_kws)
    #
    #     plot_group_list(
    #         merged_spectra,
    #         labels,
    #         save_prefix=f"PtSiO2_{temp}_{gas}",
    #         ax_ext=ax_plot_all[j],
    #     )
    #
    #     # Comprison with the reference spectrum
    #
    #     ref_file_paths: list[str] = [f"./data/PtSiO2_pellet/PtSiO2_{temp}_{gas}*.dat"]
    #
    #     ref_group: list[Group] = read_and_merge_spectra(
    #         ref_file_paths, fluorescence=False
    #     )
    #     comparison_group_list: list[Group] = [
    #         merged_bragg_peak_removed_spectrum
    #     ] + ref_group
    #
    #     # Add Manual deglitched spectra
    #
    #     manual_deglitch_file_path: str = (
    #         f"./manual_deglitch/PtSiO2_Al_Plate_{temp}_{gas}_manual_deglitch.xmu"
    #     )
    #
    #     manual_deglitch_group: Group = read_xmu_spectrum(manual_deglitch_file_path)
    #
    #     comparison_group_list.append(manual_deglitch_group)
    #
    #     comparison_labels = [f"IBR-AIC", f"ref", f"manual deglitch"]
    #
    #     for group in comparison_group_list:
    #         group.e0 = 0
    #         pre_edge(group, **pre_edge_kws)
    #         autobk(group, **autobk_kws)
    #         xftf(group, **xftf_kws)
    #
    #     plot_group_list_comparison(
    #         comparison_group_list,
    #         comparison_labels,
    #         save_prefix=f"PtSiO2_{temp}_{gas}_comparison",
    #         ax_ext=ax_compare[j],
    #     )
    #
    # figure_labels_plot_all = ["(a)", "(b)", "(c)", "(d)"]
    # figure_labels_compare = [
    #     "(a1)",
    #     "(a2)",
    #     "(a3)",
    #     "(b1)",
    #     "(b2)",
    #     "(b3)",
    #     "(c1)",
    #     "(c2)",
    #     "(c3)",
    #     "(d1)",
    #     "(d2)",
    #     "(d3)",
    # ]
    #
    # legend = ax_plot_all[0].legend(loc="lower right")
    #
    # for t in legend.get_texts():
    #     t.set_ha("right")
    #
    # legend = ax_compare[0, 0].legend(loc="lower right")
    #
    # for t in legend.get_texts():
    #     t.set_ha("right")
    #
    # for ax_item, label in zip(ax_plot_all, figure_labels_plot_all):
    #     ax_item.text(
    #         x=0.02,
    #         y=0.94,
    #         s=f"{label}",
    #         transform=ax_item.transAxes,
    #         fontsize=font_size,
    #     )
    #
    # for ax_item, label in zip(ax_compare.flatten(), figure_labels_compare):
    #     ax_item.text(
    #         x=0.05,
    #         y=0.90,
    #         s=f"{label}",
    #         transform=ax_item.transAxes,
    #         fontsize=font_size * 1.2,
    #     )
    #
    # fig_plot_all.tight_layout(pad=0.5)
    # fig_plot_all.savefig("./output/publication_PtSiO2_all.png", dpi=300)
    #
    # fig_compare.tight_layout(pad=0.5)
    # fig_compare.savefig("./output/publication_PtSiO2_comparison.png", dpi=300)
    #


if __name__ == "__main__":
    main()
