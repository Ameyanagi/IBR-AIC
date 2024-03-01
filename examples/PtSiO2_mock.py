import itertools
import os
import random
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
from larch import Group
from larch.io import merge_groups
from larch.xafs import autobk, pre_edge, preedge, xftf

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


def plot_group_list_comparison_additional(
    group_list: list[Group],
    label_list: list[str] | None,
    save_dir: str = "./output/",
    save_prefix: str = "",
    additional_group: list[Group] | None = None,
    additional_label: list[str] | list[None] | None = None,
    ax_ext=None,
):
    e0 = group_list[-1].e0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    if additional_group is not None:
        if additional_label is None:
            additional_label = [None for _ in additional_group]

        for i, group, label in zip(
            range(len(group_list)), additional_group, additional_label
        ):
            ax.plot(
                group.energy,
                group.mu,
                label=label,
                color=f"C{i + len(group_list)}",
            )
            if ax_ext is not None:
                ax_ext[0].plot(
                    group.energy,
                    group.mu,
                    label=label,
                    color=f"C{i + len(group_list)}",
                )

    for i, group, label in zip(range(len(group_list)), group_list, label_list):
        ax.plot(
            group.energy,
            group.mu,
            label=label,
            color=f"C{i}",
        )

        if ax_ext is not None:
            ax_ext[0].plot(
                group.energy,
                group.mu,
                label=label,
                color=f"C{i}",
            )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.legend()

    # set labels
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Absorption coefficient")

    save_path = os.path.join(save_dir, f"{save_prefix}energy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300)

    if ax_ext is not None:
        ax_ext[0].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_ext[0].xaxis.set_minor_locator(ticker.MaxNLocator(20))
        ax_ext[0].set_xlabel("Energy (eV)")
        ax_ext[0].set_ylabel("Absorption coefficient")

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
    ax.set_xlabel("$k$ ($\mathrm{\AA}^{-1}$)")
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


def k2e(k: float | np.ndarray, E0: float) -> float | np.ndarray:
    return ((1000 / (16.2009**2)) * (k**2)) + E0


def e2k(E: float | np.ndarray, E0: float) -> float | np.ndarray:
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


def add_nonlinear_background_to_groups(
    groups: list[Group], e0: int | float, slopes: list[float]
):
    # The slope should not be too big for mock_data generation perpose
    #
    # slopes should be something like 0.0001- 0.00001

    for group, slope in zip(groups, slopes):
        group.mu += slope * (group.energy - e0)

    return groups


def generate_mock_data_from_Pt_SiO2(bragg_width: float = 5, slope: float = 0):
    Pt_SiO2_path: str = "./data/PtSiO2_pellet/PtSiO2_RT_N2*.dat"
    Pt_SiO2_group: Group = read_and_merge_spectra([Pt_SiO2_path])[0]

    bragg_peakposition: list[float] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    bragg_intensity: float = 5

    NS_ratio: list[float] = [
        0,  # 0% noise
        0.001,  # 0.1% noise
        0.01,  # 1% noise
        0.02,  # 2% noise
        0.05,  # 5% noise
    ]

    e0 = 11564

    pre_edge_kws: dict = {
        "nnorm": 3,
        "pre1": -180,
        "pre2": -50,
        "norm1": 150,
        "norm2": 970,
    }

    pre_edge(Pt_SiO2_group, **pre_edge_kws)

    edge_step = Pt_SiO2_group.edge_step

    noise_intensity: list[float] = [intensity * edge_step for intensity in NS_ratio]

    group_dict: dict = {}

    for intensity, NS in zip(noise_intensity, NS_ratio):
        group_list = add_bagg_peak_to_groups(
            Pt_SiO2_group,
            e0,
            bragg_peakposition,
            bragg_intensity,
            bragg_width,
        )
        group_list = add_noise_to_groups(group_list, intensity)
        group_list = add_nonlinear_background_to_groups(
            group_list, e0, [slope] * len(group_list)
        )

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot(Pt_SiO2_group.energy, Pt_SiO2_group.mu, label="Pt/SiO2 pellet RT N2")

        for i, group in enumerate(group_list):
            ax.plot(
                group.energy,
                group.mu,
                label=f"N/S {NS} {bragg_peakposition[i]}",
            )
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Absorption coefficient")
        ax.legend()

        fig.tight_layout(pad=0.5)
        fig.savefig(
            f"./output/Pt_SiO2_noise_{intensity}_width{bragg_width}_slope{slope}.png",
            dpi=300,
        )

        group_dict[intensity] = {}
        for i, group in enumerate(group_list):
            # group_dict[intensity][bragg_peakposition[i]] = group
            print(
                f"./mock_data/PtSiO2_noise{NS}_peakposition{bragg_peakposition[i]}_width{bragg_width}_slope{slope}.txt"
            )
            np.savetxt(
                f"./mock_data/PtSiO2_noise{NS}_peakposition{bragg_peakposition[i]}_width{bragg_width}_slope{slope}.txt",
                np.array([group.energy, group.mu]).T,
            )


def plot_mock_experiments(
    experiments: list[dict],
    reference: Group,
    ax_ext=None,
    pre_edge_kws: dict = None,
    autobk_kws: dict = None,
    xftf_kws: dict = None,
    save_prefix: str = "",
    group_labels: list[str] | None = None,
) -> None:
    # plots
    ncols_compare = 3
    nrows_compare = 1
    fig_compare, ax_compare = plt.subplots(
        nrows=nrows_compare,
        ncols=ncols_compare,
        figsize=(3 * ncols_compare, 3 * nrows_compare),
    )

    ax_compare = ax_compare.flatten()

    comparison_group_list = [reference]
    comparison_labels = ["ref"]

    if pre_edge_kws is None:
        pre_edge_kws: dict = {
            "nnorm": 3,
            "pre1": -180,
            "pre2": -50,
            "norm1": 150,
            "norm2": 970,
        }

    if autobk_kws is None:
        xftf_kws: dict = {
            "kmin": 2,
            "kmax": 8,
            "dk": 2,
            "kweight": 2,
            "window": "Hanning",
        }

    if xftf_kws is None:
        autobk_kws: dict = {"rbkg": 1.6, "kmin": 0, "kmax": None}

    file_path = []

    for condition in experiments:
        file_path.append(
            f"./mock_data/PtSiO2_noise{condition['intensity']}_peakposition{condition['peakposition']}_width{condition['width']}_slope{condition['slope']}.txt"
        )

    group_list = [read_xmu_spectrum(path) for path in file_path]

    ia = IbrAic(group_list=group_list, file_list=file_path)
    ia = ia.calc_bragg_iter().save_dat(output_dir=f"./mock_data/IBR_AIC/")

    deglitched_grouplist = generate_larch_group_list(ia)

    merged_spectra = merge_groups(deglitched_grouplist)

    pre_edge(merged_spectra, **pre_edge_kws)
    autobk(merged_spectra, **autobk_kws)
    xftf(merged_spectra, **xftf_kws)

    label = "IBR-AIC"

    comparison_group_list.append(merged_spectra)
    comparison_labels.append(label)

    group_list = [read_xmu_spectrum(path) for path in file_path]

    for group in group_list:
        pre_edge(group, **pre_edge_kws)
        autobk(group, **autobk_kws)
        xftf(group, **xftf_kws)

    plot_group_list_comparison_additional(
        comparison_group_list,
        # [None for _ in comparison_labels],
        comparison_labels,
        save_prefix=save_prefix,
        additional_group=group_list,
        additional_label=group_labels,
        ax_ext=ax_compare,
    )

    figure_labels_plot_all = ["(a)", "(b)", "(c)"]
    legend = ax_compare[0].legend(loc="lower right")

    for t in legend.get_texts():
        t.set_ha("right")

    for ax_item, label in zip(ax_compare, figure_labels_plot_all):
        ax_item.text(
            x=0.02,
            y=0.94,
            s=f"{label}",
            transform=ax_item.transAxes,
            fontsize=font_size,
        )

    fig_compare.tight_layout(pad=0.5)
    fig_compare.savefig("./output/" + save_prefix + "_all.png", dpi=300)


def process_IBR_AIC_mock_data():
    Pt_SiO2_path: str = "./data/PtSiO2_pellet/PtSiO2_RT_N2*.dat"
    Pt_SiO2_group: Group = read_and_merge_spectra([Pt_SiO2_path])[0]

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
        "e0": e0,
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

    pre_edge(Pt_SiO2_group, **pre_edge_kws)
    autobk(Pt_SiO2_group, **autobk_kws)
    xftf(Pt_SiO2_group, **xftf_kws)

    # IBR-AIC with 2 independent sharp bragg peaks

    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_2_indenpendent_bragg_peaks"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "5", "slope": "0"},
        {"intensity": "0", "peakposition": "6.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}"
        + "$\mathrm{\AA^{-1}}$"
        + "$\sigma$="
        + f"{condition['width']} eV"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    # IBR-AIC with 2 independent broad bragg peaks:
    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_2_indenpendent_broad_bragg_peaks"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "20", "slope": "0"},
        {"intensity": "0", "peakposition": "10.0", "width": "20", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}"
        + "$\mathrm{\AA^{-1}}$"
        + ", $\sigma$="
        + f"{condition['width']} eV"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    # IBR-AIC with 2 overlapping bragg peaks
    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_2_overlapping_broad_bragg_peaks"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "20", "slope": "0"},
        {"intensity": "0", "peakposition": "6.0", "width": "20", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}"
        + "$\mathrm{\AA^{-1}}$"
        + ", $\sigma$="
        + f"{condition['width']} eV"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    # IBR-AIC with noisy data
    NS_ratio: list[float] = [
        0.01,  # 1% noise
        0.001,  # 0.1% noise
        0,  # 0% noise
    ]

    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_effect_of_noise_0"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "5", "slope": "0"},
        {"intensity": "0", "peakposition": "6.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}"
        + "$\mathrm{\AA^{-1}}$"
        + f", N/S = {condition['intensity']}"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    prefix = (
        "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_effect_of_noise_0.001"
    )

    experiments: list[dict] = [
        {"intensity": "0.001", "peakposition": "4.0", "width": "5", "slope": "0"},
        {"intensity": "0", "peakposition": "6.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}"
        + "$\mathrm{\AA^{-1}}$"
        + f", N/S = {condition['intensity']}"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    prefix = (
        "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_effect_of_noise_0.01"
    )

    experiments: list[dict] = [
        {"intensity": "0.01", "peakposition": "4.0", "width": "5", "slope": "0"},
        {"intensity": "0", "peakposition": "6.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}"
        + "$\mathrm{\AA^{-1}}$"
        + f", N/S = {condition['intensity']}"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    # IBR-AIC with different non-linear background
    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_2_nonlinear_background_slope_0"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "5", "slope": "0"},
        {"intensity": "0", "peakposition": "14.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}" + "$\mathrm{\AA^{-1}}$"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )
    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_2_nonlinear_background_slope_0.0001"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "5", "slope": "0.0001"},
        {"intensity": "0", "peakposition": "14.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}" + "$\mathrm{\AA^{-1}}$"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )

    prefix = "publication_Pt_SiO2_IBR_AIC_comparison_combination_of_2_nonlinear_background_slope_0.00001"

    experiments: list[dict] = [
        {"intensity": "0", "peakposition": "4.0", "width": "5", "slope": "1e-05"},
        {"intensity": "0", "peakposition": "14.0", "width": "5", "slope": "0"},
    ]

    group_labels = [
        f"$k$={condition['peakposition']}" + "$\mathrm{\AA^{-1}}$"
        for condition in experiments
    ]

    plot_mock_experiments(
        experiments,
        Pt_SiO2_group,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws,
        group_labels=group_labels,
        save_prefix=prefix,
    )


def main():
    generate_mock_data_from_Pt_SiO2(bragg_width=5)
    generate_mock_data_from_Pt_SiO2(bragg_width=20)
    generate_mock_data_from_Pt_SiO2(bragg_width=5, slope=0.00001)
    generate_mock_data_from_Pt_SiO2(bragg_width=5, slope=0.0001)

    process_IBR_AIC_mock_data()


if __name__ == "__main__":
    main()
