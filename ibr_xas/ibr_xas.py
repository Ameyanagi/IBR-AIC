# pright: strict

import os
from copy import copy
from glob import glob
from typing import Protocol, Self

import numpy as np
import scipy.optimize as opt


class HasEnergyMu(Protocol):
    """Protocol to define the class that has energy and mu

    This class is used for type hinting. Group class from xraylarch also has a energy and mu attributes, and can be used as an input for IbrXas class.
    """

    energy: np.ndarray
    mu: np.ndarray


class IbrXas:
    """IbrXas: A class to remove Bragg peaks from XAS spectra

    Usage:
        Following is an example usage of the class.\n
        1. Prepare list of energy, and mu. The energy and mu has to be in the format of list of np.ndarray.\n
        2. Create an instance of the class.\n
        3. Call the calc_bragg_iter method to iteratively calculate the Bragg peaks.\n
        4. Call the save_dat method to save the spectra.\n

        .. code-block:: python

            from ibr_xas import IbrXas

            energy_list, mu_list, file_list = prepare_spectra_from_qas(file_path)

            ix = IbrXas(energy_list, mu_list, file_list)
            ix.calc_bragg_iter().save_dat()

    Args:
        energy_list (list[np.ndarray]): List of the energy
        mu_list (list[np.ndarray]): List of the mu
        group_list (list[HasEnergyMu]): List of the group
        file_list (list[str]): List of the file path
    """

    energy_list: list[np.ndarray]
    mu_list: list[np.ndarray]
    min_mu_list: list[np.ndarray]
    scale_list: list[np.ndarray]
    file_list: list[str] | None
    ibr_loss: float

    def __init__(
        self,
        energy_list: list[np.ndarray] | None = None,
        mu_list: list[np.ndarray] | None = None,
        group_list: list[HasEnergyMu] | None = None,
        file_list=None,
    ) -> None:
        if group_list is not None:
            energy_list = [group.energy for group in group_list]
            mu_list = [group.mu for group in group_list]

        if energy_list is None or mu_list is None:
            raise ValueError("Please provide group_list or energy_list and mu_list")

        self.energy_list = energy_list
        self.mu_list = mu_list

        assert len(self.energy_list) == len(self.mu_list)

        if file_list is not None:
            assert len(file_list) == len(self.mu_list)

        self.file_list = file_list
        self.interpolate()

        self.min_mu_list = [np.zeros_like(mu) for mu in self.mu_list]

        self.scale_list = [np.ones_like(mu) for mu in self.mu_list]

        self.ibr_loss = np.inf

        if file_list is not None:
            assert len(file_list) == len(self.mu_list)
            self.file_list = file_list
        else:
            self.file_list = None

    def interpolate(self) -> Self:
        energy_grid = self.energy_list[0]

        for i in range(1, len(self.energy_list)):
            self.mu_list[i] = np.interp(
                energy_grid, self.energy_list[i], self.mu_list[i]
            )
            self.energy_list[i] = energy_grid

        return self

    def loss_func(
        self,
        x: float,
        spectrum: np.ndarray,
        reference: np.ndarray,
        index: np.ndarray | tuple,
    ) -> np.ndarray:
        """Mean square root error function for the optimization of the scaling factor

        The mean square is used for calculation of the loss function. If the mean square root or the mean absolute error is used, the optimization will be biased to the Bragg peaks, and there will be a large error around the Bragg peaks.
        The use of square root will enhance the possibility of fitting to the baseline spectrum.

        Args:
            x (float): scaling factor
            spectrum (np.ndarray): spectrum to be scaled
            reference (np.ndarray): reference spectrum
            index (np.ndarray): index of the energy range to be used for the calculation of the loss function

        Return:
            np.ndarray: mean square root error
        """

        scaled_spectrum = spectrum[index] * x

        return np.sqrt(abs(scaled_spectrum - reference[index])).sum()

    def loss_spectrum(
        self,
        spectrum_list: list[np.ndarray],
        reference: np.ndarray,
        ref_index: int,
        energy_range: list[float] | None = None,
        scale_range: list[float] | None = None,
    ) -> np.ndarray:
        if energy_range is None:
            energy_range = [-np.inf, np.inf]

        if scale_range is None:
            scale_range = [0.5, 1.5]

        scale = np.array([])

        index = np.where(
            (self.energy_list[ref_index] >= energy_range[0])
            & (self.energy_list[ref_index] <= energy_range[1])
        )

        for spectrum in spectrum_list:

            def func(x: float) -> np.ndarray:
                return self.loss_func(x, spectrum, reference, index)

            scale = np.append(
                scale,
                opt.minimize(
                    func, [1.0], bounds=[scale_range], options={"maxiter": 1000}
                ).x,
            )

        return scale

    def calc_bragg(
        self,
        energy_range: list[float] | None = None,
        scale_range: list[float] | None = None,
    ) -> Self:
        """Function to caculate the Bragg peaks

        The caulculation of the Bragg peaks is done in iterative manner and this function corresponds to 1 iteration.

        1. Choose a reference spectrum where the Bragg peaks will be calculated.\n
        2. Calaculate the scaling factor for the other spectra to match the reference spectrum. The loss fuction is the mean square root error, to reduce the effect of the Bragg peaks.\n
        3. Subtract the scaled spectra from the reference spectrum. This difference spectrum will be corresponding to the Bragg peaks, but it is not remove completely.\n
        4. Take an average of the difference spectra, add it to the reference spectrum, and go back to step 2.\n
        5. Repeat step 2 to 4 until the difference spectrum converges.\n

        Args:
            energy_range (list[float], optional): Energy range to be used for the calculation of the loss function. Defaults to [-np.inf, np.inf].
            scale_range (list[float], optional): Range of the scaling factor. Defaults to [0.5, 1.5].

        Returns:
            self: Reference to the class itself
        """
        minimum_mu_tmp: list[np.ndarray] = []
        scale_tmp: list[np.ndarray] = []

        if energy_range is None:
            energy_range = [-np.inf, np.inf]
        elif len(energy_range) < 2:
            print(
                "There will be no calculation performed, because the len(energy_range) is <2"
            )
            return self
        else:
            energy_range = [min(energy_range), max(energy_range)]

        spectrum_tmp = [
            mu + min_mu for mu, min_mu in zip(self.mu_list, self.min_mu_list)
        ]

        for i in range(len(self.mu_list)):
            reference: np.ndarray = copy(spectrum_tmp[i])
            scale: np.ndarray = self.loss_spectrum(
                spectrum_tmp, reference, i, energy_range, scale_range
            )
            minimum_mu: np.ndarray = np.zeros_like(self.energy_list[i])

            for j in range(len(self.mu_list)):
                diff = spectrum_tmp[j] * scale[j] - reference
                minimum_mu = np.minimum(
                    minimum_mu, np.minimum(np.zeros_like(diff), diff)
                )
            scale_tmp.append(scale)
            minimum_mu_tmp.append(minimum_mu / len(self.mu_list) + self.min_mu_list[i])

        delta_mu = 0
        for i in range(len(minimum_mu_tmp)):
            delta_mu += (
                np.abs(self.min_mu_list[i] - minimum_mu_tmp[i])
                / len(self.min_mu_list[i])
            ).sum()

        self.ibr_loss = delta_mu

        self.min_mu_list = minimum_mu_tmp
        self.scale_list = scale_tmp

        return self

    def calc_bragg_iter(
        self,
        energy_range: list[float] | None = None,
        scale_range: list[float] | None = None,
        criteria: float = 1e-10,
        max_iter=200,
    ) -> Self:
        """Iterative calculation of the Bragg peaks

        Calculate the Bragg peaks in iterative manner. The calculation will be stopped when the difference between the previous and current update is lower than the criteria.
        The amount of update is defined by MAE of the spectra, and has a same unit as absorption.

        Args:
            energy_range (list[float], optional): Energy range to be used for the calculation of the loss function. Defaults to [-np.inf, np.inf].
            scale_range (list[float], optional): Range of the scaling factor. Please change this value if if the absolute intensity changes significantly larger than 1. Defaults to [0.5, 1.5].
            criteria (float, optional): The criteria to stop the calculation. Defaults to 1e-5.
            max_iter (int, optional): Maximum number of iteration. Defaults to 200.

        Returns:
            self: Reference to the class itself
        """

        for i in range(max_iter):
            previous_ibr_loss = self.ibr_loss
            self.calc_bragg(energy_range, scale_range)

            if abs(previous_ibr_loss - self.ibr_loss) < criteria:
                print("The calculation converged after {} iterations".format(i))
                break

        return self

    def save_dat(
        self, file_list: list[str] | None = None, output_dir: str = "./output/"
    ) -> Self:
        """Save dat

        Save the spectra to the dat file. The file name will be the basename of the file_list.

        Args:
            file_list (list[str], optional): List of the file names. If none, the file_name will be taken from the __init__ arguments of the class. Defaults to None.
            output_dir (str, optional): The directory to save the ouput dat files. Defaults to "./output/"

        Returns:
            Self
        """
        if file_list is not None:
            assert len(file_list) == len(self.mu_list)
            self.file_list = file_list

        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        for i in range(len(self.energy_list)):
            if self.file_list is not None:
                save_path = os.path.join(
                    output_dir,
                    os.path.basename(self.file_list[i]) + "_remove_bragg.dat",
                )

                np.savetxt(
                    save_path,
                    np.array(
                        [self.energy_list[i], self.mu_list[i] + self.min_mu_list[i]]
                    ).T,
                )
            else:
                np.savetxt(
                    "spectrum_{}.dat".format(i),
                    np.array(
                        [self.energy_list[i], self.mu_list[i] + self.min_mu_list[i]]
                    ).T,
                )

        return self


def prepare_spectra_from_QAS(
    file_path: str, fluorescence: bool = True
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Example of preparing a input for IbrXas class

    Args:
        file_path (str): Path to the XAS file. The file_path has to be in the format of glob.
        fluorescence (bool, optional): If True, the fluorescence XAS will be used. If False, the absorption XAS will be used. Defaults to True.
        file_path (str): Path to the XAS file. The file_path has to be in the format of glob.

    Returns:
        energy_list(list[np.ndarray]): List of the energy
        mu_list(list[np.ndarray]): List of the mu
        file_list(list[str]): List of the file path

    """
    files = glob(file_path)
    energy_list: list[np.ndarray] = []
    mu_list: list[np.ndarray] = []
    file_list: list[str] = []

    for file in files:
        data = np.loadtxt(file)

        energy = data[:, 0]

        if fluorescence:
            mu = data[:, 4] / data[:, 1]
        else:
            mu = np.log(data[:, 2] / data[:, 1])

        energy_list.append(energy)
        mu_list.append(mu)
        file_list.append(file)

    return energy_list, mu_list, file_list


def prepare_group_from_QAS(
    file_path: str, fluorescence: bool = True
) -> tuple[list[HasEnergyMu], list[str]]:
    """Example of preparing an input(group) for IbrXas

    Please `pip install xraylarch` if you want to use this function.

    Args:
        file_path (str): Path to the XAS file. The file_path has to be in the format of glob.
        fluorescence (bool, optional): If True, the fluorescence XAS will be used. If False, the absorption XAS will be used. Defaults to True.

    Returns:
        group_list(list[Group]): List of the group
        file_list(list[str]): List of the file path
    """

    try:
        from larch import Group
    except ImportError:
        raise ImportError(
            "Please `pip install xraylarch` in order to use this function"
        )

    files = glob(file_path)
    group_list = []
    file_list = []
    for file in files:
        data = np.loadtxt(file)
        energy = data[:, 0]
        if fluorescence:
            mu = data[:, 4] / data[:, 1]
        else:
            mu = np.log(data[:, 2] / data[:, 1])

        group_list.append(Group(energy=energy, mu=mu))
        file_list.append(file)

    return group_list, file_list


if __name__ == "__main__":
    pass
