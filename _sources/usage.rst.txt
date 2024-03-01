=====
Usage
=====

Following is an example usage of the class.

1. Prepare list of energy, and mu. The energy and mu has to be in the format of list of np.ndarray.
2. Create an instance of the class.
3. Call the calc_bragg_iter method to iteratively calculate the Bragg peaks.
4. Call the save_dat method to save the spectra.

         .. code-block:: python

            from ibr_aic import IbrAic

            energy_list, mu_list, file_list = prepare_spectra_from_qas(file_path)

            ia = IbrAic(energy_list, mu_list, file_list)
            ia.calc_bragg_iter().save_dat()
