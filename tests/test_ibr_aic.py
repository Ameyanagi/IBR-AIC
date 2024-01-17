"""Tests for `ibr_aic` package."""

import pytest

# from click.testing import CliRunner

from ibr_aic import ibr_aic

# from ibr_aic import cli


@pytest.fixture
def spectra_data():
    file_path = "tests/testfiles/AlYN/*.dat"

    energy_list, mu_list, file_list = ibr_aic.prepare_spectra_from_QAS(
        file_path, fluorescence=True
    )

    return energy_list, mu_list, file_list


@pytest.fixture
def group_data():
    file_path = "tests/testfiles/AlYN/*.dat"
    group_list, file_list = ibr_aic.prepare_group_from_QAS(file_path, fluorescence=True)
    return group_list, file_list


def test_prepare_spectra_from_QAS(spectra_data):
    assert len(spectra_data[0]) == 60
    assert len(spectra_data[1]) == 60
    assert len(spectra_data[2]) == 60


def test_prepare_group_from_QAS(group_data):
    assert len(group_data[0]) == 60
    assert len(group_data[1]) == 60


def test_IbrAic_init(spectra_data):
    ix = ibr_aic.IbrAic(
        energy_list=spectra_data[0], mu_list=spectra_data[1], file_list=spectra_data[2]
    )
    assert len(ix.energy_list) == 60
    for i in range(len(ix.energy_list)):
        assert (ix.energy_list[i] == spectra_data[0]).all()
    assert len(ix.mu_list) == 60

    assert ix.file_list == spectra_data[2]


def test_IbrAic_init_group(group_data):
    ix = ibr_aic.IbrAic(group_list=group_data[0], file_list=group_data[1])

    assert len(ix.energy_list) == 60
    for i in range(len(ix.energy_list)):
        assert (ix.energy_list[i] == group_data[0][0].energy).all()
    assert len(ix.mu_list) == 60
    assert ix.file_list == group_data[1]


def test_IbrAic_energy_list_error(spectra_data):
    with pytest.raises(
        ValueError, match="Please provide group_list or energy_list and mu_list"
    ):
        ibr_aic.IbrAic(
            energy_list=None, mu_list=spectra_data[1], file_list=spectra_data[2]
        )

    # assert excinfo.value == ValueError(
    #     "Please provide group_list or energy_list and mu_list"
    # )


def test_IbrAic_mu_list_error(spectra_data):
    with pytest.raises(
        ValueError, match="Please provide group_list or energy_list and mu_list"
    ):
        ibr_aic.IbrAic(
            energy_list=spectra_data[0], mu_list=None, file_list=spectra_data[2]
        )


def test_IbrAic_file_list_None(spectra_data):
    ix = ibr_aic.IbrAic(
        energy_list=spectra_data[0], mu_list=spectra_data[1], file_list=None
    )

    assert ix.file_list is None


def test_IbrAic_group_list_error(group_data):
    with pytest.raises(
        ValueError, match="Please provide group_list or energy_list and mu_list"
    ):
        ibr_aic.IbrAic(group_list=None, file_list=group_data[1])


def test_IbrAic_energy_mu_different_length(spectra_data):
    with pytest.raises(AssertionError):
        ibr_aic.IbrAic(
            energy_list=spectra_data[0],
            mu_list=spectra_data[1][:-1],
        )


# def response():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
#
#
# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string
#
#
# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert "ibr_aic.cli.main" in result.output
#     help_result = runner.invoke(cli.main, ["--help"])
#     assert help_result.exit_code == 0
#     assert "--help  Show this message and exit." in help_result.output
