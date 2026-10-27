"""Check agronomic components against independently published values
(brief §5 rule 4) rather than against our own formulas."""
import numpy as np
import pytest

from tailwater.env.crop import CropParams
from tailwater.env.weather import extraterrestrial_radiation, hargreaves_et0
from tailwater.env.yield_model import fao33_relative_yield


def test_ra_fao56_example_8():
    # FAO-56 (Allen et al. 1998) Example 8: 3 September (J=246) at 20 deg S:
    # dr = 0.985, delta = 0.120 rad, ws = 1.527 rad, Ra = 32.2 MJ m-2 day-1
    ra, mid = extraterrestrial_radiation(-20.0, 246)
    assert mid["dr"] == pytest.approx(0.985, abs=5e-4)
    assert mid["delta"] == pytest.approx(0.120, abs=5e-4)
    assert mid["ws"] == pytest.approx(1.527, abs=5e-4)
    assert ra == pytest.approx(32.2, abs=0.05)


def test_hargreaves_magnitude():
    # Hargreaves & Samani (1985): ET0 = 0.0023 Ra (T+17.8) sqrt(dT), Ra in mm/day.
    # Hand calculation for Ra = 40 MJ (16.32 mm), Tmax 32, Tmin 20:
    # 0.0023 * 16.32 * 43.8 * sqrt(12) = 5.70 mm/day
    ra, _ = extraterrestrial_radiation(26.0, 166)
    et0 = hargreaves_et0(32.0, 20.0, 26.0, 166)
    expected = 0.0023 * 0.408 * ra * (26.0 + 17.8) * np.sqrt(12.0)
    assert et0 == pytest.approx(expected)
    assert 5.0 < et0 < 6.5


def test_fao33_ky_maize_values():
    # Doorenbos & Kassam (1979) FAO-33, Table 24, maize
    ky = CropParams().ky
    assert ky[1:] == (0.4, 1.5, 0.5, 0.2)


def test_fao33_single_stage_deficit():
    ky = CropParams().ky
    etm = np.full(5, 100.0)
    eta = etm.copy()
    eta[2] = 80.0  # 20% ET deficit at flowering, Ky 1.5 -> 30% yield loss
    assert fao33_relative_yield(eta, etm, ky) == pytest.approx(0.70)
    eta = etm.copy()
    eta[4] = 50.0  # 50% deficit in ripening, Ky 0.2 -> 10% loss
    assert fao33_relative_yield(eta, etm, ky) == pytest.approx(0.90)


def test_fao33_multiplicative_and_floored():
    ky = CropParams().ky
    etm = np.full(5, 100.0)
    eta = np.array([100, 100, 80, 80, 100.0])
    assert fao33_relative_yield(eta, etm, ky) == pytest.approx(0.70 * 0.90)
    eta = np.array([100, 100, 0, 100, 100.0])  # total flowering failure
    assert fao33_relative_yield(eta, etm, ky) == 0.0


def test_crop_calendar():
    c = CropParams()
    assert c.stage_lengths().sum() == c.season_length == 120
    assert c.kc(0) == pytest.approx(0.30) and c.kc(70) == pytest.approx(1.20)
    assert c.kc(119) == pytest.approx(0.6 + (1.2 - 0.6) * 1 / 25, abs=1e-9)
    assert list(c.stage_of_day([0, 19, 20, 55, 72, 73, 107, 108, 119])) == [0, 0, 1, 2, 2, 3, 3, 4, 4]
