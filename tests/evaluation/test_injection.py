from factories import make_light_curve, write_qlp
from varhunt.config import Settings
from varhunt.evaluation.injection import Injection, Trial, completeness_table, plan, run_trial

SETTINGS = Settings()


def test_plan_is_reproducible_and_stays_in_range(tmp_path):
    paths = [tmp_path / "a.fits", tmp_path / "b.fits"]
    first = plan(paths, 50, SETTINGS, seed=3)
    assert first == plan(paths, 50, SETTINGS, seed=3)
    assert all(0.01 <= i.period <= SETTINGS.max_period for i in first)
    assert all(1e-4 <= i.amplitude <= 3e-2 for i in first)


def test_strong_injection_is_recovered_and_a_tiny_one_is_not(tmp_path):
    path = write_qlp(tmp_path / "quiet.fits", make_light_curve(noise=1e-3))
    assert run_trial(Injection(path, 0.37, 3e-3, 0.4), SETTINGS).recovered
    assert not run_trial(Injection(path, 0.37, 5e-5, 0.4), SETTINGS).recovered


def test_completeness_table_counts_per_cell():
    trials = [Trial(11.5, 0.3, 0.005, True), Trial(11.2, 0.4, 0.004, False)]
    table = completeness_table(trials)
    assert "| 3 to 10 |" in table
    assert "50% (2)" in table
