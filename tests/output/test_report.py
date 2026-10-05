from pathlib import Path

from factories import candidate, sure_candidate
from varhunt.config import Field, Run, Settings
from varhunt.output.report import (
    KNOWN,
    candidate_details,
    period_matches,
    read_results,
    write_results,
    write_summary,
    write_survey,
)


def make_run(tmp_path: Path) -> Run:
    run = Run(tmp_path, Field(100.0, 20.0, 0.2), 72, Settings())
    run.save()
    return run


def test_results_survive_the_csv(tmp_path):
    results = [
        candidate(gaia_scatter=6.4, gaia_scatter_reliable=False),
        candidate(tic=2, teff=5000.0, shared_with=1, snr=None, status=KNOWN),
        candidate(tic=3, gaia_scatter_reliable=True),
    ]
    write_results(tmp_path / "stars.csv", results)
    assert read_results(tmp_path / "stars.csv") == results


def test_period_matches_allow_half_and_double():
    assert period_matches(0.5, 0.5)
    assert period_matches(0.25, 0.5)
    assert period_matches(1.0, 0.5)
    assert not period_matches(0.4, 0.5)
    assert not period_matches(None, 0.5)


def test_summary_lists_candidates_with_vsx_details(tmp_path):
    run = make_run(tmp_path)
    known = candidate(tic=7, status=KNOWN, known_period=0.5695, known_separation=2.0)
    write_summary(run, [candidate(), known])
    text = run.summary_md.read_text()
    assert "### TIC 194990179" in text
    assert "Epoch of maximum: BJD 2460300.12000" in text
    assert "1 of 1 recovered" in text
    assert "not run yet" in text


def test_run_settings_round_trip(tmp_path):
    run = make_run(tmp_path)
    assert Run.load(tmp_path) == run


def test_sure_needs_every_piece_of_evidence():
    assert sure_candidate().sure
    assert not sure_candidate(confirmed_sectors="").sure
    assert not sure_candidate(catalog_hits="J/ApJS/258/16/tess-ebs").sure
    assert not sure_candidate(vet_verdict="ambiguous").sure
    assert not sure_candidate(vet_centroid_error_px=0.4).sure
    assert not sure_candidate(other_verdicts="43:no_signal;45:no_signal").sure
    assert not sure_candidate(survives_wider_detrending=False).sure
    assert not sure_candidate(other_verdicts="43:on_target;45:off_target").sure
    assert not sure_candidate(vet_offset_px=0.45).sure
    assert not sure_candidate(snr=6.5, confirmed_sectors="43;44;45").sure


def test_sure_needs_a_visible_light_range():
    assert not sure_candidate(light_max_ppt=0.06, light_min_ppt=-0.06).sure


def test_weaker_signal_is_sure_when_two_more_sectors_repeat_it():
    assert not sure_candidate(snr=8.0, confirmed_sectors="43").sure
    assert sure_candidate(snr=8.0, confirmed_sectors="43;45").sure


def test_sure_only_fears_unresolved_stars_at_the_signal_position():
    assert sure_candidate(blend_required_ppt=13.0).sure
    assert not sure_candidate(vet_companions="Gaia DR3 1", vet_companion_needs_ppt=105.0).sure
    assert sure_candidate(vet_companions="Gaia DR3 1", vet_companion_needs_ppt=2400.0).sure


def test_survey_report_puts_sure_candidates_first(tmp_path):
    run = make_run(tmp_path / "100.000+20.000")
    plain = candidate(tic=5, vet_verdict="on_target", vet_offset_px=0.2, snr=40.0)
    rejected = candidate(tic=6, vet_verdict="motion", vet_offset_px=4.0)
    write_survey(tmp_path / "survey.md", [(run, [plain, sure_candidate(), rejected])])
    text = (tmp_path / "survey.md").read_text()
    assert "2 candidates past the pixel check, 1 sure" in text
    assert text.index("### TIC 194990179") < text.index("### TIC 5")
    assert "TIC 6" not in text
    assert "- Run: 100.000+20.000" in text


def test_a_flat_topped_binary_without_a_box_eclipse_is_described_by_its_sinusoids():
    details = "\n".join(candidate_details(candidate(type_hint="EA", periods="0.2847500")))
    assert "Strongest sinusoid" in details
    assert "twice the strongest photometric period" in details


def test_a_period_equal_to_the_strongest_sinusoid_is_not_called_doubled():
    details = "\n".join(candidate_details(candidate()))
    assert "twice" not in details


def test_only_candidates_that_could_become_sure_are_worth_vetting():
    assert candidate().could_be_sure
    assert not candidate(snr=6.0).could_be_sure
    assert not candidate(light_max_ppt=0.3, light_min_ppt=-0.3).could_be_sure
    assert not candidate(status=KNOWN).could_be_sure


def test_a_secondary_eclipse_is_listed_with_its_magnitude():
    eclipsing = candidate(
        type_hint="EA", eclipse_depth_ppt=114.7, eclipse_snr=117.6, eclipse_secondary_ppt=86.0
    )
    details = "\n".join(candidate_details(eclipsing))
    assert "Secondary eclipse at phase 0.5: 86.00 ppt, 11.688 mag" in details


def test_a_joint_period_of_spots_comes_with_a_warning():
    spotted = candidate(joint_period=1.2928, joint_period_error=3e-6, joint_span=2578.0)
    assert "prefer the period of one sector" in "\n".join(candidate_details(spotted))
