from factories import candidate, sure_candidate
from varhunt.config import Field, Run, Settings
from varhunt.output.report import write_results
from varhunt.workflows.survey import sure


def make_run(root, results):
    run = Run(root, Field(100.0, 20.0, 0.2), 72, Settings())
    run.save()
    write_results(run.stars_csv, results)
    return run


def test_sure_candidates_are_gathered_from_runs_and_surveys(tmp_path):
    make_run(tmp_path / "demo", [sure_candidate(tic=1, snr=20.0), candidate(tic=2)])
    make_run(tmp_path / "survey" / "field", [sure_candidate(tic=3, snr=40.0)])
    out = tmp_path / "sure.md"
    found = sure([tmp_path / "demo", tmp_path / "survey"], out)
    assert [result.tic for _, result in found] == [3, 1]
    text = out.read_text()
    assert "2 sure candidates in 2 runs" in text
    assert "### TIC 3" in text
    assert "TIC 2" not in text
