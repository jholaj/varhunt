"""A VSX New Star Form draft, field by field in the order of the form."""

from dataclasses import dataclass
from pathlib import Path

from varhunt.sources.catalogs import Identity

REVISION_COMMENT = "Type, range and elements from TESS FFI photometry (TESS-SPOC and QLP)."


@dataclass(frozen=True)
class Attachment:
    path: Path
    title: str
    description: str


@dataclass(frozen=True)
class Submission:
    """Everything the VSX form asks for, plus the evidence a human should read before sending."""

    identity: Identity
    variable_type: str
    amplitude: float
    period: float
    period_error: float | None
    epoch_hjd: float
    epoch_bjd: float
    epoch_kind: str
    duration_percent: float | None
    discoverer: str
    remarks: str
    attachments: tuple[Attachment, ...]
    evidence: tuple[str, ...]


def write_draft(path: Path, submission: Submission) -> None:
    identity = submission.identity
    position = identity.position.to_string("hmsdms", sep=" ", precision=3)
    duration = (
        f"{submission.duration_percent:.2f}" if submission.duration_percent is not None else ""
    )
    error = (
        f"{submission.period_error:.7f} d (no field, see the remarks)"
        if submission.period_error is not None
        else "not known"
    )
    rows = [
        ("Novice discovery", "leave unchecked, the data come from a survey"),
        ("Position (J2000.0)", f"{position}, then click Spot check"),
        ("Primary name", f"{identity.name}, then click Check name"),
        ("Other names", "<br>".join(identity.aliases)),
        ("Standard variable type", submission.variable_type),
        ("Maximum magnitude", f"{identity.v_mag:.2f}, passband V ({identity.v_source})"),
        ("Minimum magnitude", f"{submission.amplitude:.4f}, passband TESS, Minimum is amplitude"),
        ("Period", f"{submission.period:.7f} days"),
        ("Period error", error),
        ("Epoch", f"{submission.epoch_hjd:.4f} HJD, time of {submission.epoch_kind}"),
        ("Rise/eclipse duration", duration),
        ("Discoverer", submission.discoverer),
        ("Remarks", submission.remarks),
        ("Revision comment", REVISION_COMMENT),
        ("Draft only", "check it for the first save"),
    ]
    lines = [
        f"# VSX draft: {identity.name}",
        "",
        "Nothing has been sent. Check every field, then fill in the New Star Form at "
        "vsx.aavso.org yourself.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        *(f"| {field} | {value} |" for field, value in rows),
        "",
        "## Supporting files",
        "",
    ]
    for number, attachment in enumerate(submission.attachments, start=1):
        lines += [
            f"{number}. `{attachment.path.name}`",
            f"   - Title: {attachment.title}",
            f"   - Description: {attachment.description}",
        ]
    lines += ["", "## Evidence to read before sending", ""]
    lines += [f"- {line}" for line in submission.evidence]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
