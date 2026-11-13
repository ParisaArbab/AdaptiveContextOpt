"""Portable input identities for the fixed experiment, separate from host paths."""
from dataclasses import replace
from pathlib import PurePosixPath, PureWindowsPath

from rave_fl.eval.schema import CompatibilityError, fingerprint
from rave_fl.graft.graph import build_graph


def absolute(value):
    if not isinstance(value, str):
        raise CompatibilityError("Evidence and checkout locators must be strings")
    return bool(value) and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute())


def portable_evidence(evidence):
    if not absolute(evidence.source):
        return evidence
    # The local locator remains in the saved preparation. The graph identifies
    # the observed artifact by content and kind, which do not depend on its mount.
    return replace(evidence, source=f"artifact:sha256:{evidence.sha256}")


def fixed_graph(instance, entities):
    for locator in (instance.checkout, instance.evidence.source):
        if absolute(locator) and locator.rstrip("/\\") and locator.rstrip("/\\") in instance.evidence.text:
            raise CompatibilityError(
                "Failure text embeds a local resource path. Review the evidence representation before freezing a cohort."
            )
    return build_graph(replace(instance, evidence=portable_evidence(instance.evidence)), entities)


def portable_preparation_sha256(prepared, instance):
    online = dict(prepared["online"])
    for key in ("checkout", "candidate_file", "flexfl_repo"):
        online.pop(key)
    online["evidence"] = {**online["evidence"], "source": portable_evidence(instance.evidence).source}
    return fingerprint({**prepared, "online": online})
