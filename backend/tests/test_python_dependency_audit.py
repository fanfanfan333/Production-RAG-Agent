import pytest

from scripts.audit_python_dependencies import public_version


def test_only_official_cpu_suffix_uses_public_release_version():
    assert public_version("torch", "2.14.0+cpu") == "2.14.0"
    assert public_version("torchvision", "0.29.0+cpu") == "0.29.0"
    assert public_version("custom", "1.0+cpu") == "1.0+cpu"
    assert public_version("torch", "2.14.0+custom") == "2.14.0+custom"


@pytest.mark.parametrize("dependencies, expected", [
    ([{"name": "torch", "vulns": []}], 0),
    ([{"name": "torch", "vulns": [{"id": "CVE-test"}]}], 1),
    ([{"name": "torch", "vulns": [], "skip_reason": "unknown package"}], 1),
    ([], 1),
])
def test_audit_cannot_pass_with_skips_missing_packages_or_vulnerabilities(monkeypatch, tmp_path, dependencies, expected):
    import json
    from types import SimpleNamespace
    from scripts import audit_python_dependencies as audit
    output = tmp_path / "audit.json"
    monkeypatch.setattr(audit, "distributions", lambda: [SimpleNamespace(metadata={"Name": "torch"}, version="2.14.0+cpu")])

    def run(command, **kwargs):
        output.write_text(json.dumps({"dependencies": dependencies}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(audit.subprocess, "run", run)
    assert audit.main(SimpleNamespace(auditor_python="python", out=str(output))) == expected
