"""Audit the installed runtime, including CPU wheel public versions, without resolving models."""
import argparse
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def public_version(name: str, version: str) -> str:
    # PyPI lists the public release; official PyTorch CPU wheels add a local suffix.
    return version.removesuffix("+cpu") if name.lower() in {"torch", "torchvision"} else version


def main(args) -> int:
    packages = sorted({f"{d.metadata['Name']}=={public_version(d.metadata['Name'], d.version)}" for d in distributions()})
    with tempfile.TemporaryDirectory() as directory:
        requirements = Path(directory) / "installed.txt"
        requirements.write_text("\n".join(packages), encoding="utf-8")
        result = subprocess.run([
            args.auditor_python, "-m", "pip_audit", "-r", str(requirements),
            "--no-deps", "--disable-pip", "--progress-spinner", "off",
            "--format", "json", "--output", args.out,
        ], check=False)
    if result.returncode:
        return result.returncode
    report = json.loads(Path(args.out).read_text(encoding="utf-8"))
    audited = report.get("dependencies", [])
    if len(audited) != len(packages) or any(d.get("skip_reason") or d.get("vulns") for d in audited):
        print("Dependency gate failed: incomplete coverage or unresolved vulnerabilities", file=sys.stderr)
        return 1
    print(f"Dependency gate passed: {len(audited)} installed packages audited")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor-python", default=sys.executable)
    parser.add_argument("--out", default="/tmp/python-dependency-audit.json")
    raise SystemExit(main(parser.parse_args()))
