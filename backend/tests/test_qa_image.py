import re
from pathlib import Path


def test_qa_image_uses_ci_tool_versions_without_changing_production_image():
    root = Path(__file__).parents[1]
    image = (root / "Dockerfile.qa").read_text(encoding="utf-8")
    requirements = (root / "requirements-ci.txt").read_text(encoding="utf-8")
    for package in ("pytest", "pip-audit"):
        pinned = re.search(r"(?m)^" + re.escape(package) + r"==[^\s]+", requirements).group()
        assert pinned in image
    assert "Dockerfile.qa" not in (root / "docker-compose.yml").read_text(encoding="utf-8")
