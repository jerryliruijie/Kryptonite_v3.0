"""Resolve historical record paths after moving MLP files out of member_c."""
import hashlib
from pathlib import Path
from common import ROOT


def recorded_path(relative, checksum=None):
    """Keep old records unchanged; locate their files in the current layout.

    When a historical source checksum is supplied, use the exact saved source
    snapshot if the maintained source has changed. Never rewrite old hashes.
    """
    relative = str(relative)
    translated = relative
    for old, new in (("member_c/results/", "results/mlp/"),
                     ("member_c/configs/", "results/mlp/provenance/"),
                     ("member_c/report/", "docs/mlp/")):
        if translated.startswith(old):
            translated = new + translated[len(old):]
            break
    else:
        if relative.startswith("member_c/") and relative.endswith(".py"):
            translated = "mlp_" + Path(relative).name
    candidates = [ROOT / translated]
    if checksum is not None:
        if relative.startswith("member_c/configs/"):
            candidates.extend(sorted((ROOT / "results/mlp").glob("*/config.json")))
        candidates.extend(sorted((ROOT / "results/mlp").glob("*/source/" + relative)))
        candidates.append(ROOT / "results/strict_87ec2ce_20261118/source" / relative)
    for path in candidates:
        if path.is_file() and (checksum is None or hashlib.sha256(path.read_bytes()).hexdigest() == checksum):
            return path
    raise FileNotFoundError(f"No matching recorded artifact: {relative}")
