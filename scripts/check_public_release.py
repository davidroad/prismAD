#!/usr/bin/env python3
"""Inspect a code release or Git index without reading controlled input files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

FIGURES = {"assets/figure1a.png": "36fc09c9f152a77b827d7622901342ef51fe9fc7feded58d8bd98305dfaac837"}
JSON_FILES = {"config/input.example.json", "config/protocol.json",
              "config/statistics_protocol.json", "environments/render_runtime.json"}
TEXT_SUFFIXES = {".py", ".r", ".md", ".txt", ".cff", ".toml", ".yml", ".yaml"}
DATA_SUFFIX = re.compile(
    r"\.(?:csv|tsv|txt\.gz|vcf(?:\.gz)?|bcf|bed|bim|fam|pgen|pvar(?:\.zst)?|"
    r"psam|bgen|sample|bam|cram|fastq(?:\.gz)?|fq(?:\.gz)?|parquet|feather|"
    r"npy|npz|pkl|pickle|pt|pth|ckpt|h5|hdf5|h5ad|rds|rdata|xlsx|xls|"
    r"sav|dta|zip|tar|gz|7z)$", re.I)
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub credential": re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "AWS access key": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "participant-like identifier": re.compile(
        r"\b(?:NACC[0-9]{4,}|[0-9]{3}_S_[0-9]{4,})\b"),
    "local absolute path": re.compile(r"/(?:home|root|working)/"),
    "VCF payload": re.compile(r"(?:^|\n)(?:##fileformat=VCF|#CHROM\tPOS\tID)"),
    "LFS pointer": re.compile(r"(?m)^version https://git-lfs.github.com/spec/v1$"),
}
PRIVATE_KEYS = {"subject_id", "participant_id", "patient_id", "sample_id",
                "participants", "genotypes", "dosages", "individual_prs",
                "fold_assignments", "IID", "FID"}

def inspect_json(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("participant-shaped JSON keys")
        for child in value.values():
            inspect_json(child)
    elif isinstance(value, list):
        for child in value:
            inspect_json(child)

def inspect(path, data):
    parts = Path(path).parts
    if any(part.lower() in {"data", "sourcedata", "private", "private_local",
                            "models", "checkpoints", "restricted_local",
                            "local_restricted"} for part in parts):
        raise ValueError("private-data directory")
    if DATA_SUFFIX.search(path):
        raise ValueError("data table, genomic container or fitted artifact")
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("file requires manual size/content review")
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        if FIGURES.get(path) != hashlib.sha256(data).hexdigest():
            raise ValueError("image requires manual content/metadata review")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("unapproved binary payload")
    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            raise ValueError(name)
    if suffix == ".json":
        if path not in JSON_FILES:
            raise ValueError("unapproved JSON file")
        inspect_json(json.loads(text))
    elif suffix not in TEXT_SUFFIXES and path != ".gitignore":
        raise ValueError("unapproved file format")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.staged:
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "-z"], cwd=root).decode().split("\0")
        paths = [name for name in names if name]
    else:
        paths = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                       if p.is_file() and not {".git", "__pycache__",
                                              ".pytest_cache"}.intersection(p.parts)
                       and not p.name.endswith((".pyc", ".pyo")))
    findings = []
    for path in paths:
        try:
            if args.staged:
                data = subprocess.check_output(["git", "show", ":" + path], cwd=root)
            else:
                data = (root / path).read_bytes()
            inspect(path, data)
        except (ValueError, subprocess.CalledProcessError) as exc:
            findings.append({"path": path, "reason": str(exc)})
    print(json.dumps({"status": "FAIL" if findings else "PASS",
                      "scope": "Git index" if args.staged else "release directory",
                      "files": len(paths), "findings": findings}, indent=2))
    return 1 if findings else 0

if __name__ == "__main__":
    sys.exit(main())
