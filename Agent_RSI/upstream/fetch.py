"""Download a source-only snapshot from a paper's repository and record its commit."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=180,
    )
    return result.stdout.strip()


def fetch(name: str, url: str, destination: Path, commit: str | None = None):
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError("Invalid source name")
    parsed = urlparse(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("Use a public HTTPS repository URL")
    if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("--commit must be a full Git SHA-1 commit")
    if commit is None:
        commit = git("ls-remote", "--exit-code", url, "HEAD").split()[0]
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("Could not resolve an exact SHA-1 commit")

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{name}-{commit[:12]}.zip"
    manifest = archive.with_suffix(".json")
    if archive.exists() or manifest.exists():
        raise FileExistsError(f"Snapshot already exists for {name} at {commit}")
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        folder = Path(temp)
        git("init", "--bare", str(folder / "repo"))
        repo = folder / "repo"
        git("fetch", "--depth=1", "--no-tags", url, commit, cwd=repo)
        resolved = git("rev-parse", "FETCH_HEAD", cwd=repo)
        if resolved.lower() != commit.lower():
            raise ValueError(f"Expected {commit}, fetched {resolved}")
        pending = folder / "source.zip"
        git("archive", "--format=zip", f"--prefix={name}/", f"--output={pending.resolve()}",
            resolved, cwd=repo)
        with ZipFile(pending) as source:
            if source.testzip() is not None:
                raise ValueError("Source archive did not pass checksum verification")
            notices = [
                path for path in source.namelist()
                if Path(path).name.upper().split(".")[0] in {"LICENSE", "LICENCE", "NOTICE", "COPYING"}
            ]
        digest = hashlib.sha256()
        with pending.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        record = {
            "source": name, "repository": url, "commit": resolved,
            "archive": archive.name, "sha256": checksum, "license_notice_files": notices,
            "source_license_review": "Review these notices before copying or redistributing source.",
        }
        pending.replace(archive)
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", choices=SOURCES)
    p.add_argument("--list", action="store_true")
    p.add_argument("--repo", help="Repository URL obtained from the paper or author")
    p.add_argument("--commit", help="Full Git commit SHA")
    p.add_argument("--out", type=Path, default=HERE / "snapshots")
    args = p.parse_args()
    if args.list:
        for key, source in SOURCES.items():
            print(key, source["repository"] or source["paper"])
        return
    if not args.source:
        p.error("Specify a source or --list")
    url = args.repo or SOURCES[args.source]["repository"]
    if url is None:
        p.error("Use --repo with the repository URL confirmed from the paper or project page")
    try:
        print(fetch(args.source, url, args.out, args.commit))
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Git download failed: {e.stderr.strip()}") from e
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
