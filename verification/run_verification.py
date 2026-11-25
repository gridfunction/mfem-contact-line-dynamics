"""Run selected 2D cases using the published verification manifest."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd(),
                        help="Root of the built MFEM contact-line solver")
    parser.add_argument("--output", type=Path, default=Path("runs"),
                        help="Directory for the new case output")
    parser.add_argument("--include", default=".*", help="Regular expression for case names")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent single-rank runs")
    parser.add_argument("--list", action="store_true", help="List selected cases and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and exit")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")

    manifest = json.loads(Path(__file__).with_name("run_manifest.json").read_text())
    case_pattern = re.compile(args.include)
    cases = [case for case in manifest["verification_runs"] if case_pattern.search(case["case"])]
    if not cases:
        parser.error("No case names matched --include")
    if args.list:
        print(*(case["case"] for case in cases), sep="\n")
        return 0

    source = args.source.resolve()
    output = args.output.resolve()
    commands = {}
    settings = {}
    for case in cases:
        name = case["case"]
        command = [str(source / case["executable"]), *case["arguments"]]
        command[command.index("-out") + 1] = str(output / name)
        commands[name] = command
        settings[name] = {"name": name, **case["settings"]}
    if args.dry_run:
        for command in commands.values():
            print(shlex.join(command))
        return 0
    for case, command in commands.items():
        if not Path(command[0]).is_file():
            parser.error(f"Executable is missing. Build {command[0]} first.")
        if (output / case).exists():
            parser.error(f"Output exists for {case}. Choose a new --output directory.")

    overrides = manifest["verification_environment_overrides"]
    env = os.environ | overrides
    executable_digests = {
        executable: sha256(Path(executable).read_bytes()).hexdigest()
        for executable in {command[0] for command in commands.values()}
    }

    def run(case):
        directory = output / case
        directory.mkdir(parents=True)
        command = commands[case]
        description = directory / "run.json"
        metadata = {
            "case": settings[case],
            "command": command,
            "working_directory": str(source),
            "environment_overrides": overrides,
            "executable_sha256": executable_digests[command[0]],
        }
        description.write_text(json.dumps(metadata, indent=2) + "\n")
        began = time.monotonic()
        with (directory / "solver.log").open("w") as log:
            result = subprocess.run(
                command, cwd=source, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        metadata.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - began)
        completed = description.with_suffix(".json.tmp")
        completed.write_text(json.dumps(metadata, indent=2) + "\n")
        completed.replace(description)
        print(f"{case} exited with status {result.returncode}", flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, commands))
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
