"""
Compress AIUQ output files that already exist, in place.

usage: compress_outputs.py PATH [PATH ...] --scheme mixed [--keepbits 12] [--chunks point] [--complevel 1]
                           [--custom "t2m: {step: 0.5, offset: 273.15}" ...] [--procs 8] [--dry] [--force]

PATH is a NetCDF file, an experiment folder (its ``outputs/<variable>/<member>/out-*.nc`` are taken) or any
folder that holds such files. Restart checkpoints (``outputs/checkpoints``) and the intermediate files of a
running job (``_temp.nc``, ``_grid.nc``) are never touched.

Each file is rewritten beside the original, read back and compared with it (same missing values, same
coordinates, error within what the scheme allows) and only then renamed over it. Running the command again
does nothing: files already stored as asked are skipped, also those compressed by the earlier scripts. A
file stored with less precision than asked is left alone, because a rewrite cannot bring the precision back
(``--force`` rewrites it anyway).

The schemes are those of ``EXPERIMENT.OUT_COMPRESSION``: none, lossless, bits, integer, mixed, custom
(``AIUQst_lib/compression.py``). The steps of integer and mixed come from the AIUQ-st standard.
"""

# Built-in/Generics
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

# Third party
import yaml

# Local
from AIUQst_lib.compression import PRESETS, CompressionError, Scheme, compress_file

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_SUFFIXES = ("_temp.nc", "_grid.nc")


def output_files(path: str) -> list:
    """The output files below ``path`` (or ``path`` itself), without checkpoints and intermediate files."""
    if os.path.isfile(path):
        found = [path]
    else:
        root = os.path.join(path, "outputs") if os.path.isdir(os.path.join(path, "outputs")) else path
        found = glob.glob(os.path.join(root, "**", "out-*.nc"), recursive=True)
    keep = []
    for name in sorted(found):
        parts = os.path.normpath(name).split(os.sep)
        if "checkpoints" in parts or name.endswith(SKIP_SUFFIXES):
            continue
        keep.append(name)
    return keep


def standard_specs(folder: str) -> dict:
    """{short name: definition} from variables.yml and outputs.yml of an AIUQ-st folder."""
    specs = {}
    for name in ("variables.yml", "outputs.yml"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path) as handle:
                for spec in ((yaml.safe_load(handle) or {}).get("data") or {}).values():
                    specs[str(spec["short_name"])] = spec
    if not specs:
        raise SystemExit(f"no AIUQ-st definitions in {folder} (--standard)")
    return specs


def _one(job):
    path, options, specs, force, dry = job
    try:
        return (path,) + compress_file(path, Scheme(**options), specs, force=force, dry=dry)
    except Exception as error:   # noqa: BLE001  one bad file must not stop the others
        return path, f"FAILED {error}", 0, 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compress existing AIUQ output files in place.")
    parser.add_argument("paths", nargs="+", help="files, experiment folders or folders with output files")
    parser.add_argument("--scheme", required=True, choices=PRESETS)
    parser.add_argument("--keepbits", type=int)
    parser.add_argument("--chunks", help="point | map | T,LAT,LON")
    parser.add_argument("--complevel", type=int)
    parser.add_argument("--custom", action="append", default=[], metavar="'VAR: {...}'",
                        help="a rule of the custom scheme; repeat for each variable and for 'default'")
    parser.add_argument("--standard", default=os.path.join(HERE, "..", "conf", "AIUQ-st", "v100"),
                        help="folder of the AIUQ-st standard (variables.yml, outputs.yml)")
    parser.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dry", action="store_true", help="write and verify, but do not replace the files")
    parser.add_argument("--force", action="store_true", help="also rewrite files stored with less precision")
    args = parser.parse_args(argv)

    chunks = args.chunks
    if chunks and "," in chunks:
        chunks = [part.strip() for part in chunks.split(",")]
    options = dict(preset=args.scheme, keepbits=args.keepbits, chunks=chunks, complevel=args.complevel,
                   custom=args.custom or None)
    try:
        Scheme(**options)
    except CompressionError as error:
        raise SystemExit(f"compress_outputs: {error}")
    specs = standard_specs(args.standard)
    files = [name for path in args.paths for name in output_files(path)]
    if not files:
        raise SystemExit("compress_outputs: no output file found")
    print(f"{len(files)} files, scheme {args.scheme}, {args.procs} processes{', dry run' if args.dry else ''}", flush=True)

    started = time.time()
    counts, before, after, failed = {}, 0, 0, 0
    jobs = [(name, options, specs, args.force, args.dry) for name in files]
    with Pool(max(1, args.procs)) as pool:
        for done, (path, status, size, new_size) in enumerate(pool.imap_unordered(_one, jobs), 1):
            key = status.split()[0]
            counts[key] = counts.get(key, 0) + 1
            before, after = before + size, after + new_size
            if key == "FAILED":
                failed += 1
                print(f"{status}: {path}", flush=True)
            elif key == "kept":
                print(f"kept (stored with less precision than {args.scheme}): {path}", flush=True)
            if done % 50 == 0 or done == len(files):
                print(f"[{done}/{len(files)}] {time.time() - started:6.0f} s  {before / 1e9:8.2f} GB -> {after / 1e9:8.2f} GB",
                      flush=True)
    summary = ", ".join(f"{count} {key}" for key, count in sorted(counts.items()))
    print(f"done: {summary}; {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB in {time.time() - started:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
