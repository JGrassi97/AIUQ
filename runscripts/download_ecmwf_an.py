"""
Initial conditions from the latest ECMWF IFS analysis available in ECMWF Open Data.

The IFS HRES state at step 0 of an operational run is the analysis interpolated to
0.25 deg. AIUQ models need two consecutive states, so the newest cycle T for which
both T-6h and T are published is selected (see the 'open_data' block of the card).
Only the GRIB messages needed are fetched, through HTTP range requests driven by the
.index files published next to each GRIB file. The ocean-wave fields (AIUQ-st v100) come
from the wave-model file of the same cycle (stream 'wave').

Data: (c) ECMWF, CC BY 4.0 - https://www.ecmwf.int/en/forecasts/datasets/open-data
"""

# Built-in/Generics
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Third party
import cfgrib
import numpy as np
import xarray as xr

# Local
from AIUQst_lib.cards import read_ic_card, read_std_version
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.support_data import add_fallback
from AIUQst_lib.variables import (define_ics_mappers, drop_optional_missing, optional_variables,
                                  reassign_long_names_units, variables_of_type)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

G = 9.80665                 # m s-2, converts geopotential height (gh) to geopotential (z)
RD = 287.05                 # J kg-1 K-1, dry air gas constant (hypsometric extrapolation)
SEAWATER_FREEZING = 271.35  # K, used to detect ice where the ocean model is undefined
SST_UNDER_ICE = 271.46      # K, value used by ERA5/OSTIA under sea ice
CYCLE_HOURS = 6
USER_AGENT = "AIUQ-ecmwf-an/1.0"
FALLBACK_REASON = "Not published in ECMWF Open Data"


# --------------------------------------------------------------------------------------
# Cycle selection
# --------------------------------------------------------------------------------------

def floor_cycle(moment: datetime) -> datetime:
    moment = moment.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return moment.replace(hour=moment.hour - moment.hour % CYCLE_HOURS)


def candidate_cycles(start_time: str, max_lookback_hours: int,
                     now: Optional[datetime] = None) -> List[datetime]:
    """Cycles to try, newest first: from START_TIME 18 UTC (or now) back to the lookback."""
    start = datetime.strptime(start_time, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    newest = min(start + timedelta(hours=24 - CYCLE_HOURS), floor_cycle(now))
    oldest = start - timedelta(hours=int(max_lookback_hours))
    cycles = []
    cycle = newest
    while cycle >= oldest:
        cycles.append(cycle)
        cycle -= timedelta(hours=CYCLE_HOURS)
    return cycles


def file_url(base: str, cycle: datetime, stream: str, open_data: dict, ext: str) -> str:
    return (f"{base.rstrip('/')}/{cycle:%Y%m%d}/{cycle:%H}z/{open_data['model']}/"
            f"{open_data['resol']}/{stream}/{cycle:%Y%m%d%H}0000-0h-{stream}-fc.{ext}")


def http_get(url: str, byte_range: Optional[Tuple[int, int]] = None,
             retries: int = 3, timeout: int = 60) -> Optional[bytes]:
    """GET with retries. Returns None when the object does not exist (403/404)."""
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            # S3 answers 403 for missing keys when listing is not allowed.
            if err.code in (403, 404):
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    return None


def parse_index(text: str) -> List[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_index(url: str, fetch: Callable[[str], Optional[bytes]]) -> Optional[List[dict]]:
    """Step-0 entries of an index file, None when it is not there or the source fails."""
    try:
        index = fetch(url)
    except Exception as err:  # noqa: BLE001 - any transport error means "try the next one"
        logging.warning(f"Cannot read {url}: {err}")
        return None
    if not index:
        return None
    entries = [e for e in parse_index(index.decode()) if str(e.get("step")) == "0"]
    return entries or None


def locate_cycle(cycle: datetime, open_data: dict,
                 fetch: Callable[[str], Optional[bytes]] = http_get,
                 wave: bool = False) -> Optional[dict]:
    """Find the step-0 file of a cycle in the configured sources/streams.

    A source that fails (outage, throttling) is skipped with a warning, so that the
    next mirror is tried instead of aborting the selection. With ``wave`` the wave-model
    file of the cycle (``open_data['wave_stream']``) must be there as well, from the same
    source: it is returned under 'wave_index'/'wave_grib'.
    """
    for base in open_data["sources"]:
        for stream in open_data["streams"]:
            entries = _read_index(file_url(base, cycle, stream, open_data, "index"), fetch)
            if not entries:
                continue
            found = {"cycle": cycle, "stream": stream, "source": base, "index": entries,
                     "grib": file_url(base, cycle, stream, open_data, "grib2")}
            if wave:
                wave_stream = open_data["wave_stream"]
                wave_entries = _read_index(file_url(base, cycle, wave_stream, open_data, "index"), fetch)
                if not wave_entries:
                    logging.warning(f"No {wave_stream} file for cycle {cycle:%Y-%m-%d %H}Z at {base}")
                    continue
                found.update(wave_index=wave_entries,
                             wave_grib=file_url(base, cycle, wave_stream, open_data, "grib2"))
            return found
    return None


def select_cycle(candidates: Iterable[datetime],
                 locate: Callable[[datetime], Optional[dict]]) -> Tuple[dict, dict]:
    """Newest cycle T such that T-6h and T are both published."""
    cache: Dict[datetime, Optional[dict]] = {}

    def cached(cycle):
        if cycle not in cache:
            cache[cycle] = locate(cycle)
        return cache[cycle]

    tried = []
    for cycle in candidates:
        tried.append(f"{cycle:%Y-%m-%d %H}Z")
        current = cached(cycle)
        if current is None:
            continue
        previous = cached(cycle - timedelta(hours=CYCLE_HOURS))
        if previous is not None:
            return previous, current
    raise RuntimeError("No ECMWF analysis pair (T-6h, T) found in Open Data for cycles: "
                       + ", ".join(tried))


# --------------------------------------------------------------------------------------
# Field selection and download
# --------------------------------------------------------------------------------------

ATMOSPHERE = "atm"      # key of the fields of the main (oper/scda) file in required_fields()


def required_fields(ic_card: dict, standard_variables: dict) -> Dict[str, Dict[Tuple[str, str], Optional[set]]]:
    """{file: {(levtype, param): levels or None}} for the standard variables in the card.

    ``file`` is ATMOSPHERE for the main step-0 file, or the stream of the variables that
    carry one in the card (the wave-model file, 'wave').
    """
    files: Dict[str, Dict[Tuple[str, str], Optional[set]]] = {ATMOSPHERE: {}}
    for code, spec in ic_card["variables"].items():
        if code not in standard_variables or spec.get("derived"):
            continue
        if spec.get("stream"):
            files.setdefault(spec["stream"], {})[(spec["levtype"], spec["param"])] = None
            continue
        fields = files[ATMOSPHERE]
        key = (spec["levtype"], spec["param"])
        if spec["levtype"] == "pl":
            fields[key] = {str(p) for p in ic_card["pressure_levels"]["values"]}
        elif spec["levtype"] == "sol":
            levels = fields.get(key) or set()
            levels.add(str(spec.get("soil_level", 1)))
            if spec["param"] == "sot":
                levels |= {"1", "2"}
            fields[key] = levels
        else:
            fields[key] = None
    fields = files[ATMOSPHERE]
    if any(spec.get("derived") for code, spec in ic_card["variables"].items() if code in standard_variables):
        fields[("sfc", "skt")] = None
    for levtype, params in ic_card.get("auxiliary", {}).items():
        for param in params:
            fields[(levtype, param)] = None
    return files


def select_entries(index: List[dict], fields: Dict[Tuple[str, str], Optional[set]],
                   optional: Iterable[Tuple[str, str]] = (("sfc", "sithick"),)) -> List[dict]:
    """Index entries for the requested fields. Missing required fields raise.

    Pressure-level fields only need the levels the file has (older cycles stop at
    50 hPa); geopotential falls back to geopotential height (before IFS 50r1).
    """
    available = {(e["levtype"], e["param"]) for e in index}
    if ("pl", "z") in fields and ("pl", "z") not in available and ("pl", "gh") in available:
        fields = dict(fields)
        fields[("pl", "gh")] = fields.pop(("pl", "z"))
    missing = [k for k in fields if k not in available and k not in set(optional)]
    if missing:
        raise RuntimeError(f"Fields not published in this cycle: {sorted(missing)}")
    selected = []
    for entry in index:
        key = (entry["levtype"], entry["param"])
        if key not in fields:
            continue
        levels = fields[key]
        if levels is not None and str(entry.get("levelist")) not in levels:
            continue
        selected.append(entry)
    pl_levels = {str(e["levelist"]) for e in selected if e["levtype"] == "pl"}
    for needed in ("50", "1000"):
        if any(k[0] == "pl" for k in fields) and needed not in pl_levels:
            raise RuntimeError(f"Pressure level {needed} hPa not published in this cycle")
    return sorted(selected, key=lambda e: int(e["_offset"]))


def merge_ranges(entries: List[dict]) -> List[Tuple[int, int]]:
    """Contiguous GRIB messages -> inclusive byte ranges (fewer HTTP requests)."""
    ranges: List[List[int]] = []
    for entry in entries:
        start, end = int(entry["_offset"]), int(entry["_offset"]) + int(entry["_length"]) - 1
        if ranges and ranges[-1][1] + 1 == start:
            ranges[-1][1] = end
        else:
            ranges.append([start, end])
    return [tuple(r) for r in ranges]


def download_fields(grib_url: str, entries: List[dict], target: str, workers: int = 8,
                    fetch: Callable = http_get) -> str:
    ranges = merge_ranges(entries)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        chunks = list(pool.map(lambda r: fetch(grib_url, r), ranges))
    for (start, end), chunk in zip(ranges, chunks):
        if chunk is None or len(chunk) != end - start + 1 or not chunk.startswith(b"GRIB"):
            raise RuntimeError(f"Incomplete download of bytes {start}-{end} from {grib_url}")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "wb") as f:
        for chunk in chunks:
            f.write(chunk)
    return target


# --------------------------------------------------------------------------------------
# Decoding and transformation to the AIUQ standard
# --------------------------------------------------------------------------------------

def open_cycle(path: str) -> xr.Dataset:
    """All hypercubes of one step-0 file merged, with a length-1 time dimension."""
    datasets = []
    for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
        ds = ds.drop_vars([c for c in ds.coords
                           if c not in {"time", "latitude", "longitude", "isobaricInhPa", "soilLayer"}])
        datasets.append(ds)
    dataset = xr.merge(datasets, join="exact", compat="no_conflicts")
    if "gh" in dataset and "z" not in dataset:
        dataset["z"] = (dataset["gh"] * G).assign_attrs(units="m**2 s**-2", long_name="Geopotential")
        dataset = dataset.drop_vars("gh")
    return dataset.expand_dims("time")


def derive_sea_surface(dataset: xr.Dataset) -> xr.Dataset:
    """Sea-ice cover and SST, which Open Data does not publish.

    ci: 1 where the ocean model has sea ice (sithick > 0); where it is undefined
        (inland water) where the skin is below the seawater freezing point.
    sst: skin temperature over open water, the freezing point under ice. Both are NaN
        over land, like ERA5.
    """
    sea = dataset["lsm"] < 0.5
    cold = dataset["skt"] < SEAWATER_FREEZING
    if "sithick" in dataset:
        ice = xr.where(dataset["sithick"].notnull(), dataset["sithick"] > 0, cold)
    else:
        ice = cold
    ci = xr.where(sea, ice.astype("float32"), np.nan)
    sst = xr.where(sea, xr.where(ice, SST_UNDER_ICE, np.maximum(dataset["skt"], SST_UNDER_ICE)), np.nan)
    dataset["ci"] = ci.astype("float32").assign_attrs(
        derivation="1 where sithick > 0 (skt < 271.35 K where sithick is undefined), sea points only")
    dataset["sst"] = sst.astype("float32").assign_attrs(
        derivation="skt over open sea, 271.46 K under sea ice, sea points only")
    return dataset


def normalize_soil(dataset: xr.Dataset, ic_card: dict) -> xr.Dataset:
    """Expose soil moisture layers under the standard's separate IDs; keep 2 sot layers."""
    for spec in ic_card["variables"].values():
        if "soil_level" in spec:
            dataset[spec["name"]] = dataset["vsw"].sel(soilLayer=spec["soil_level"], drop=True)
    dataset = dataset.drop_vars("vsw", errors="ignore")
    if "soilLayer" in dataset.dims:
        dataset = dataset.sel(soilLayer=[1, 2])
        dataset = dataset.assign_coords(soilLayer=dataset.soilLayer.astype("int64"))
    return dataset


# US Standard Atmosphere 1976 layers: (base pressure Pa, base temperature K, lapse rate dT/dh K/m)
_STD_ATMOSPHERE = [
    (22632.06, 216.65, 0.0),
    (5474.889, 216.65, 0.001),
    (868.0187, 228.65, 0.0028),
    (110.9063, 270.65, 0.0),
    (66.93887, 270.65, -0.0028),
    (3.956420, 214.65, -0.002),
]


def standard_atmosphere_temperature(p_hpa):
    """US Standard Atmosphere 1976 temperature (K) at pressure p (hPa), above 226 hPa."""
    p = np.atleast_1d(np.asarray(p_hpa, dtype="float64")) * 100.0
    if (p > _STD_ATMOSPHERE[0][0]).any():
        raise ValueError("Standard-atmosphere profile defined only above 226.32 hPa")
    temperature = np.empty_like(p)
    for i, value in enumerate(p):
        base_p, base_t, lapse = [layer for layer in _STD_ATMOSPHERE if value <= layer[0]][-1]
        temperature[i] = base_t * (value / base_p) ** (-RD * lapse / G) if lapse else base_t
    return temperature if np.ndim(p_hpa) else temperature[0]


def extrapolate_above(top: xr.Dataset, p_top: float, above: np.ndarray, dim: str) -> xr.Dataset:
    """Levels above the highest available one.

    t follows the standard-atmosphere shape anchored at the top level,
    t(p) = t(p_top) + T_std(p) - T_std(p_top); z is integrated hydrostatically with the
    layer-mean temperature; the other variables (u, v, q, w) are held constant.
    """
    levels = np.sort(above)[::-1]                       # from just above p_top upwards
    profile = standard_atmosphere_temperature(np.concatenate([[p_top], levels]))
    out = top.expand_dims({dim: levels}).copy()
    if "t" in top:
        t_levels = [top["t"] + (profile[k + 1] - profile[0]) for k in range(levels.size)]
        out["t"] = xr.concat(t_levels, dim=dim).assign_coords({dim: levels})
        if "z" in top:
            z_levels, z_below, t_below, p_below = [], top["z"], top["t"], p_top
            for k, p in enumerate(levels):
                z_below = z_below + RD * 0.5 * (t_below + t_levels[k]) * np.log(p_below / p)
                t_below, p_below = t_levels[k], p
                z_levels.append(z_below)
            out["z"] = xr.concat(z_levels, dim=dim).assign_coords({dim: levels})
    return out.sortby(dim)


def to_standard_levels(dataset: xr.Dataset, required: Iterable[float], dim: str = "level") -> xr.Dataset:
    """Interpolate linearly in log(p); extrapolate above the top level with
    ``extrapolate_above``. Source levels are returned unchanged."""
    pl_vars = [v for v in dataset.data_vars if dim in dataset[v].dims]
    if not pl_vars:
        return dataset
    source = dataset[pl_vars].sortby(dim)
    p_source = source[dim].values.astype("float64")
    target = np.asarray(sorted(float(p) for p in required))
    if (target > p_source.max()).any():
        raise ValueError(f"Cannot extrapolate below the lowest level {p_source.max()} hPa")
    inside = target[target >= p_source.min()]
    above = target[target < p_source.min()]

    logp = source.assign_coords({dim: np.log(p_source)})
    parts = []
    if above.size:
        top = source.sel({dim: p_source.min()}, drop=True)
        parts.append(extrapolate_above(top, float(p_source.min()), above, dim))
    parts.append(logp.interp({dim: np.log(inside)}).assign_coords({dim: inside}))
    parts = [part.transpose(*source[pl_vars[0]].dims) for part in parts]
    levels_ds = xr.concat(parts, dim=dim)
    for v in pl_vars:
        levels_ds[v] = levels_ds[v].transpose(*dataset[v].dims).astype(dataset[v].dtype)
        levels_ds[v].attrs = dict(dataset[v].attrs)
    if np.all(target == np.round(target)):
        levels_ds = levels_ds.assign_coords({dim: target.astype("int64")})
    levels_ds[dim].attrs = dict(dataset[dim].attrs)
    out = dataset.drop_vars(pl_vars + [dim]).merge(levels_ds)
    out.attrs["levels_extrapolated_hPa"] = [float(p) for p in above]
    if above.size:
        out.attrs["levels_extrapolation"] = ("t: anchored US Standard Atmosphere 1976 profile; "
                                             "z: hydrostatic; u, v, q, w: constant")
    return out


def prepare_initial_conditions(dataset: xr.Dataset, config: dict, ic_card: dict,
                               standard_dict: dict) -> xr.Dataset:
    standard_variables = standard_dict["variables"]["data"]
    names, rename, long_names, units, missing = define_ics_mappers(ic_card["variables"], standard_variables)

    dataset = dataset.assign_coords(longitude=dataset.longitude % 360).sortby("longitude")
    dataset = normalize_soil(dataset, ic_card)
    dataset = derive_sea_surface(dataset)

    # Optional variables (wave fields) may not have been downloaded: treat them as missing.
    absent = {code: standard_variables[code] for code, name in names.items() if name not in dataset}
    if absent:
        missing = {**(missing or {}), **absent}
        names = {code: name for code, name in names.items() if code not in absent}
        rename = {name: rename[name] for name in names.values()}
    selected = dataset[list(names.values())].rename(rename)
    selected = selected.rename({ic_card["pressure_levels"]["name"]: "level"})
    selected["level"].attrs["units"] = standard_dict["pressure_levels"]["units"]["valid"]
    selected = to_standard_levels(selected, standard_dict["pressure_levels"]["values"])
    selected = reassign_long_names_units(selected, long_names, units)

    for card_name, key, climatology in [("static", "STATIC_DATA", False),
                                        ("climatology", "CLIMATOLOGY_DATA", True)]:
        selected, missing = add_fallback(
            selected, missing, read_ic_card(config["HPCROOTDIR"], card_name),
            config[key], climatology, reason=FALLBACK_REASON,
        )
    missing = drop_optional_missing(missing, standard_variables)
    if missing:
        raise ValueError(f"No ECMWF Open Data/static/climatology source for variable IDs: {sorted(missing)}")
    if "soilLayer" in selected.dims:
        selected = selected.rename({"soilLayer": "SoilLevel"})

    # Masked by construction: sst/ci and the wave fields over land, soil fields over sea.
    sea_only = {"sst", "ci", "swvl1", "swvl2", "sot"} | variables_of_type(standard_variables, "wave")
    for name, da in selected.data_vars.items():
        if name in sea_only:
            continue
        if not bool(np.isfinite(da).all()):
            raise ValueError(f"Non-finite values in ECMWF analysis initial conditions: {name}")
    return selected.transpose("time", "level", ..., "latitude", "longitude")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def check_existing(output: str, ic_card: dict, standard: dict) -> bool:
    if not (os.path.isdir(output) and os.listdir(output)):
        return False
    with xr.open_zarr(output) as existing:
        variables = standard["variables"]["data"]
        optional = optional_variables(variables)
        expected = {v["short_name"] for code, v in variables.items() if code not in optional}
        if existing.attrs.get("ic_name") != ic_card["ic_name"] or not expected.issubset(existing.data_vars):
            raise ValueError(f"Existing ICs at {output} do not match the {ic_card['ic_name']} card; "
                             "use a new INI_DATA_PATH or explicitly remove the old ICs.")
        print(f"Output already exists at {output} (analysis {existing.attrs.get('analysis_time')}), "
              "skipping download.")
    return True


def main() -> None:
    args = parse_arguments()
    config = read_config(args.config)
    output = config["INI_DATA_PATH"]
    ic_card = read_ic_card(config["HPCROOTDIR"], config["IC_NAME"])
    standard = read_std_version(config["HPCROOTDIR"], config["STD_VERSION"])
    if check_existing(output, ic_card, standard):
        return

    open_data = ic_card["open_data"]
    files = required_fields(ic_card, standard["variables"]["data"])
    wave_stream = open_data.get("wave_stream")
    need_wave = bool(wave_stream and files.get(wave_stream))
    previous, current = select_cycle(
        candidate_cycles(config["START_TIME"], open_data["max_lookback_hours"]),
        lambda cycle: locate_cycle(cycle, open_data, wave=need_wave),
    )
    logging.info(f"Selected ECMWF analysis {current['cycle']:%Y-%m-%d %H}Z "
                 f"(with {previous['cycle']:%Y-%m-%d %H}Z) from {current['source']}")

    temp_dir = os.path.join(config.get("ICS_TEMP_DIR") or config["TEMP_DIR"], "ecmwf-an")
    cycles, grib_files = [], []
    for located in (previous, current):
        parts = []
        sources = [(files[ATMOSPHERE], located["index"], located["grib"], "")]
        if need_wave:
            sources.append((files[wave_stream], located["wave_index"], located["wave_grib"], f"-{wave_stream}"))
        for wanted, index, grib, suffix in sources:
            target = os.path.join(temp_dir, f"{located['cycle']:%Y%m%d%H}-0h{suffix}.grib2")
            entries = select_entries(index, wanted)
            logging.info(f"Downloading {len(entries)} fields from {grib}")
            download_fields(grib, entries, target, open_data.get("workers", 8))
            parts.append(open_cycle(target))
            grib_files.append(grib)
        cycles.append(xr.merge(parts, join="exact", compat="no_conflicts"))
    dataset = xr.concat(cycles, dim="time", join="exact", data_vars="all")

    selected = prepare_initial_conditions(dataset, config, ic_card, standard)
    selected.attrs.update(
        source="ECMWF Open Data, IFS HRES step 0 (analysis)",
        ic_name=ic_card["ic_name"],
        analysis_time=f"{current['cycle']:%Y-%m-%dT%H:%M}Z",
        previous_time=f"{previous['cycle']:%Y-%m-%dT%H:%M}Z",
        open_data_files=grib_files,
        fallback_source="static support data; ERA5 climatology (clwc, ciwc)",
        licence="CC BY 4.0, (c) ECMWF",
    )

    # A failed write must not appear to be a complete set of initial conditions.
    temporary = output + ".partial"
    shutil.rmtree(temporary, ignore_errors=True)
    selected.chunk({"time": 1}).to_zarr(temporary, mode="w", zarr_format=2)
    os.replace(temporary, output)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Wrote ECMWF analysis {selected.attrs['analysis_time']} initial conditions to {output}")


if __name__ == "__main__":
    main()
