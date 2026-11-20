"""
Initial conditions from the EC-Earth3 CMIP6 simulations, read from ESGF over OPeNDAP.

AIUQ models need two states, START_TIME 00 and 06 UTC, on the 0.25 deg grid and the 37 standard
pressure levels. EC-Earth3 publishes (member r4i1p1f1, see conf/cards/ics/ec-earth3_card.yml):

* instantaneous 6-hourly upper air on 7 levels only (6hrPlevPt), plus psl and soil temperature;
* instantaneous 3-hourly near-surface fields (3hr: tas, huss, uas, vas);
* 6-hourly means (6hrPlev: wap on 4 levels, 100 m wind) and daily means (Eday upper air on 19
  levels up to 1 hPa; day: cloud cover, snow, soil moisture, surface temperature; sea ice, SST).

Upper-air profiles are the daily means at 00/06 UTC corrected by the 6-hourly departure from them
(``anchor_profile``); interval means are interpolated in time to 00/06 UTC; the variables not
published are derived (sp, d2m, tcwv, tcw, skt, sot, cloud layers) or taken from the static file
and the ERA5 climatology. Only the time steps needed are transferred.

Data: EC-Earth Consortium, CMIP6, CC BY 4.0 - https://wcrp-cmip.github.io/CMIP6_CVs/
"""

# Built-in/Generics
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Third party
import numpy as np
import xarray as xr

# Local
from AIUQst_lib.cards import read_ic_card, read_std_version
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.pressure_levels import G, RD, to_standard_levels
from AIUQst_lib.regrid import regrid_latlon
from AIUQst_lib.support_data import add_fallback, align_support_grid
from AIUQst_lib.variables import (define_ics_mappers, drop_optional_missing, optional_variables,
                                  variables_of_type)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

USER_AGENT = "AIUQ-ec-earth3/1.0"
FALLBACK_REASON = "Not published by EC-Earth3 (CMIP6)"
IC_HOURS = (0, 6)                       # the two states, hours after START_TIME
LAPSE_RATE = 0.0065                     # K m-1, reduction of msl to the surface
ICE_SURFACE_MAX = 273.15                # K, cap of the sea-ice skin temperature
TARGET_LATITUDES = np.linspace(90.0, -90.0, 721)
TARGET_LONGITUDES = np.arange(1440) * 0.25


class NotPublished(LookupError):
    """No file of the requested field covers the requested time window."""


# --------------------------------------------------------------------------------------
# Experiment and times
# --------------------------------------------------------------------------------------

def _setting(config: dict, key: str) -> Optional[str]:
    """A config value, or None when Autosubmit left it empty or unresolved (%VAR%)."""
    value = str(config.get(key) or "").strip()
    return None if not value or value.startswith("%") else value


def experiment_for(start_time: str, esgf: dict, override: Optional[str] = None) -> str:
    """CMIP6 experiment of START_TIME: 'historical' up to esgf['historical_until'], then the scenario."""
    if override:
        return override
    until = datetime.strptime(str(esgf["historical_until"]), "%Y-%m-%d")
    return "historical" if datetime.strptime(start_time, "%Y-%m-%d") <= until else esgf["scenario"]


def ic_times(start_time: str) -> np.ndarray:
    start = np.datetime64(start_time, "ns")
    return start + np.array(IC_HOURS) * np.timedelta64(1, "h")


def time_window(kind: str, times: np.ndarray) -> Tuple[np.datetime64, np.datetime64]:
    """Time range to read: the instants themselves ('point'), or one day around them ('mean')."""
    pad = np.timedelta64(0, "h") if kind == "point" else np.timedelta64(24, "h")
    return times.min() - pad, times.max() + pad


# --------------------------------------------------------------------------------------
# ESGF search and OPeNDAP access
# --------------------------------------------------------------------------------------

def http_json(url: str, timeout: int = 120, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            if 400 <= err.code < 500 and err.code != 429:
                raise
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Unreachable: {url}")


def search_files(esgf: dict, experiment: str, table: str, variable: str,
                 fetch: Callable[[str], dict] = None) -> List[dict]:
    """File records of one CMIP6 field (all replicas, latest version), from the first index that answers."""
    fetch = fetch or (lambda url: http_json(url, esgf.get("timeout", 120), esgf.get("retries", 3)))
    query = {"type": "File", "project": "CMIP6", "source_id": esgf["source_id"],
             "experiment_id": experiment, "member_id": esgf["member"], "table_id": table,
             "variable_id": variable, "latest": "true", "limit": 500,
             "format": "application/solr+json"}
    errors = []
    for index in esgf["indexes"]:
        docs, offset = [], 0
        try:
            while True:
                answer = fetch(index + "?" + urllib.parse.urlencode({**query, "offset": offset}))
                page = answer["response"]["docs"]
                docs.extend(page)
                offset += len(page)
                if not page or offset >= int(answer["response"]["numFound"]):
                    return docs
        except Exception as err:  # noqa: BLE001 - any failure means "try the next index"
            logging.warning(f"ESGF index {index} failed for {table}/{variable}: {err}")
            errors.append(f"{index}: {err}")
    raise RuntimeError(f"No ESGF index answered for {experiment} {table}/{variable}: {errors}")


_PERIOD = re.compile(r"_(\d{6,12})-(\d{6,12})\.nc$")


def _parse_stamp(stamp: str, end: bool) -> np.datetime64:
    fmt = {6: "%Y%m", 8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M"}[len(stamp)]
    moment = datetime.strptime(stamp, fmt)
    if end and len(stamp) == 8:
        moment += timedelta(days=1) - timedelta(seconds=1)      # a daily file ends at 23:59:59
    if end and len(stamp) == 6:
        moment = (moment.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(seconds=1)
    return np.datetime64(moment, "ns")


def file_period(title: str) -> Optional[Tuple[np.datetime64, np.datetime64]]:
    """(first, last) time covered by a CMIP6 file name, None for time-independent files."""
    match = _PERIOD.search(title)
    if not match:
        return None
    return _parse_stamp(match.group(1), end=False), _parse_stamp(match.group(2), end=True)


def opendap_candidates(docs: Iterable[dict], window: Tuple[np.datetime64, np.datetime64],
                       preferred_nodes: List[str]) -> List[Tuple[str, List[str]]]:
    """[(file name, [OPeNDAP URLs, preferred replica first])] of the files overlapping ``window``.

    Only the latest version of each file is kept; replicas without an OPeNDAP endpoint are skipped.
    """
    rank = {node: i for i, node in enumerate(preferred_nodes)}
    by_title: Dict[str, List[dict]] = {}
    for doc in docs:
        period = file_period(doc["title"])
        if period and (period[1] < window[0] or period[0] > window[1]):
            continue
        by_title.setdefault(doc["title"], []).append(doc)
    out = []
    for title in sorted(by_title):
        replicas = by_title[title]
        newest = max(str(d.get("version", "")) for d in replicas)
        replicas = sorted((d for d in replicas if str(d.get("version", "")) == newest),
                          key=lambda d: rank.get(d.get("data_node"), len(rank)))
        urls = []
        for doc in replicas:
            for entry in doc.get("url", []):
                url, _, service = entry.split("|")
                if service == "OPENDAP":
                    urls.append(url[:-len(".html")] if url.endswith(".html") else url)
        if urls:
            out.append((title, urls))
        else:
            logging.warning(f"No OPeNDAP endpoint for {title}")
    return out


def _standardize(da: xr.DataArray) -> xr.DataArray:
    """CMIP6 names -> AIUQ names: lat/lon -> latitude/longitude, plev (Pa) -> level (hPa)."""
    da = da.rename({k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in da.dims})
    if "plev" in da.dims:
        da = da.assign_coords(plev=da.plev.values / 100.0).rename(plev="level")
    keep = set(da.dims) | ({"latitude", "longitude"} if "latitude" in da.coords else set())
    da = da.drop_vars([c for c in da.coords if c not in keep])
    return da


def open_window(url: str, variable: str, window: Tuple[np.datetime64, np.datetime64],
                opener: Callable = xr.open_dataset) -> xr.DataArray:
    with opener(url, decode_timedelta=False) as ds:
        da = ds[variable]
        if "time" in da.dims:
            da = da.sel(time=slice(window[0], window[1]))
        return _standardize(da.load())


def read_field(esgf: dict, experiment: str, table: str, variable: str,
               window: Tuple[np.datetime64, np.datetime64],
               search: Callable = None, opener: Callable = xr.open_dataset) -> Tuple[xr.DataArray, List[str]]:
    """One CMIP6 field over ``window`` (all files it spans) and the URLs read.

    Raises NotPublished when no file of the field overlaps the window.
    """
    search = search or search_files
    docs = search(esgf, experiment, table, variable)
    files = opendap_candidates(docs, window, esgf.get("preferred_nodes", []))
    if not files:
        raise NotPublished(f"EC-Earth3 {experiment} {esgf['member']} {table}/{variable}: "
                           f"no file covers {window[0]} - {window[1]}")
    parts, used = [], []
    for title, urls in files:
        errors = []
        for url in urls:
            try:
                parts.append(open_window(url, variable, window, opener))
                used.append(url)
                break
            except Exception as err:  # noqa: BLE001 - try the next replica
                logging.warning(f"Cannot read {url}: {err}")
                errors.append(f"{url}: {err}")
        else:
            raise RuntimeError(f"No replica of {title} could be read: {errors}")
    parts = [p for p in parts if "time" not in p.dims or p.sizes["time"]]
    if not parts:
        raise NotPublished(f"{table}/{variable}: no time step in {window[0]} - {window[1]}")
    da = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time").sortby("time")
    if "time" in da.dims:
        _, unique = np.unique(da.time.values, return_index=True)
        da = da.isel(time=np.sort(unique))
    logging.info(f"Read {table}/{variable} {dict(da.sizes)} from {len(used)} file(s)")
    return da, used


def at_instants(da: xr.DataArray, times: np.ndarray, kind: str) -> xr.DataArray:
    """Values at ``times``: exact time stamps ('point') or linear in time between interval
    means ('mean'; the nearest mean beyond the first/last one)."""
    if kind == "point":
        missing = [str(t) for t in times if t not in set(da.time.values)]
        if missing:
            raise NotPublished(f"{da.name}: instantaneous values missing at {missing}")
        return da.sel(time=times)
    source = da.time.values
    clipped = np.clip(times, source.min(), source.max())
    if source.size == 1:
        out = da.isel(time=[0] * len(times))
    else:
        out = da.interp(time=clipped)
    return out.assign_coords(time=times)


# --------------------------------------------------------------------------------------
# Profiles and fields on the EC-Earth3 grid
# --------------------------------------------------------------------------------------

def anomaly_weights(target: np.ndarray, anchors: np.ndarray, taper_decades: float) -> np.ndarray:
    """(target, anchor) weights that spread a departure known at ``anchors`` (hPa) to ``target``:
    linear in log(p) between anchors, constant below the lowest anchor (higher pressure), tapered
    linearly in log10(p) to zero over ``taper_decades`` above the highest one."""
    order = np.argsort(anchors)
    anchors = np.asarray(anchors, dtype=float)[order]
    log_a, weights = np.log(anchors), np.zeros((len(target), len(anchors)))
    for i, p in enumerate(np.asarray(target, dtype=float)):
        if p >= anchors[-1]:
            weights[i, -1] = 1.0
        elif p <= anchors[0]:
            decades = np.log10(anchors[0] / p)
            weights[i, 0] = max(0.0, 1.0 - decades / taper_decades) if taper_decades > 0 else 0.0
        else:
            k = np.searchsorted(anchors, p) - 1
            f = (np.log(p) - log_a[k]) / (log_a[k + 1] - log_a[k])
            weights[i, k], weights[i, k + 1] = 1.0 - f, f
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return weights[:, inverse]


def anchor_profile(daily: xr.DataArray, instant: Optional[xr.DataArray],
                   taper_decades: float = 1.0) -> xr.DataArray:
    """Daily-mean profile (time, level) at the instants plus the 6-hourly departure from it.

    The departure instant - daily is computed on the 6-hourly levels (daily interpolated there in
    log(p)) and spread to every daily level with ``anomaly_weights``. Exact on the 6-hourly levels.
    """
    if instant is None:
        return daily
    anchors = instant.level.values.astype(float)
    daily_at = daily.assign_coords(level=np.log(daily.level.values.astype(float))).sortby("level")
    daily_at = daily_at.interp(level=np.log(anchors)).assign_coords(level=anchors)
    departure = (instant - daily_at.transpose(*instant.dims)).transpose("level", ...)
    weights = anomaly_weights(daily.level.values, anchors, taper_decades)
    spread = np.tensordot(weights, departure.values, axes=(1, 0))      # (level, rest...)
    spread = xr.DataArray(spread, dims=departure.dims,
                          coords={**{d: departure[d] for d in departure.dims if d != "level"},
                                  "level": daily.level.values})
    out = daily + spread.transpose(*daily.dims)
    out.attrs = dict(daily.attrs)
    return out


def fill_nearest(da: xr.DataArray) -> xr.DataArray:
    """NaN (sea points of land-only fields) -> value of the nearest valid point, per 2D slice."""
    from scipy import ndimage
    values = da.values.copy()
    flat = values.reshape((-1,) + values.shape[-2:])
    for field in flat:
        invalid = ~np.isfinite(field)
        if invalid.all() or not invalid.any():
            continue
        _, (iy, ix) = ndimage.distance_transform_edt(invalid, return_indices=True)
        field[invalid] = field[iy[invalid], ix[invalid]]
    return da.copy(data=values)


def to_fraction(da: xr.DataArray) -> xr.DataArray:
    """Sea-ice cover as a fraction. EC-Earth3 declares siconca in % but stores fractions: divide
    by 100 only when the values say so."""
    if float(da.max()) > 1.5:
        da = da / 100.0
    return da.clip(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Regridding to the 0.25 deg grid
# --------------------------------------------------------------------------------------

def target_grid() -> xr.Dataset:
    return xr.Dataset(coords={"latitude": TARGET_LATITUDES, "longitude": TARGET_LONGITUDES})


def to_target(da: xr.DataArray) -> xr.DataArray:
    """Bilinear (periodic in longitude) from the EC-Earth3 regular grid to the target grid."""
    da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    out = regrid_latlon(da.to_dataset(name="_v"), TARGET_LATITUDES, TARGET_LONGITUDES)["_v"]
    return out.astype("float32").rename(da.name)


def ocean_to_target(tos: xr.DataArray, sea: xr.DataArray, neighbours: int = 4) -> xr.DataArray:
    """Ocean-grid (curvilinear, 2D latitude/longitude) field -> target sea points, inverse-distance
    weighted over the ``neighbours`` nearest valid ocean points. Land is NaN."""
    from scipy.spatial import cKDTree

    def xyz(lat, lon):
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)

    horizontal = [d for d in tos.dims if d != "time"]
    values = tos.transpose("time", *horizontal).values.reshape(tos.sizes["time"], -1)
    valid = np.isfinite(values).all(axis=0)
    source = xyz(np.asarray(tos.latitude).ravel()[valid], np.asarray(tos.longitude).ravel()[valid])
    lat2d, lon2d = np.meshgrid(sea.latitude.values, sea.longitude.values, indexing="ij")
    wanted = sea.values.ravel()
    distance, index = cKDTree(source).query(xyz(lat2d.ravel()[wanted], lon2d.ravel()[wanted]), k=neighbours)
    weights = 1.0 / np.maximum(distance, 1e-9)
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.full((tos.sizes["time"], wanted.size), np.nan, dtype="float32")
    for t in range(tos.sizes["time"]):
        out[t, wanted] = (values[t, valid][index] * weights).sum(axis=1)
    return xr.DataArray(out.reshape((tos.sizes["time"],) + sea.shape),
                        dims=("time", "latitude", "longitude"),
                        coords={"time": tos.time.values, "latitude": sea.latitude, "longitude": sea.longitude})


# --------------------------------------------------------------------------------------
# Derived variables (target grid)
# --------------------------------------------------------------------------------------

def surface_pressure(msl: xr.DataArray, t2m: xr.DataArray, surface_geopotential: xr.DataArray) -> xr.DataArray:
    """sp = msl exp(-phi_s / (Rd T)), T the mean of t2m and its value reduced to sea level."""
    height = surface_geopotential / G
    t_mean = t2m + 0.5 * LAPSE_RATE * height
    return msl * np.exp(-surface_geopotential / (RD * t_mean))


def dewpoint(huss: xr.DataArray, sp: xr.DataArray, t2m: xr.DataArray) -> xr.DataArray:
    """2 m dewpoint from specific humidity and surface pressure (Tetens over water, IFS
    constants), never above t2m."""
    vapour = huss * sp / (0.622 + 0.378 * huss)
    ratio = np.log(np.maximum(vapour, 1e-3) / 611.21)
    td = (32.19 * ratio - 17.502 * 273.16) / (ratio - 17.502)
    return np.minimum(td, t2m)


def column_integral(field: xr.DataArray, sp: xr.DataArray, dim: str = "level") -> xr.DataArray:
    """(1/g) * integral of ``field`` (on levels in hPa) from the top level down to the surface
    pressure sp (Pa), trapezoidal; levels below the surface are left out."""
    p = field[dim].values.astype(float) * 100.0
    order = np.argsort(p)
    p, values = p[order], field.isel({dim: order}).transpose(dim, ...).values
    surface = sp.transpose(*[d for d in field.dims if d != dim]).values
    total = np.zeros_like(surface, dtype="float64")
    for k in range(len(p) - 1):
        thickness = np.clip(np.minimum(p[k + 1], surface) - p[k], 0.0, None)
        total += 0.5 * (values[k] + values[k + 1]) * thickness
    return sp.copy(data=(total / G).astype("float32"))


def cloud_layers(tcc: xr.DataArray) -> xr.DataArray:
    """Low/medium/high cloud cover (not published): three equal layers with random overlap that
    reproduce tcc, c = 1 - (1 - tcc)^(1/3)."""
    return 1.0 - (1.0 - tcc.clip(0.0, 1.0)) ** (1.0 / 3.0)


def skin_temperature(sea: xr.DataArray, sst: xr.DataArray, ci: xr.DataArray, t2m: xr.DataArray,
                     tslsi: xr.DataArray, tas_day: xr.DataArray) -> xr.DataArray:
    """Land: daily-mean surface temperature plus the diurnal departure of t2m from its daily mean.
    Sea: SST over open water, t2m (at most 273.15 K) over sea ice, weighted by ci."""
    land = tslsi + (t2m - tas_day)
    ice = np.minimum(t2m, ICE_SURFACE_MAX)
    ice_fraction = ci.fillna(0.0)
    ocean = (1.0 - ice_fraction) * sst.fillna(ice) + ice_fraction * ice
    return xr.where(sea, ocean, land)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

def collect_requests(ic_card: dict, standard_variables: dict) -> Dict[Tuple[str, str], str]:
    """{(table, variable): role} of every CMIP6 field needed; role 'instant' marks the optional
    6-hourly part of a profile."""
    requests = {}
    for code, spec in ic_card["variables"].items():
        if code not in standard_variables:
            continue
        if "profile" in spec:
            requests[tuple(spec["profile"]["daily"])] = "daily"
            requests.setdefault(tuple(spec["profile"]["instant"]), "instant")
        elif "field" in spec:
            requests[tuple(spec["field"])] = "field"
    for source in ic_card.get("inputs", {}).values():
        requests[tuple(source)] = "field"
    return requests


def read_sources(ic_card: dict, standard_variables: dict, esgf: dict, experiment: str,
                 times: np.ndarray, reader: Callable = read_field) -> Tuple[Dict[Tuple[str, str], xr.DataArray], List[str]]:
    """Every CMIP6 field at the two instants; absent 6-hourly profile parts are left out."""
    fields, urls = {}, []
    tables = ic_card["tables"]
    for (table, variable), role in collect_requests(ic_card, standard_variables).items():
        kind = tables[table]
        window = time_window(kind, times)
        if (table, variable) == tuple(ic_card["inputs"].get("tsl", ())):
            window = (times.min(), times.min() + np.timedelta64(18, "h"))     # whole day for layer 2
        try:
            da, used = reader(esgf, experiment, table, variable, window)
        except NotPublished as err:
            if role != "instant":
                raise
            logging.warning(f"{err}: the profile uses the daily means only")
            continue
        fields[(table, variable)] = da
        urls.extend(used)
    return fields, urls


def native_state(fields: dict, ic_card: dict, standard_variables: dict,
                 times: np.ndarray) -> Tuple[Dict[str, xr.DataArray], List[str]]:
    """Standard variables published by EC-Earth3 and the derivation inputs, at the two instants,
    on the EC-Earth3 grid (profiles already on the standard levels)."""
    tables, taper = ic_card["tables"], ic_card.get("profiles", {}).get("taper_decades", 1.0)
    state, profile_names = {}, []
    for code, spec in ic_card["variables"].items():
        if code not in standard_variables or spec.get("derived"):
            continue
        scale = float(spec.get("scale", 1.0))
        if "profile" in spec:
            daily_key, instant_key = tuple(spec["profile"]["daily"]), tuple(spec["profile"]["instant"])
            daily = at_instants(fields[daily_key], times, tables[daily_key[0]])
            instant = None
            if instant_key in fields:
                instant = at_instants(fields[instant_key], times, tables[instant_key[0]])
            profile = anchor_profile(daily, instant, taper)
            if spec["name"] == "q":
                profile = profile.clip(min=0.0)
            state[spec["name"]] = (profile * scale).astype("float32")
            profile_names.append(spec["name"])
        else:
            key = tuple(spec["field"])
            da = at_instants(fields[key], times, tables[key[0]])
            if spec.get("fraction"):
                da = to_fraction(da)
            if spec.get("land_only"):
                da = fill_nearest(da)
            state[spec["name"]] = (da * scale).astype("float32")
    inputs = ic_card.get("inputs", {})
    for name, key in inputs.items():
        key = tuple(key)
        if name == "tsl":
            day = fill_nearest(fields[key])
            state["tsl"] = at_instants(day, times, "point")
            state["tsl_day"] = day.mean("time").expand_dims(time=times)
        elif name == "tos":
            state["tos"] = at_instants(fields[key], times, tables[key[0]]) + 273.15     # degC -> K
        else:
            da = at_instants(fields[key], times, tables[key[0]])
            state[name] = fill_nearest(da) if name == "tslsi" else da
    return state, profile_names


def prepare_initial_conditions(fields: dict, config: dict, ic_card: dict, standard_dict: dict,
                               times: np.ndarray) -> xr.Dataset:
    standard_variables = standard_dict["variables"]["data"]
    state, profile_names = native_state(fields, ic_card, standard_variables, times)

    # Upper air: daily levels -> 37 standard levels on the EC-Earth3 grid, then horizontally.
    profiles = xr.Dataset({name: state.pop(name) for name in profile_names})
    profiles["level"].attrs["units"] = standard_dict["pressure_levels"]["units"]["valid"]
    profiles = to_standard_levels(profiles, standard_dict["pressure_levels"]["values"])
    target = {name: to_target(profiles[name]) for name in profile_names}

    # Surface fields: horizontally (tos separately, from the ocean grid).
    tos = state.pop("tos")
    for name, da in state.items():
        target[name] = to_target(da.rename(name))

    grid = target_grid()
    static_card = read_ic_card(config["HPCROOTDIR"], "static")
    static_names = {spec["name"] for spec in static_card["variables"].values()}
    with xr.open_dataset(config["STATIC_DATA"]) as source:
        static = align_support_grid(source[[n for n in ("lsm", "z") if n in static_names]].load(), grid)
    sea = static["lsm"] < 0.5

    t2m = target["t2m"]
    sst = ocean_to_target(tos, sea)
    ci = target["ci"].clip(0.0, 1.0).where(sea)        # interpolation leaves ~1e-19 below zero
    tcc = target["tcc"].clip(0.0, 1.0)
    out = {name: target[name] for name in profile_names}
    out.update({name: target[name] for name in ("msl", "u10", "v10", "t2m", "u100", "v100")})
    out["tcc"] = tcc
    out["sd"] = target["sd"].clip(min=0.0).where(~sea, 0.0)
    out["swvl1"] = target["swvl1"].clip(0.0, 1.0).where(~sea, 0.0)
    out["swvl2"] = target["swvl2"].clip(0.0, 1.0).where(~sea, 0.0)
    out["ci"] = ci
    out["sst"] = sst.where(sea)
    out["sp"] = surface_pressure(target["msl"], t2m, static["z"])
    out["d2m"] = dewpoint(target["huss"], out["sp"], t2m)
    out["skt"] = skin_temperature(sea, sst, ci, t2m, target["tslsi"], target["tas_day"])
    out["sot"] = xr.concat([xr.where(sea, out["skt"], target["tsl"]),
                            xr.where(sea, out["skt"], target["tsl_day"])],
                           dim=xr.DataArray([1, 2], dims="SoilLevel", name="SoilLevel"))
    for name in ("lcc", "mcc", "hcc"):
        out[name] = cloud_layers(tcc)
    out["tcwv"] = column_integral(out["q"], out["sp"])
    selected = xr.Dataset({name: da.astype("float32").drop_vars(
        [c for c in da.coords if c not in ("time", "level", "latitude", "longitude", "SoilLevel")])
        for name, da in out.items()})

    # Static and climatological variables, then tcw = tcwv + cloud condensate of the climatology.
    names, _, long_names, units, missing = define_ics_mappers(ic_card["variables"], standard_variables)
    for card_name, key, climatology in [("static", "STATIC_DATA", False),
                                        ("climatology", "CLIMATOLOGY_DATA", True)]:
        selected, missing = add_fallback(selected, missing, read_ic_card(config["HPCROOTDIR"], card_name),
                                         config[key], climatology, reason=FALLBACK_REASON)
    missing = drop_optional_missing(missing, standard_variables)
    if missing:
        raise ValueError(f"No EC-Earth3/static/climatology source for variable IDs: {sorted(missing)}")
    condensate = selected["clwc"] + selected["ciwc"] if {"clwc", "ciwc"} <= set(selected) else 0.0
    selected["tcw"] = (selected["tcwv"] + (column_integral(condensate, selected["sp"])
                                           if not np.isscalar(condensate) else 0.0)).astype("float32")
    for name in long_names:                 # set directly: the CMIP6 attributes were dropped
        if name in selected:
            selected[name].attrs.update(long_name=long_names[name], units=units[name])
    selected["level"].attrs.update(units="hPa", standard_name="air_pressure", long_name="pressure",
                                   positive="down")

    derivations = {
        "z": "zg * g", "q": "clipped at 0",
        "sd": "snw / 1000 (0 over sea)", "swvl1": "mrsos / 100 (0 over sea)",
        "swvl2": "mrsos / 100, no deeper layer published (0 over sea)",
        "ci": "siconca as a fraction (NaN over land)", "sst": "tos (ORCA1) inverse-distance weighted, K",
        "sp": "msl reduced to the static orography with t2m and a 6.5 K/km lapse rate",
        "d2m": "from huss and sp (Tetens, IFS constants), at most t2m",
        "tcwv": "vertical integral of q down to sp", "tcw": "tcwv + integral of the climatological clwc+ciwc",
        "skt": "land: daily tslsi + (tas - daily tas); sea: sst, sea ice: min(t2m, 273.15 K), weighted by ci",
        "sot": "layer 1: tsl (5 cm); layer 2: mean of the day's four tsl instants; skt over sea",
        "lcc": "1 - (1 - tcc)^(1/3)", "mcc": "1 - (1 - tcc)^(1/3)", "hcc": "1 - (1 - tcc)^(1/3)",
    }
    for name, text in derivations.items():
        if name in selected:
            selected[name].attrs["derivation"] = text
    for name in profile_names:
        selected[name].attrs["derivation"] = (selected[name].attrs.get("derivation", "") +
                                              "; daily mean (Eday) + 6-hourly departure").lstrip("; ")

    # Masked by construction: sst and ci over land, the wave fields (not published) everywhere.
    allowed_nan = {"sst", "ci"} | variables_of_type(standard_variables, "wave")
    for name, da in selected.data_vars.items():
        if name not in allowed_nan and not bool(np.isfinite(da).all()):
            raise ValueError(f"Non-finite values in EC-Earth3 initial conditions: {name}")
    return selected.transpose("time", "level", ..., "latitude", "longitude")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def check_existing(output: str, ic_card: dict, standard: dict, experiment: str, member: str) -> bool:
    if not (os.path.isdir(output) and os.listdir(output)):
        return False
    with xr.open_zarr(output) as existing:
        variables = standard["variables"]["data"]
        optional = optional_variables(variables)
        expected = {v["short_name"] for code, v in variables.items() if code not in optional}
        same = (existing.attrs.get("ic_name") == ic_card["ic_name"]
                and existing.attrs.get("cmip6_experiment") == experiment
                and existing.attrs.get("cmip6_member") == member)
        if not same or not expected.issubset(existing.data_vars):
            raise ValueError(f"Existing ICs at {output} do not match {ic_card['ic_name']} {experiment} "
                             f"{member}; use a new INI_DATA_PATH or explicitly remove the old ICs.")
    print(f"Output already exists at {output}, skipping download.")
    return True


def main() -> None:
    args = parse_arguments()
    config = read_config(args.config)
    output = config["INI_DATA_PATH"]
    ic_card = read_ic_card(config["HPCROOTDIR"], config["IC_NAME"])
    standard = read_std_version(config["HPCROOTDIR"], config["STD_VERSION"])

    esgf = dict(ic_card["esgf"])
    esgf["member"] = _setting(config, "IC_MEMBER") or esgf["member"]
    experiment = experiment_for(config["START_TIME"], esgf, _setting(config, "IC_EXPERIMENT"))
    if check_existing(output, ic_card, standard, experiment, esgf["member"]):
        return
    times = ic_times(config["START_TIME"])
    logging.info(f"EC-Earth3 {experiment} {esgf['member']}: states at {[str(t) for t in times]}")

    fields, urls = read_sources(ic_card, standard["variables"]["data"], esgf, experiment, times)
    selected = prepare_initial_conditions(fields, config, ic_card, standard, times)
    selected.attrs.update(
        source=f"EC-Earth3 CMIP6 {experiment} {esgf['member']} (ESGF, OPeNDAP)",
        ic_name=ic_card["ic_name"],
        cmip6_source_id=esgf["source_id"],
        cmip6_experiment=experiment,
        cmip6_member=esgf["member"],
        cmip6_files=sorted(set(urls)),
        fallback_source="static support data; ERA5 climatology (clwc, ciwc)",
        licence="CC BY 4.0, EC-Earth Consortium",
    )

    # A failed write must not appear to be a complete set of initial conditions.
    temporary = output + ".partial"
    shutil.rmtree(temporary, ignore_errors=True)
    selected.chunk({"time": 1}).to_zarr(temporary, mode="w", zarr_format=2)
    os.replace(temporary, output)
    print(f"Wrote EC-Earth3 {experiment} {esgf['member']} initial conditions to {output}")


if __name__ == "__main__":
    main()
