"""
The AIUQ output convention: one writer for every model.

Every model output is a NetCDF-4 file that follows the CF conventions (CF-1.8) and holds one variable of one
ensemble member. Names, units and CF standard names come from the AIUQ-st standard
(``conf/AIUQ-st/<version>/variables.yml`` and ``outputs.yml``); this module adds what is the same for all
variables: the order and the names of the dimensions, the coordinate variables, the description of time
(valid time, initial time, lead time, the period of accumulations and means) and the global attributes that say
which run wrote the file. How the values are stored (type, compression, chunks) is the experiment's choice,
``EXPERIMENT.OUT_COMPRESSION``: see ``AIUQst_lib/compression.py``.

Layout of a file (``OUT_RES`` a regular grid)::

    dimensions:  valid_time, [level,] latitude, longitude[, bnds]
    <short_name>(valid_time, [level,] latitude, longitude)   float32, or integers with scale_factor
    valid_time(valid_time)          standard_name time, hours since 1970-01-01
    valid_time_bnds(valid_time, 2)  only for totals and means: the period each value refers to
    time()                          forecast_reference_time: the initial time of the forecast
    step(valid_time)                forecast_period, hours (not for daily output)
    number()                        realization: the ensemble member
    level(level)                    air_pressure, hPa, from the surface upwards (1000 ... 1)
    latitude(latitude)              degrees_north, south to north
    longitude(longitude)            degrees_east, 0 ... 360
    height()                        only for fields at a fixed height (2 m, 10 m, 100 m)

On a grid that is not regular (AIFS ``OUT_RES: original``) the two horizontal dimensions are one dimension
``values`` and ``latitude(values)`` / ``longitude(values)`` are auxiliary coordinates.

The format has a version (global attribute ``aiuq_output_format``); readers can rely on it.
"""

# Built-in/Generics
import datetime as dt
import os
import re
from dataclasses import dataclass, field

# Third party
import netCDF4
import numpy as np

# Local
from AIUQst_lib.compression import (Scheme, check_values, create_variable, largest_integer, read_values,
                                    write_values)

FORMAT_VERSION = "1.0"
CONVENTIONS = "CF-1.8"
TIME_UNITS = "hours since 1970-01-01 00:00:00"
CALENDAR = "proleptic_gregorian"
EPOCH = np.datetime64("1970-01-01T00:00:00", "s")
REFERENCES = "https://github.com/JGrassi97/AIUQ"

BLOCK_BYTES = 256 * 2 ** 20                    # data are written in blocks of valid times of about this size

MODEL_LABELS = {"neuralgcm": "NeuralGCM", "aifs": "AIFS", "ms-aurora": "Aurora", "weathernext": "WeatherNext"}
ICS_LABELS = {"era5": "ERA5", "eerie": "EERIE", "ecmwf-an": "ECMWF analysis", "ec-earth3": "EC-Earth3"}
ALIASES = {"lat": "latitude", "lon": "longitude", "plev": "level", "pressure_level": "level", "isobaricInhPa": "level"}

COORD_ATTRS = {
    "valid_time": {"standard_name": "time", "long_name": "valid time", "axis": "T"},
    "time": {"standard_name": "forecast_reference_time", "long_name": "initial time of the forecast"},
    "step": {"standard_name": "forecast_period", "long_name": "time since the initial time of the forecast",
             "units": "hours"},
    "number": {"standard_name": "realization", "long_name": "ensemble member", "units": "1"},
    "level": {"standard_name": "air_pressure", "long_name": "pressure level", "units": "hPa", "positive": "down",
              "axis": "Z"},
    "latitude": {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north"},
    "longitude": {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east"},
    "height": {"standard_name": "height", "long_name": "height above the surface", "units": "m", "positive": "up"},
}


def _clean(value) -> str:
    """A configuration value as text; empty when it is unset or an Autosubmit placeholder left unresolved."""
    text = "" if value is None else str(value).strip()
    return "" if (text.startswith("%") and text.endswith("%")) or text in ("None", "null") else text


@dataclass
class RunInfo:
    """What a file says about the run that wrote it (global attributes)."""
    model: str = ""
    checkpoint: str = ""
    ics: str = ""
    member: object = None
    run_type: str = "hindcast"
    expid: str = ""
    std_version: str = ""
    commit: str = ""
    branch: str = ""
    repository: str = ""
    institution: str = ""
    forcing: str = ""
    ics_experiment: str = ""
    ics_member: str = ""
    scheme: Scheme = field(default_factory=Scheme)   # how the values are stored (OUT_COMPRESSION)
    extra: dict = field(default_factory=dict)        # further global attributes, written as they are

    @classmethod
    def from_config(cls, config: dict, **extra) -> "RunInfo":
        """From the job configuration (templates/config.yml as rendered by Autosubmit)."""
        get = lambda key: _clean(config.get(key))                                    # noqa: E731
        checkpoint = get("CHECKPOINT_NAME") or os.path.basename(get("MODEL_CHECKPOINT"))
        run_type = (get("RUN_TYPE") or "hindcast").lower()
        return cls(model=get("MODEL_NAME"), checkpoint=checkpoint, ics=get("IC_NAME"), member=get("RNG_KEY"),
                   run_type=run_type, expid=get("EXPID"), std_version=get("STD_VERSION"), commit=get("GIT_COMMIT"),
                   branch=get("GIT_BRANCH"), repository=get("GIT_ORIGIN"), institution=get("INSTITUTION"),
                   forcing=get("FORCING_VERSION") if run_type == "amip" else "",
                   ics_experiment=get("IC_EXPERIMENT"), ics_member=get("IC_MEMBER"),
                   scheme=Scheme.from_config(config), extra=extra)

    def global_attributes(self, long_name: str, compression: str) -> dict:
        model = MODEL_LABELS.get(self.model, self.model or "AI model")
        ics = ICS_LABELS.get(self.ics, self.ics)
        what = "AMIP simulation" if self.run_type == "amip" else "forecast"
        title = f"{model} {what}" + (f" from {ics} initial conditions" if ics else "") + f": {long_name}"
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        attrs = {
            "Conventions": CONVENTIONS,
            "title": title,
            "institution": self.institution,
            "source": f"{model}" + (f" ({self.checkpoint})" if self.checkpoint else "") + ", AI weather model run with AIUQ",
            "history": f"{now} written by AIUQ (output format {FORMAT_VERSION})",
            "references": self.repository or REFERENCES,
            "aiuq_output_format": FORMAT_VERSION,
            "aiuq_standard": f"AIUQ-st {self.std_version}" if self.std_version else "",
            "aiuq_model": self.model,
            "aiuq_checkpoint": self.checkpoint,
            "aiuq_initial_conditions": self.ics,
            "aiuq_ics_experiment": self.ics_experiment,
            "aiuq_ics_member": self.ics_member,
            "aiuq_run_type": self.run_type,
            "aiuq_forcing": self.forcing,
            "aiuq_experiment": self.expid,
            "aiuq_member": "" if self.member in (None, "") else str(self.member),
            "aiuq_commit": self.commit,
            "aiuq_branch": self.branch,
            "aiuq_compression": compression,
        }
        attrs.update({str(k): v for k, v in self.extra.items()})
        return {k: v for k, v in attrs.items() if v not in ("", None)}


def window_hours(window, step_hours=None):
    """Length in hours of the period of a card ``window`` ('6 h ending at valid_time', 'output step ...')."""
    if not window:
        return None
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*h\b", str(window))
    if match:
        return float(match.group(1))
    return float(step_hours) if step_hours and "output step" in str(window) else None


def _hours(value) -> str:
    return f"{value:g}"


def time_cell(spec: dict, daily: bool = False, step_hours=None):
    """How a value relates to its valid time: ``(cell_methods, (start, end), comment)``.

    ``start`` and ``end`` are the limits of the period in hours from the valid time; all three are None for
    a field valid at an instant. ``spec`` is an entry of ``variables.output_specs``; ``step_hours`` the
    spacing of the model output (6) and ``daily`` the output frequency OUT_FREQ daily, whose values carry the
    day (00 UTC) as valid time.
    """
    accumulated = bool(spec.get("accumulated"))
    window = window_hours(spec.get("window"), step_hours)
    every = f" (interval: {_hours(step_hours)} hours)" if step_hours else ""
    if daily:
        if accumulated:
            return "valid_time: sum", (0.0, 24.0), "Daily total: sum of the totals of the periods that end within the day."
        if window:
            return (f"valid_time: mean{every}", (0.0, 24.0),
                    f"Daily mean of values that are each the mean over the last {_hours(window)} h before a model output time.")
        return f"valid_time: mean{every}", (0.0, 24.0), None
    if accumulated or window:
        length = window or (float(step_hours) if step_hours else None)
        if length is None:
            raise ValueError(f"The period of '{spec.get('long_name', '?')}' is unknown: give the card a 'window' "
                             "or pass the output step.")
        return ("valid_time: sum" if accumulated else "valid_time: mean"), (-length, 0.0), None
    return None, None, None


def variable_attributes(short_name: str, spec: dict, cell_methods=None, comment=None) -> dict:
    """netCDF attributes of the data variable, from its AIUQ-st definition."""
    attrs = {}
    if spec.get("standard_name"):
        attrs["standard_name"] = str(spec["standard_name"])
    attrs["long_name"] = str(spec.get("long_name") or short_name)
    attrs["units"] = str(spec.get("units") if spec.get("units") not in (None, "") else "1")
    if cell_methods:
        attrs["cell_methods"] = cell_methods
    if comment:
        attrs["comment"] = comment
    code = spec.get("code")
    if isinstance(code, (int, np.integer)) and 0 < int(code) < 999999:      # 999999: not an ECMWF parameter
        attrs["GRIB_paramId"] = np.int32(code)
    return attrs


def _to_hours(times) -> np.ndarray:
    """datetime64 (or timedelta64) values as hours since the epoch (or as a number of hours)."""
    values = np.asarray(times)
    if np.issubdtype(values.dtype, np.datetime64):
        return (values.astype("datetime64[s]") - EPOCH) / np.timedelta64(1, "s") / 3600.0
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[s]") / np.timedelta64(1, "s") / 3600.0
    return values.astype("f8")


def _number_type(values) -> str:
    """int32 when every value is a whole number that fits, otherwise float64 (never 64-bit integers)."""
    values = np.atleast_1d(np.asarray(values, dtype="f8"))
    whole = np.all(np.isfinite(values)) and np.all(values == np.round(values)) and np.all(np.abs(values) < 2 ** 31)
    return "i4" if whole else "f8"


def write_output(path: str, data, short_name: str, spec: dict, run: RunInfo, *, daily: bool = False,
                 step_hours=None, reference_time=None) -> str:
    """Write one output variable of one member as an AIUQ output file and return its path.

    ``data`` is an ``xarray.DataArray`` with the dimension ``valid_time``, optionally ``level``, and either
    ``latitude`` and ``longitude`` (also named ``lat`` / ``lon``, in any order and direction) or one
    horizontal dimension with ``latitude`` / ``longitude`` coordinates along it. It may be in memory or read
    lazily from a file: values are taken one block of valid times at a time. The initial time of the forecast
    is ``reference_time`` or the scalar coordinate ``time`` of ``data``. ``spec`` is the variable's entry of
    ``variables.output_specs``; ``daily`` and ``step_hours`` as in :func:`time_cell`.

    The values are stored as ``run.scheme`` asks for this variable (``AIUQst_lib/compression.py``). The file
    is written beside ``path``, read back and compared with the data, and renamed only then: a file with the
    final name is complete and holds what was computed, within the error its compression allows.
    """
    da = data.rename({old: new for old, new in ALIASES.items() if old in data.dims or old in data.coords})
    if "valid_time" not in da.dims:
        raise ValueError(f"'{short_name}': outputs need the dimension 'valid_time', got {da.dims}")
    regular = "latitude" in da.dims and "longitude" in da.dims
    if regular:
        horizontal = ["latitude", "longitude"]
    else:
        others = [d for d in da.dims if d not in ("valid_time", "level")]
        if len(others) != 1 or "latitude" not in da.coords or "longitude" not in da.coords:
            raise ValueError(f"'{short_name}': expected latitude and longitude dimensions, or one horizontal "
                             f"dimension with latitude/longitude coordinates; got {da.dims}")
        if others[0] != "values":
            da = da.rename({others[0]: "values"})
        horizontal = ["values"]
    has_level = "level" in da.dims
    da = da.transpose("valid_time", *(["level"] if has_level else []), *horizontal)

    # Fixed directions: latitude from south to north, levels from the surface upwards (decreasing pressure).
    if regular and da.sizes["latitude"] > 1 and float(da["latitude"][0]) > float(da["latitude"][-1]):
        da = da.isel(latitude=slice(None, None, -1))
    if has_level and da.sizes["level"] > 1:
        order = np.argsort(-np.asarray(da["level"].values, dtype="f8"), kind="stable")
        if not np.array_equal(order, np.arange(order.size)):
            da = da.isel(level=order)

    if reference_time is None:
        if "time" not in da.coords or da["time"].ndim != 0:
            raise ValueError(f"'{short_name}': the initial time of the forecast is missing (scalar coordinate 'time')")
        reference_time = da["time"].values
    reference_hours = float(_to_hours(reference_time))
    valid_hours = _to_hours(da["valid_time"].values)
    if valid_hours.size > 1 and not np.all(np.diff(valid_hours) > 0):
        raise ValueError(f"'{short_name}': valid times must increase")
    if step_hours is None and not daily and valid_hours.size > 1:
        step_hours = float(valid_hours[1] - valid_hours[0])

    cell_methods, limits, comment = time_cell(spec, daily, step_hours)
    attrs = variable_attributes(short_name, spec, cell_methods, comment)
    height = spec.get("height")
    member = run.member if run.member not in (None, "") else None
    try:
        member = None if member is None else int(member)
    except (TypeError, ValueError):
        member = None                              # a member that is not a number stays in the global attributes
    scheme = run.scheme
    plan = scheme.plan(short_name, spec)

    # Blocks of whole chunks of valid times, so each chunk is compressed once.
    per_time = int(np.prod([da.sizes[name] for name in da.dims[1:]], dtype="i8")) * 8
    block = max(1, int(BLOCK_BYTES // max(per_time, 1)))
    time_chunk = scheme.chunksizes(da.dims, da.sizes)[0]
    block = max(time_chunk, block - block % time_chunk)
    starts = range(0, da.sizes["valid_time"], block)

    def values_of(start):
        return np.asarray(da.isel(valid_time=slice(start, start + block)).values)

    # Integers: 16 bits when every value fits, so the largest one must be known before the variable exists.
    largest = largest_integer((values_of(start) for start in starts), plan) if plan.mode == "integer" else 0.0

    scalars = ["time"] + ([] if daily else ["step"]) + (["number"] if member is not None else [])
    scalars += ["height"] if height not in (None, "") else []
    attrs["coordinates"] = " ".join(scalars + ([] if regular else ["latitude", "longitude"]))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.writing"
    try:
        with netCDF4.Dataset(temporary, "w", format="NETCDF4") as nc:
            nc.setncatts(run.global_attributes(attrs["long_name"], scheme.tag({short_name: plan})))
            for name in da.dims:
                nc.createDimension(name, da.sizes[name])

            kind = _number_type(np.concatenate([valid_hours, [reference_hours]] +
                                               ([valid_hours + limits[0], valid_hours + limits[1]] if limits else [])))
            var = nc.createVariable("valid_time", kind, ("valid_time",))
            var.setncatts({**COORD_ATTRS["valid_time"], "units": TIME_UNITS, "calendar": CALENDAR})
            var[:] = valid_hours
            if limits:
                var.setncattr("bounds", "valid_time_bnds")
            var = nc.createVariable("time", kind, ())
            var.setncatts({**COORD_ATTRS["time"], "units": TIME_UNITS, "calendar": CALENDAR})
            var.assignValue(reference_hours)
            if not daily:
                lead = valid_hours - reference_hours
                var = nc.createVariable("step", _number_type(lead), ("valid_time",))
                var.setncatts(COORD_ATTRS["step"])
                var[:] = lead
            if member is not None:
                var = nc.createVariable("number", "i4", ())
                var.setncatts(COORD_ATTRS["number"])
                var.assignValue(member)
            if has_level:
                levels = np.asarray(da["level"].values, dtype="f8")
                var = nc.createVariable("level", _number_type(levels), ("level",))
                var.setncatts(COORD_ATTRS["level"])
                var[:] = levels
            for name, axis in (("latitude", "Y"), ("longitude", "X")):
                var = nc.createVariable(name, "f8", da[name].dims)
                var.setncatts({**COORD_ATTRS[name], **({"axis": axis} if regular else {})})
                var[:] = np.asarray(da[name].values, dtype="f8")
            if "height" in scalars:
                var = nc.createVariable("height", "f8", ())
                var.setncatts(COORD_ATTRS["height"])
                var.assignValue(float(height))

            var = create_variable(nc, short_name, da.dims, plan, scheme, da.dtype, largest)
            var.setncatts(attrs)
            for start in starts:
                values = values_of(start)
                write_values(var, slice(start, start + values.shape[0]), values, plan)
            if limits:
                # After the data variable: a reader that takes the first variable with two or more dimensions
                # finds the data, not the bounds.
                nc.createDimension("bnds", 2)
                bounds = nc.createVariable("valid_time_bnds", kind, ("valid_time", "bnds"))
                bounds[:] = np.stack([valid_hours + limits[0], valid_hours + limits[1]], axis=1)
        # What is on disk must be what was computed, within the error this plan allows.
        with netCDF4.Dataset(temporary) as nc:
            for start in starts:
                values = values_of(start)
                check_values(values, read_values(nc[short_name], slice(start, start + values.shape[0])), plan,
                             f"{os.path.basename(path)} {short_name}: ")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    record_output(path)
    return path


# Environment variable with the file in which a job lists the outputs it wrote (set by the job templates).
RECORD_ENV = "AIUQ_OUTPUT_RECORD"


def record_output(path: str) -> None:
    """Add ``path`` (absolute) to the job's list of written outputs, when the job keeps one.

    The templates of the jobs that write final outputs set ``AIUQ_OUTPUT_RECORD`` to
    ``<HPCROOTDIR>/store_pending/<SDATE>/<job>.txt``; the STORE job of that start date moves exactly the files
    listed there to the output store (``templates/store.sh``). Outside a job nothing is recorded.
    """
    record = os.environ.get(RECORD_ENV, "")
    if not record:
        return
    os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
    with open(record, "a") as fh:
        fh.write(os.path.abspath(path) + "\n")
