"""
Compression of the AIUQ outputs, chosen per experiment (``EXPERIMENT.OUT_COMPRESSION``).

A *scheme* is a preset plus a few options; for every variable it gives a *plan*, one of four ways of storing
the values:

  none      as the model gives them: its floating-point type, no compression, no chunking
  lossless  float32 (the precision the models compute in), zlib with shuffle, chunked
  bits      as lossless, with the values rounded to ``keepbits`` significant bits (netCDF BitRound:
            relative error at most 2**-keepbits)
  integer   integers with a step, the CF way (``scale_factor`` / ``add_offset``): error at most step / 2.
            Readers decode them to floating point in the units of the variable.

The presets:

  none      every variable 'none'
  lossless  every variable 'lossless'                                     (the default)
  bits      every variable 'bits'
  integer   the variables with a step in the AIUQ-st standard as integers with that step (the 'integer'
            column: 1 in the unit people read: 1 degree C, 1 m/s, 1 hPa, 1 m, 1 mm, 1 %), the others 'bits'
  mixed     as integer with the 'mixed' column: the same, but precipitation at 0.1 mm
  custom    a rule per variable, given in the options

The steps live in the standard (``pack`` of ``conf/AIUQ-st/<version>/variables.yml`` and ``outputs.yml``),
so that the code, the Launcher and the documentation read the same table.

Every file says how it is stored in the global attribute ``aiuq_compression`` (JSON). Files compressed by
the earlier scripts carry a text instead ("float32, BitRound keepbits=12, ..." or "AIUQ mixed integer packing
v1 (...)"): they are read like the others and count as already compressed when what is on disk is what the
scheme asks for.

Needs netCDF4-python >= 1.6 on netcdf-c >= 4.9 (BitRound). zstd is not used: the HDF5 plugin is missing on
the machines this was measured on, and zlib level 1 with shuffle compresses as much as higher levels.
"""

# Built-in/Generics
import ast
import json
import os
import re
from dataclasses import dataclass

# Third party
import netCDF4
import numpy as np
import yaml

SCHEMA = 1
PRESETS = ("none", "lossless", "bits", "integer", "mixed", "custom")
DEFAULT_PRESET = "lossless"
DEFAULT_KEEPBITS = 12
KEEPBITS_RANGE = (4, 23)                      # float32 has 23 mantissa bits
DEFAULT_COMPLEVEL = 1
# Chunks along (valid_time, latitude, longitude); every other dimension (level) has one item per chunk.
# 'point': time series at a point and small maps in milliseconds, about 5 % larger files than 'map'.
CHUNK_PRESETS = {"point": (4, 45, 90), "map": (1, None, None)}      # None: the whole dimension
DEFAULT_CHUNKS = "point"
FILL = {"i2": -32767, "i4": -2147483647}
INT16_LIMIT = 32000                           # largest stored integer that still goes in 16 bits
# From the most to the least exact; rewriting a file with a more exact plan cannot bring back what was lost.
LOSS_ORDER = {"none": 0, "lossless": 0, "bits": 1, "integer": 2}
LATITUDES = ("latitude", "lat")
LONGITUDES = ("longitude", "lon")
LEGACY_BITS = re.compile(r"^float32, BitRound keepbits=(\d+),")
LEGACY_MIXED = "AIUQ mixed integer packing v1"
ENCODING_ATTRS = ("_FillValue", "missing_value", "scale_factor", "add_offset",
                  "_QuantizeBitRoundNumberOfSignificantBits", "_QuantizeBitGroomNumberOfSignificantDigits",
                  "_QuantizeGranularBitRoundNumberOfSignificantDigits")


class CompressionError(ValueError):
    """A scheme that cannot be used, or a file that does not hold what was written into it."""


@dataclass(frozen=True)
class Plan:
    """How one variable is stored."""
    mode: str                     # none | lossless | bits | integer
    keepbits: int = 0             # bits
    step: float = 0.0             # integer
    offset: float = 0.0           # integer

    def describe(self) -> dict:
        if self.mode == "bits":
            return {"mode": "bits", "keepbits": self.keepbits}
        if self.mode == "integer":
            return {"mode": "integer", "step": self.step, "offset": self.offset}
        return {"mode": self.mode}


def _unset(value) -> bool:
    """True for a setting that was not given (also an Autosubmit placeholder left unresolved)."""
    if value is None:
        return True
    text = str(value).strip()
    return text in ("", "None", "null", "{}", "[]") or (text.startswith("%") and text.endswith("%"))


def _literal(value):
    """A list or mapping that reached the job configuration as text (Autosubmit renders them with repr)."""
    if isinstance(value, str):
        text = value.strip()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return yaml.safe_load(text)
    return value


def parse_custom(value) -> dict:
    """The per-variable rules of the 'custom' preset as ``{variable: rule}``.

    Autosubmit hands a job only the single values of the experiment configuration, never a nested mapping,
    so in ``main.yml`` the rules are a list of texts, one per variable::

        CUSTOM:
          - "t2m: {step: 0.5, offset: 273.15}"
          - "tp: {step: 0.0001}"
          - "default: {keepbits: 12}"

    A mapping is accepted too, for the Launcher and for calls from Python.
    """
    if _unset(value):
        return {}
    value = _literal(value)
    rules = {}
    if isinstance(value, dict):
        items = [{k: v} for k, v in value.items()]
    elif isinstance(value, (list, tuple)):
        items = [yaml.safe_load("{" + item + "}") if isinstance(item, str) else item for item in value]
    else:
        raise CompressionError(f"custom compression rules must be a list of 'variable: {{...}}' texts, got {value!r}")
    for item in items:
        if not isinstance(item, dict) or len(item) != 1:
            raise CompressionError(f"a custom compression rule is 'variable: {{...}}', got {item!r}")
        (name, rule), = item.items()
        if not isinstance(rule, dict):
            raise CompressionError(f"the custom compression rule of '{name}' must be a mapping, got {rule!r}")
        rules[str(name).strip().lower()] = {str(k).strip().lower(): v for k, v in rule.items()}
    return rules


def _plan_from_rule(name: str, rule: dict, keepbits: int) -> Plan:
    """One rule of the 'custom' preset: a step (integers), a number of bits, or ``mode: lossless`` / ``none``."""
    unknown = set(rule) - {"step", "offset", "keepbits", "mode"}
    if unknown:
        raise CompressionError(f"custom compression rule of '{name}': unknown key(s) {sorted(unknown)}")
    mode = str(rule.get("mode", "")).strip().lower()
    if "step" in rule or mode == "integer":
        try:
            step, offset = float(rule["step"]), float(rule.get("offset", 0.0))
        except (KeyError, TypeError, ValueError):
            raise CompressionError(f"custom compression rule of '{name}': 'step' must be a number")
        if not np.isfinite(step) or step <= 0 or not np.isfinite(offset):
            raise CompressionError(f"custom compression rule of '{name}': the step must be greater than zero")
        return Plan("integer", step=step, offset=offset)
    if "keepbits" in rule or mode == "bits":
        return Plan("bits", keepbits=_check_keepbits(rule.get("keepbits", keepbits), f"custom rule of '{name}'"))
    if mode in ("lossless", "none"):
        return Plan(mode)
    raise CompressionError(f"custom compression rule of '{name}': give 'step', 'keepbits' or 'mode: lossless'")


def _check_keepbits(value, where="keepbits") -> int:
    try:
        bits = int(float(value))
    except (TypeError, ValueError):
        raise CompressionError(f"{where}: the number of bits must be a whole number, got {value!r}")
    if not KEEPBITS_RANGE[0] <= bits <= KEEPBITS_RANGE[1]:
        raise CompressionError(f"{where}: {bits} bits is outside {KEEPBITS_RANGE[0]}-{KEEPBITS_RANGE[1]}")
    return bits


def _check_chunks(value):
    """'point', 'map' or three whole numbers [valid times, latitudes, longitudes]."""
    if _unset(value):
        return DEFAULT_CHUNKS
    value = _literal(value) if not isinstance(value, str) or value.strip().startswith(("[", "(")) else value.strip().lower()
    if isinstance(value, str):
        if value not in CHUNK_PRESETS:
            raise CompressionError(f"chunks must be 'point', 'map' or [valid times, latitudes, longitudes], got '{value}'")
        return value
    try:
        sizes = tuple(int(v) for v in value)
        if len(sizes) != 3 or any(v < 1 for v in sizes) or any(int(v) != float(v) for v in value):
            raise ValueError
    except (TypeError, ValueError):
        raise CompressionError(f"chunks must be 'point', 'map' or three whole numbers of at least 1, got {value!r}")
    return sizes


class Scheme:
    """A compression preset with its options, resolved against the AIUQ-st standard."""

    def __init__(self, preset=DEFAULT_PRESET, keepbits=None, chunks=None, complevel=None, custom=None):
        self.preset = DEFAULT_PRESET if _unset(preset) else str(preset).strip().lower()
        if self.preset not in PRESETS:
            raise CompressionError(f"OUT_COMPRESSION must be one of {', '.join(PRESETS)}; got '{preset}'")
        self.keepbits = DEFAULT_KEEPBITS if _unset(keepbits) else _check_keepbits(keepbits)
        self.chunks = _check_chunks(chunks)
        try:
            self.complevel = DEFAULT_COMPLEVEL if _unset(complevel) else int(float(complevel))
        except (TypeError, ValueError):
            raise CompressionError(f"complevel must be a whole number from 1 to 9, got {complevel!r}")
        if not 1 <= self.complevel <= 9:
            raise CompressionError(f"complevel must be from 1 to 9, got {self.complevel}")
        self.custom = parse_custom(custom)
        if self.preset == "custom":
            if not self.custom:
                raise CompressionError(
                    "OUT_COMPRESSION custom needs its rules: EXPERIMENT.OUT_COMPRESSION_OPTS.CUSTOM as a list of "
                    "texts such as \"t2m: {step: 0.5, offset: 273.15}\" (a nested mapping does not reach the jobs)")
            for name, rule in self.custom.items():       # fail now, not at the first file
                _plan_from_rule(name, rule, self.keepbits)
        elif self.custom:
            raise CompressionError(f"custom compression rules are only for OUT_COMPRESSION custom, not '{self.preset}'")

    @classmethod
    def from_config(cls, config: dict) -> "Scheme":
        """From the job configuration (templates/config.yml as rendered by Autosubmit)."""
        return cls(config.get("OUT_COMPRESSION"), config.get("OUT_COMPRESSION_KEEPBITS"),
                   config.get("OUT_COMPRESSION_CHUNKS"), config.get("OUT_COMPRESSION_COMPLEVEL"),
                   config.get("OUT_COMPRESSION_CUSTOM"))

    def plan(self, name: str, spec: dict = None) -> Plan:
        """How the variable ``name`` is stored; ``spec`` is its AIUQ-st definition (for its ``pack`` steps)."""
        if self.preset in ("none", "lossless"):
            return Plan(self.preset)
        if self.preset == "bits":
            return Plan("bits", keepbits=self.keepbits)
        if self.preset == "custom":
            rule = self.custom.get(str(name).lower(), self.custom.get("default"))
            return Plan("bits", keepbits=self.keepbits) if rule is None else _plan_from_rule(name, rule, self.keepbits)
        pack = (spec or {}).get("pack") or {}
        step = pack.get(self.preset)
        if step in (None, ""):
            return Plan("bits", keepbits=self.keepbits)          # no step for this variable: keep its bits
        return Plan("integer", step=float(step), offset=float(pack.get("offset", 0.0)))

    def chunksizes(self, dims, sizes) -> list:
        """Chunk size along each dimension of a variable (``sizes``: {dimension: length})."""
        times, lats, lons = CHUNK_PRESETS[self.chunks] if isinstance(self.chunks, str) else self.chunks
        out = []
        for dim in dims:
            if dim == "valid_time":
                want = times
            elif dim in LATITUDES:
                want = lats
            elif dim in LONGITUDES:
                want = lons
            elif dim == "values":                               # one horizontal dimension: as many points
                want = None if lats is None or lons is None else lats * lons
            else:
                want = 1
            out.append(sizes[dim] if want is None else max(1, min(int(want), sizes[dim])))
        return out

    def describe(self, plans: dict = None) -> dict:
        info = {"schema": SCHEMA, "preset": self.preset, "keepbits": self.keepbits,
                "chunks": self.chunks if isinstance(self.chunks, str) else list(self.chunks),
                "complevel": self.complevel, "variables": {}}
        for name, plan in (plans or {}).items():
            info["variables"][name] = plan.describe()
        return info

    def tag(self, plans: dict = None) -> str:
        """Text of the global attribute ``aiuq_compression``."""
        return json.dumps(self.describe(plans), sort_keys=True)


def read_tag(text) -> dict:
    """What an ``aiuq_compression`` attribute says, also in the two older forms; {} when it says nothing."""
    if not text:
        return {}
    text = str(text)
    try:
        info = json.loads(text)
        return info if isinstance(info, dict) else {}
    except ValueError:
        pass
    match = LEGACY_BITS.match(text)
    if match:
        return {"schema": 0, "preset": "bits", "keepbits": int(match.group(1)), "chunks": "point", "complevel": 1,
                "legacy": text}
    if text.startswith(LEGACY_MIXED):
        return {"schema": 0, "preset": "mixed", "keepbits": DEFAULT_KEEPBITS, "chunks": "point", "complevel": 1,
                "legacy": text}
    return {}


# ------------------------------------------------------------------------------------------------ encoding
def integer_type(values_max: float) -> str:
    """'i2' when every stored integer fits in 16 bits, else 'i4'."""
    return "i2" if values_max < INT16_LIMIT else "i4"


def to_integers(values, plan: Plan) -> np.ndarray:
    """The whole numbers stored for ``values`` (float64, NaN where there is no value)."""
    return np.round((np.asarray(values, dtype="f8") - plan.offset) / plan.step)


def largest_integer(blocks, plan: Plan) -> float:
    """Largest absolute stored integer over blocks of values (decides between 16 and 32 bits)."""
    largest = 0.0
    for block in blocks:
        ints = to_integers(block, plan)
        if np.isfinite(ints).any():
            largest = max(largest, float(np.nanmax(np.abs(ints))))
    if largest >= -FILL["i4"]:
        raise CompressionError(f"step {plan.step:g} is too small for these values: they do not fit in 32-bit integers")
    return largest


def create_variable(nc, name: str, dims, plan: Plan, scheme: Scheme, source_dtype="f4", largest=0.0):
    """Create the data variable ``name`` in the open netCDF4 dataset ``nc`` as the plan asks.

    ``largest`` (for integers) is the result of :func:`largest_integer`. Values are then written with
    :func:`write_values`; the variable has automatic scaling switched off.
    """
    sizes = {dim: len(nc.dimensions[dim]) for dim in dims}
    if plan.mode == "none":
        kind = np.dtype(source_dtype)
        kind = kind if kind.kind == "f" else np.dtype("f4")
        var = nc.createVariable(name, kind, dims, fill_value=kind.type(np.nan))
    else:
        storage = dict(compression="zlib", complevel=scheme.complevel, shuffle=True,
                       chunksizes=scheme.chunksizes(dims, sizes))
        if plan.mode == "integer":
            kind = integer_type(largest)
            var = nc.createVariable(name, kind, dims, fill_value=FILL[kind], **storage)
            var.setncattr("scale_factor", np.float64(plan.step))
            var.setncattr("add_offset", np.float64(plan.offset))
        elif plan.mode == "bits":
            var = nc.createVariable(name, "f4", dims, fill_value=np.float32(np.nan), significant_digits=plan.keepbits,
                                    quantize_mode="BitRound", **storage)
        else:
            var = nc.createVariable(name, "f4", dims, fill_value=np.float32(np.nan), **storage)
    var.set_auto_maskandscale(False)
    return var


def write_values(var, index, values, plan: Plan) -> None:
    """Store ``values`` (floating point, NaN for no value) in ``var[index]``."""
    if plan.mode == "integer":
        ints = to_integers(values, plan)
        fill = FILL[var.dtype.str[1:]]
        var[index] = np.where(np.isnan(ints), fill, ints).astype(var.dtype)
    else:
        var[index] = np.asarray(values, dtype=var.dtype)


def read_values(var, index=Ellipsis) -> np.ndarray:
    """Values of ``var[index]`` as the readers see them: float64 in the units of the variable, NaN for none."""
    var.set_auto_maskandscale(False)
    raw = np.asarray(var[index])
    if raw.dtype.kind == "f":
        values = raw.astype("f8")
        fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
        if fill is not None and not np.isnan(fill):
            values[raw == fill] = np.nan
        packed = "scale_factor" in var.ncattrs() or "add_offset" in var.ncattrs()
    else:
        values = raw.astype("f8")
        if "_FillValue" in var.ncattrs():
            values[raw == var.getncattr("_FillValue")] = np.nan
        packed = True
    if packed:
        scale = float(var.getncattr("scale_factor")) if "scale_factor" in var.ncattrs() else 1.0
        offset = float(var.getncattr("add_offset")) if "add_offset" in var.ncattrs() else 0.0
        values = values * scale + offset
    return values


def check_values(original, stored, plan: Plan, where="") -> None:
    """Raise unless ``stored`` (as read back) is ``original`` within what the plan allows.

    Same missing values; then exactly equal for 'none', equal to the float32 value for 'lossless', within
    2**-keepbits of it for 'bits', within half a step for 'integer'.
    """
    x, y = np.asarray(original, dtype="f8"), np.asarray(stored, dtype="f8")
    if x.shape != y.shape:
        raise CompressionError(f"{where}shape {y.shape} instead of {x.shape}")
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        raise CompressionError(f"{where}the missing values are not where they were")
    there = ~np.isnan(x)
    if not there.any():
        return
    if plan.mode == "integer":
        error, limit = np.abs(y[there] - x[there]), plan.step * (0.5 + 1e-6)
        if np.any(error > limit):
            raise CompressionError(f"{where}error {float(error.max()):.3g} above half the step ({plan.step:g})")
        return
    if plan.mode == "none":
        if not np.array_equal(x[there], y[there]):
            raise CompressionError(f"{where}values changed although nothing should")
        return
    single = np.asarray(original, dtype="f4").astype("f8")[there]            # the precision the models compute in
    if plan.mode == "lossless":
        if not np.array_equal(single, y[there]):
            raise CompressionError(f"{where}values differ from their float32 value")
        return
    error, limit = np.abs(y[there] - single), 2.0 ** -plan.keepbits * np.abs(single) + 1e-30
    if np.any(error > limit):
        raise CompressionError(f"{where}relative error above 2**-{plan.keepbits}: {float((error / np.maximum(np.abs(single), 1e-30)).max()):.3g}")


def xarray_encoding(name: str, dims, sizes: dict, plan: Plan, scheme: Scheme, largest=0.0) -> dict:
    """The same storage as an ``encoding`` entry for ``xarray.Dataset.to_netcdf`` (netCDF4 engine)."""
    if plan.mode == "none":
        return {}
    encoding = {"zlib": True, "complevel": scheme.complevel, "shuffle": True,
                "chunksizes": tuple(scheme.chunksizes(dims, sizes))}
    if plan.mode == "integer":
        kind = integer_type(largest)
        encoding.update(dtype="int16" if kind == "i2" else "int32", scale_factor=plan.step, add_offset=plan.offset,
                        _FillValue=FILL[kind])
    else:
        encoding.update(dtype="float32", _FillValue=np.float32(np.nan))
        if plan.mode == "bits":
            encoding.update(significant_digits=plan.keepbits, quantize_mode="BitRound")
    return encoding


# ------------------------------------------------------------------------------------------- existing files
def data_variables(nc) -> list:
    """Names of the output variables of a file: along ``valid_time``, two or more dimensions, not bounds."""
    bounds = {v.getncattr("bounds") for v in nc.variables.values() if "bounds" in v.ncattrs()}
    return [name for name, v in nc.variables.items()
            if v.ndim >= 2 and "valid_time" in v.dimensions and name not in nc.dimensions and name not in bounds]


def stored_plan(var) -> Plan:
    """The plan a variable on disk was stored with, read from the variable itself."""
    attrs = var.ncattrs()
    if var.dtype.kind in "iu":
        return Plan("integer", step=float(var.getncattr("scale_factor")) if "scale_factor" in attrs else 1.0,
                    offset=float(var.getncattr("add_offset")) if "add_offset" in attrs else 0.0)
    filters = var.filters() or {}
    if not filters.get("zlib") and var.chunking() == "contiguous":
        return Plan("none")
    if "_QuantizeBitRoundNumberOfSignificantBits" in attrs and var.dtype == np.float32:
        return Plan("bits", keepbits=int(var.getncattr("_QuantizeBitRoundNumberOfSignificantBits")))
    return Plan("lossless") if var.dtype == np.float32 and filters.get("zlib") else Plan("none")


def same_plan(a: Plan, b: Plan) -> bool:
    if a.mode != b.mode:
        return False
    if a.mode == "bits":
        return a.keepbits == b.keepbits
    if a.mode == "integer":
        return bool(np.isclose(a.step, b.step, rtol=1e-9, atol=0) and np.isclose(a.offset, b.offset, rtol=1e-9, atol=1e-12))
    return True


def is_stored_as(var, plan: Plan, scheme: Scheme) -> bool:
    """True when the variable on disk already is what the plan and the scheme ask for (values and chunks)."""
    if not same_plan(stored_plan(var), plan):
        return False
    if plan.mode == "none":
        return True
    filters = var.filters() or {}
    sizes = {dim: var.shape[i] for i, dim in enumerate(var.dimensions)}
    return (bool(filters.get("zlib")) and bool(filters.get("shuffle")) and filters.get("complevel") == scheme.complevel
            and list(var.chunking()) == scheme.chunksizes(var.dimensions, sizes))


def compress_file(path: str, scheme: Scheme, specs: dict = None, force: bool = False, dry: bool = False):
    """Store an existing output file as the scheme asks, in place. Returns ``(status, bytes before, bytes after)``.

    status: 'ok' rewritten; 'skip' already stored that way (also by the earlier scripts); 'kept' stored with
    less precision than asked, which a rewrite cannot give back (``force`` rewrites anyway); 'dry' as 'ok'
    without replacing the file. ``specs`` maps AIUQ-st short names to their definitions (the ``pack`` steps).

    The new file is written beside the old one, read back and compared with it (missing values, coordinates,
    error within what each plan allows), and only then renamed over it.
    """
    specs = specs or {}
    size = os.path.getsize(path)
    with netCDF4.Dataset(path) as src:
        names = data_variables(src)
        if not names:
            raise CompressionError(f"{path}: no output variable along valid_time")
        plans = {name: scheme.plan(name, specs.get(name)) for name in names}
        current = {name: stored_plan(src[name]) for name in names}
        if all(is_stored_as(src[name], plans[name], scheme) for name in names):
            return "skip", size, size
        if not force and any(LOSS_ORDER[current[n].mode] > LOSS_ORDER[plans[n].mode] for n in names):
            return "kept", size, size

        temporary = f"{path}.compressing"
        try:
            with netCDF4.Dataset(temporary, "w", format="NETCDF4") as dst:
                dst.setncatts({k: src.getncattr(k) for k in src.ncattrs() if k != "aiuq_compression"})
                dst.setncattr("aiuq_compression", scheme.tag(plans))
                for name, dim in src.dimensions.items():
                    dst.createDimension(name, None if dim.isunlimited() else len(dim))
                largest = {}
                for name, var in src.variables.items():
                    if name in plans:
                        continue
                    var.set_auto_maskandscale(False)
                    fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
                    out = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill)
                    out.set_auto_maskandscale(False)
                    out.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"})
                    if var.ndim == 0:
                        out.assignValue(var.getValue())
                    else:
                        out[:] = var[:]
                for name in names:
                    var, plan = src[name], plans[name]
                    blocks = _blocks(var)
                    if plan.mode == "integer":
                        largest[name] = largest_integer((read_values(var, index) for index in blocks), plan)
                    kind = var.dtype if var.dtype.kind == "f" else np.dtype("f4")
                    out = create_variable(dst, name, var.dimensions, plan, scheme, kind, largest.get(name, 0.0))
                    out.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k not in ENCODING_ATTRS})
                    for index in blocks:
                        write_values(out, index, read_values(var, index), plan)
            with netCDF4.Dataset(temporary) as dst:
                for name, var in src.variables.items():
                    if name in plans:
                        for index in _blocks(var):
                            check_values(read_values(var, index), read_values(dst[name], index), plans[name],
                                         f"{os.path.basename(path)} {name}: ")
                    else:
                        var.set_auto_maskandscale(False)
                        other = dst[name]
                        other.set_auto_maskandscale(False)
                        if not np.array_equal(np.asarray(var[...]), np.asarray(other[...]), equal_nan=var.dtype.kind == "f"):
                            raise CompressionError(f"{os.path.basename(path)}: coordinate {name} changed")
            new_size = os.path.getsize(temporary)
            if dry:
                return "dry", size, new_size
            os.replace(temporary, path)
            return "ok", size, new_size
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


def _blocks(var, target_bytes=256 * 2 ** 20) -> list:
    """Index tuples that cover a variable in blocks of valid times of about ``target_bytes`` (as float64)."""
    axis = var.dimensions.index("valid_time")
    per_time = max(1, int(np.prod([n for i, n in enumerate(var.shape) if i != axis], dtype="i8")) * 8)
    step = max(1, int(target_bytes // per_time))
    length = var.shape[axis]
    # Slices end at the last valid time: along an unlimited dimension (cdo writes one) a longer slice would
    # ask for that many values when writing.
    return [tuple(slice(start, min(start + step, length)) if i == axis else slice(None) for i in range(var.ndim))
            for start in range(0, length, step)]
