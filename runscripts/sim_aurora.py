#!/usr/bin/env python3
import os
import logging
from datetime import datetime, timezone
import json
import ast

import numpy as np
import xarray as xr
import torch

from aurora import Batch, Metadata
from aurora import Aurora, rollout

from AIUQst_lib.functions import parse_arguments, read_config, define_variables
from anemoi.inference.checkpoint import Checkpoint  # se serve per compatibilità col tuo env
# riuso alcune funzioni utili dallo script AIFS (adattale se le hai in un modulo comune)
import re

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("run_aurora")

LEVEL_VAR_RE = re.compile(r"^(?P<var>[a-zA-Z0-9]+)_(?P<level>\d+)$")


def add_gaussian_noise_da(
    da: xr.DataArray,
    frac: float = 0.1,
    seed: int | None = None,
) -> xr.DataArray:
    """
    Aggiunge rumore gaussiano N(0, sigma_noise) dove sigma_noise = frac * std_originale(da).
    Rumore indipendente per ogni elemento (punto griglia / livello / time ecc.).
    """
    # calcolo std originale (ignora NaN se presenti)
    std_orig = float(da.std(skipna=True).values)

    # se il campo è costante o std=0, non aggiungo rumore
    if not np.isfinite(std_orig) or std_orig == 0.0:
        return da

    sigma = frac * std_orig

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=da.shape)

    # preserva coords/dims grazie a xarray
    return da + xr.DataArray(noise, dims=da.dims, coords=da.coords)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return list(v)
                return [str(v)]
            except Exception:
                pass
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x)]

def guess_var_names_from_model_card(model):
    """
    Legge model_card e restituisce mapping era->ifs e lista livelli di pressione.
    Si aspetta che read_model_card sia definita nello stesso env (come nello script AIFS).
    """
    try:
        from __main__ import read_model_card  # prova a usare la funzione già definita
    except Exception:
        # Se non è nel main, import locale (assumi che esista nello stesso package)
        try:
            from your_module import read_model_card  # alternativa, modifica se serve
        except Exception:
            raise RuntimeError("read_model_card non trovato: assicurati sia disponibile")

    vars_era, vars_ifs, pressure_levels, era_to_ifs, ifs_to_era = read_model_card(model)
    return vars_era, vars_ifs, pressure_levels, era_to_ifs, ifs_to_era

def select_var(ds: xr.Dataset, possible_names):
    """
    Dato un dataset xarray e una lista di nomi candidati, ritorna il primo presente.
    """
    for n in possible_names:
        if n in ds.data_vars:
            return ds[n]
    return None

def build_batch_from_zarr(INI_DATA_PATH: str, noise_frac: float = 0.1):
    """
    Costruisce un Batch Aurora con chiavi canoniche:
      surf_vars: 2t, 10u, 10v, msl     -> (B, T, H, W) con B=1, T=2
      static_vars: z, slt, lsm         -> (H, W)
      atmos_vars: t, u, v, q, z        -> (B, T, L, H, W) con B=1, T=2
    e Metadata coerente con la doc.

    Fix importanti:
      - lat strettamente decrescente (sort)
      - selezione variabili basata anche su dims (evita z statico scambiato per z atmos)
      - fail-fast se mancano variabili essenziali
    """
    ds = xr.open_zarr(INI_DATA_PATH, chunks=None)

    # ----------------------------
    # coord detection
    # ----------------------------
    # time
    if "valid_time" in ds.coords:
        time_coord = "valid_time"
    elif "time" in ds.coords:
        time_coord = "time"
    else:
        raise RuntimeError(
            f"[{INI_DATA_PATH}] Nessuna coord temporale trovata (valid_time/time). "
            f"Coords: {list(ds.coords)}"
        )

    # lat/lon
    if "latitude" in ds.coords:
        lat_name = "latitude"
    elif "lat" in ds.coords:
        lat_name = "lat"
    else:
        lat_candidates = [c for c in ds.coords if "lat" in c.lower()]
        if not lat_candidates:
            raise RuntimeError(f"[{INI_DATA_PATH}] Nessuna coord lat trovata. Coords: {list(ds.coords)}")
        lat_name = lat_candidates[0]

    if "longitude" in ds.coords:
        lon_name = "longitude"
    elif "lon" in ds.coords:
        lon_name = "lon"
    else:
        lon_candidates = [c for c in ds.coords if "lon" in c.lower()]
        if not lon_candidates:
            raise RuntimeError(f"[{INI_DATA_PATH}] Nessuna coord lon trovata. Coords: {list(ds.coords)}")
        lon_name = lon_candidates[0]

    # level coord (pressure levels)
    level_coord = None
    for cand in ["pressure_level", "level", "isobaricInhPa", "plev"]:
        if cand in ds.coords:
            level_coord = cand
            break

    # ----------------------------
    # enforce strictly decreasing latitude
    # ----------------------------
    lat_vals = ds.coords[lat_name].values
    if lat_vals[0] < lat_vals[-1]:
        LOG.info("Latitudine crescente nel file: riordino in senso decrescente per Aurora.")
        ds = ds.sortby(lat_name, ascending=False)
        lat_vals = ds.coords[lat_name].values

    if not np.all(np.diff(lat_vals) < 0):
        raise ValueError(
            f"[{INI_DATA_PATH}] Latitudes not strictly decreasing after sort. "
            f"Controlla duplicati / coordinate '{lat_name}'."
        )

    # ----------------------------
    # require at least 2 time points
    # ----------------------------
    ntime = ds.sizes.get(time_coord, None)
    if ntime is None or ntime < 2:
        raise RuntimeError(
            f"[{INI_DATA_PATH}] Servono almeno 2 timepoint su '{time_coord}'. size={ntime}"
        )

    # take first 2 times
    ds2 = ds.isel({time_coord: slice(0, 2)})

    # lat/lon copy (writable)
    lat = np.array(ds.coords[lat_name].values, copy=True)
    lon = np.array(ds.coords[lon_name].values, copy=True)

    # ----------------------------
    # helpers: choose variable by name candidates + required dims
    # ----------------------------
    def select_var_by_dims(ds_: xr.Dataset, candidates: list[str], required_dims: set[str]) -> xr.DataArray | None:
        """
        Trova il primo candidato presente in ds_.data_vars che contenga almeno required_dims.
        """
        for c in candidates:
            if c in ds_.data_vars:
                da = ds_[c]
                if required_dims.issubset(set(da.dims)):
                    return da
        return None

    def select_any_by_dims(ds_: xr.Dataset, candidates: list[str], required_dims: set[str]) -> xr.DataArray | None:
        """
        Come sopra, ma se un candidato esiste ma dims non matchano, continua a cercare.
        """
        return select_var_by_dims(ds_, candidates, required_dims)

    def to_torch_surf(da: xr.DataArray) -> torch.Tensor:
        # expect (time, lat, lon) with 2 times; add batch dim -> (1,2,H,W)
        da2 = da
        if noise_frac and noise_frac > 0:
            da2 = add_gaussian_noise_da(da2, frac=noise_frac, seed=None)
        arr = np.array(da2.values, copy=True)
        if arr.ndim != 3:
            raise RuntimeError(f"Surface var '{da.name}' ndim={arr.ndim}, atteso 3 (time,lat,lon). dims={da.dims}")
        return torch.from_numpy(arr[None].astype(np.float32, copy=False))

    def to_torch_atmos(da: xr.DataArray) -> torch.Tensor:
        # expect (time, level, lat, lon) with 2 times; add batch dim -> (1,2,L,H,W)
        da2 = da
        if noise_frac and noise_frac > 0:
            da2 = add_gaussian_noise_da(da2, frac=noise_frac, seed=None)
        arr = np.array(da2.values, copy=True)
        if arr.ndim != 4:
            raise RuntimeError(f"Atmos var '{da.name}' ndim={arr.ndim}, atteso 4 (time,level,lat,lon). dims={da.dims}")
        return torch.from_numpy(arr[None].astype(np.float32, copy=False))

    def to_torch_static(da: xr.DataArray) -> torch.Tensor:
        # static can be (lat,lon) or (time,lat,lon) -> take first time
        arr = np.array(da.values, copy=True)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError(f"Static var '{da.name}' ndim={arr.ndim}, atteso 2 (lat,lon) (o 3 con time). dims={da.dims}")
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    # ----------------------------
    # candidates (names only)
    # ----------------------------
    surf_candidates = {
        "2t": ["t2m", "2t", "2m_temperature", "t_2m", "t2m_surface", "TMP_2m", "T2M"],
        "10u": ["u10", "10u", "u_10m", "10m_u_component_of_wind", "U10"],
        "10v": ["v10", "10v", "v_10m", "10m_v_component_of_wind", "V10"],
        "msl": ["msl", "mslp", "mean_sea_level_pressure", "MSL", "prmsl"],
    }

    static_candidates = {
        "z": ["z", "surface_geopotential", "geopotential_surface", "orography", "z_surf"],
        "slt": ["slt", "soil_layer_thickness", "soil_thickness"],
        "lsm": ["lsm", "land_sea_mask"],
    }

    atmos_candidates = {
        "t": ["t", "temperature", "air_temperature"],
        "u": ["u", "u_component_of_wind", "u_wind"],
        "v": ["v", "v_component_of_wind", "v_wind"],
        "q": ["q", "specific_humidity", "sphum"],
        # IMPORTANT: do NOT blindly take "z" unless it has (time,level,lat,lon)
        "z": ["z", "geopotential", "gh", "geopotential_height"],
    }

    # ----------------------------
    # build surf_vars (must exist)
    # ----------------------------
    surf_vars: dict[str, torch.Tensor] = {}
    for key, cands in surf_candidates.items():
        da = select_any_by_dims(ds2, cands, required_dims={time_coord, lat_name, lon_name})
        if da is None:
            LOG.warning("Surface var '%s' non trovata con dims (%s,%s,%s). Candidati: %s",
                        key, time_coord, lat_name, lon_name, cands)
            continue
        surf_vars[key] = to_torch_surf(da)

    if len(surf_vars) == 0:
        LOG.error("Nessuna surface var trovata. data_vars (prime 120): %s", list(ds2.data_vars)[:120])
        raise RuntimeError(
            "surf_vars è vuoto: devi mappare almeno una variabile di superficie (idealmente 2t/10u/10v/msl)."
        )

    # ----------------------------
    # build static_vars
    # ----------------------------
    static_vars: dict[str, torch.Tensor] = {}
    # per static: accetto (lat,lon) oppure (time,lat,lon)
    for key, cands in static_candidates.items():
        da = None
        # prima prova 2D
        da = select_any_by_dims(ds, cands, required_dims={lat_name, lon_name})
        if da is None:
            LOG.warning("Static var '%s' non trovata. Candidati: %s", key, cands)
            continue
        static_vars[key] = to_torch_static(da)

    # ----------------------------
    # build atmos_vars (requires level coord)
    # ----------------------------
    if level_coord is None:
        raise RuntimeError(
            f"[{INI_DATA_PATH}] Nessuna coord livelli trovata (pressure_level/level/...). "
            f"Coords: {list(ds.coords)}"
        )

    atmos_vars: dict[str, torch.Tensor] = {}
    for key, cands in atmos_candidates.items():
        da = select_any_by_dims(ds2, cands, required_dims={time_coord, level_coord, lat_name, lon_name})
        if da is None:
            # nota: se esiste un "z" 2D, qui non verrà preso (giusto!)
            LOG.warning("Atmos var '%s' non trovata con dims (%s,%s,%s,%s). Candidati: %s",
                        key, time_coord, level_coord, lat_name, lon_name, cands)
            continue
        atmos_vars[key] = to_torch_atmos(da)

    # ----------------------------
    # Metadata
    # ----------------------------
    vt = ds2[time_coord].values
    time_for_metadata = np.datetime64(vt[1]).astype("datetime64[s]").tolist()

    atmos_levels_tuple = tuple(int(x) for x in np.array(ds.coords[level_coord].values).tolist())

    metadata = Metadata(
        lat=torch.from_numpy(np.array(lat, copy=True)),
        lon=torch.from_numpy(np.array(lon, copy=True)),
        time=(time_for_metadata,),
        atmos_levels=atmos_levels_tuple,
    )

    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )

    # Log riepilogo (utile)
    LOG.info("Batch costruito. surf_vars=%s static_vars=%s atmos_vars=%s",
             list(batch.surf_vars.keys()), list(batch.static_vars.keys()), list(batch.atmos_vars.keys()))

    return batch, metadata, ds2
    
def preds_to_xarray(preds, metadata: Metadata, prefer_levels_from_meta: bool = True) -> xr.Dataset:
    """
    Converte output di `aurora.rollout()` in xarray.Dataset.

    In molte versioni di Aurora, `rollout()` yielda oggetti `Batch` (non Tensor).
    Ogni Batch contiene dict:
      - surf_vars: dict[str, Tensor] tipicamente shape (B, T, H, W) oppure (B, H, W)
      - atmos_vars: dict[str, Tensor] tipicamente shape (B, T, L, H, W) oppure (B, L, H, W)
    Nel rollout, spesso ad ogni step i tensori sono "single-time" (T=1) o senza T.
    Questa funzione normalizza e concatena sugli step -> dim 'valid_time'.

    Ritorna Dataset con:
      - valid_time: index steps
      - latitude, longitude
      - level (per atmos vars)
    """
    if not preds:
        raise RuntimeError("preds è vuoto: rollout non ha prodotto output.")

    # coords
    lat = metadata.lat.detach().cpu().numpy() if torch.is_tensor(metadata.lat) else np.asarray(metadata.lat)
    lon = metadata.lon.detach().cpu().numpy() if torch.is_tensor(metadata.lon) else np.asarray(metadata.lon)

    if prefer_levels_from_meta and getattr(metadata, "atmos_levels", None):
        levels = np.asarray(list(metadata.atmos_levels), dtype=int)
    else:
        levels = None

    def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _norm_surf(t: torch.Tensor) -> np.ndarray:
        """
        Ritorna array shape (H,W)
        accetta:
          (B,T,H,W) -> prende B=0, T=-1
          (B,H,W)   -> prende B=0
          (T,H,W)   -> prende T=-1
          (H,W)     -> ok
        """
        a = _tensor_to_np(t)
        if a.ndim == 4:       # (B,T,H,W)
            return a[0, -1]
        elif a.ndim == 3:     # (B,H,W) o (T,H,W)
            return a[0] if (a.shape[0] in (1,)) else a[-1]
        elif a.ndim == 2:
            return a
        else:
            raise RuntimeError(f"Surf tensor ndim={a.ndim} non supportato. shape={a.shape}")

    def _norm_atmos(t: torch.Tensor) -> np.ndarray:
        """
        Ritorna array shape (L,H,W)
        accetta:
          (B,T,L,H,W) -> prende B=0, T=-1
          (B,L,H,W)   -> prende B=0
          (T,L,H,W)   -> prende T=-1
          (L,H,W)     -> ok
        """
        a = _tensor_to_np(t)
        if a.ndim == 5:       # (B,T,L,H,W)
            return a[0, -1]
        elif a.ndim == 4:     # (B,L,H,W) oppure (T,L,H,W)
            return a[0] if (a.shape[0] in (1,)) else a[-1]
        elif a.ndim == 3:
            return a
        else:
            raise RuntimeError(f"Atmos tensor ndim={a.ndim} non supportato. shape={a.shape}")

    # accumulators: var -> list of arrays per step
    surf_acc: dict[str, list[np.ndarray]] = {}
    atmos_acc: dict[str, list[np.ndarray]] = {}

    for i, pb in enumerate(preds):
        if not hasattr(pb, "surf_vars") or not hasattr(pb, "atmos_vars"):
            raise RuntimeError(f"Elemento preds[{i}] non è un Batch Aurora (manca surf_vars/atmos_vars). type={type(pb)}")

        # surf
        for k, v in pb.surf_vars.items():
            if v is None:
                continue
            surf_acc.setdefault(k, []).append(_norm_surf(v))

        # atmos
        for k, v in pb.atmos_vars.items():
            if v is None:
                continue
            atmos_acc.setdefault(k, []).append(_norm_atmos(v))

    # build Dataset
    data_vars = {}

    # surf vars -> (valid_time, latitude, longitude)
    for k, seq in surf_acc.items():
        arr = np.stack(seq, axis=0)  # (S,H,W)
        data_vars[k] = (("valid_time", "latitude", "longitude"), arr)

    # atmos vars -> (valid_time, level, latitude, longitude)
    for k, seq in atmos_acc.items():
        arr = np.stack(seq, axis=0)  # (S,L,H,W)
        # infer levels if not provided
        if levels is None:
            levels = np.arange(arr.shape[1], dtype=int)
        data_vars[k] = (("valid_time", "level", "latitude", "longitude"), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "valid_time": np.arange(len(preds), dtype=int),
            "latitude": lat,
            "longitude": lon,
            **({"level": levels} if levels is not None else {}),
        },
    )

    return ds


def rename_outputs_to_friendly(ds: xr.Dataset) -> xr.Dataset:
    ren = {
        # pressure-level base vars
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "q": "specific_humidity",
        "z": "geopotential",
        "w": "vertical_velocity",
        # surface / near-surface
        "2t": "2m_temperature",
        "2d": "2m_dewpoint_temperature",
        "10u": "10m_u_component_of_wind",
        "10v": "10m_v_component_of_wind",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "msl": "mean_sea_level_pressure",
        "sp": "surface_pressure",
        "tcw": "total_column_water",
        "skt": "skin_temperature",
        "sst": "sea_surface_temperature",
        "lsm": "land_sea_mask",
        "sdor": "standard_deviation_of_orography",
        "slor": "slope_of_sub_gridscale_orography",
        "stl1": "soil_temperature_level_1",
        "stl2": "soil_temperature_level_2",
    }
    present = {k: v for k, v in ren.items() if k in ds.data_vars}
    if present:
        ds = ds.rename(present)
    return ds

def main():
    args = parse_arguments()
    config = read_config(args.config)

    (model_checkpoint, INI_DATA_PATH, start_time, end_time,
     data_inner_steps, inner_steps, rng_key, output_path, output_vars, ics_temp_dir, static_data) = define_variables(config)

    os.makedirs(output_path, exist_ok=True)

    start_date = datetime.strptime(start_time, "%Y-%m-%d")
    end_date = datetime.strptime(end_time, "%Y-%m-%d")
    days_to_run = (end_date - start_date).days + 1
    outer_steps = days_to_run * 24

    # --- Build batch da INI zarr ---
    LOG.info("Costruisco batch da %s", INI_DATA_PATH)
    batch, metadata, ds2 = build_batch_from_zarr(INI_DATA_PATH)

    # --- Carica modello Aurora ---
    LOG.info("Carico modello Aurora da checkpoint: %s", model_checkpoint)
    model = Aurora(use_lora=False)

    ckpt_path = os.path.expanduser(str(model_checkpoint))

    if os.path.isfile(ckpt_path):
        LOG.info("Uso checkpoint locale: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # case più comune: Lightning-style
        state = ckpt.get("state_dict", ckpt)

        # pulizia prefissi frequenti
        cleaned = {}
        for k, v in state.items():
            kk = k
            if kk.startswith("model."):
                kk = kk[len("model."):]
            if kk.startswith("net."):
                kk = kk[len("net."):]
            if kk.startswith("module."):
                kk = kk[len("module."):]
            cleaned[kk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        LOG.info("Checkpoint caricato. missing=%d unexpected=%d", len(missing), len(unexpected))

    else:
        # Qui va usato solo se model_checkpoint è davvero un repo HF, es: "microsoft/aurora"
        LOG.info("Checkpoint non è un file locale. Provo HuggingFace Hub: repo=%s", ckpt_path)
        model.load_checkpoint(repo=ckpt_path)  # NOTA: qui 'repo' deve essere user/repo

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # porta batch tensors su device se necessario
    def move_batch_to_device(b, device):
        for k, v in b.surf_vars.items():
            b.surf_vars[k] = v.to(device)
        for k, v in b.static_vars.items():
            b.static_vars[k] = v.to(device)
        for k, v in b.atmos_vars.items():
            b.atmos_vars[k] = v.to(device)
        # metadata tensors lasciati su CPU (lat/lon) o convertiti
        return b

    batch = move_batch_to_device(batch, device)

    # --- Rollout / inferenza ---
    LOG.info("Eseguo rollout per %d step", outer_steps)
    with torch.inference_mode():
        preds = list(rollout(model, batch, steps=int(outer_steps)))

    # libera GPU
    model = model.to("cpu")

    # --- Converti preds in xarray.Dataset ---
    LOG.info("Converto predizioni in xarray")
    try:
        ds_preds = preds_to_xarray(preds, metadata)
    except Exception as e:
        LOG.exception("Errore conversione preds -> xarray: %s", e)
        raise

    # Se conosciam lat/lon/level dal metadata/ds2, assegniamoli
    if "latitude" not in ds_preds.coords and hasattr(metadata, "lat"):
        ds_preds = ds_preds.assign_coords(latitude=("latitude", metadata.lat.numpy()))
    if "longitude" not in ds_preds.coords and hasattr(metadata, "lon"):
        ds_preds = ds_preds.assign_coords(longitude=("longitude", metadata.lon.numpy()))
    if "level" not in ds_preds.coords and hasattr(metadata, "atmos_levels") and metadata.atmos_levels:
        ds_preds = ds_preds.assign_coords(level=("level", np.array(metadata.atmos_levels)))

    # rename friendly
    ds_preds = rename_outputs_to_friendly(ds_preds)

    # filtra output_vars se richiesto
    output_vars = _as_list(output_vars)
    if output_vars:
        keep = [v for v in output_vars if v in ds_preds.data_vars]
        if not keep:
            LOG.warning("Nessuna delle output_vars richieste è presente nelle predizioni")
        else:
            ds_preds = ds_preds[keep]

    # Build valid_time/step like nello script AIFS: assumo che iniziale sia ds2.time.isel(0)
    if "time" in ds2.coords:
        initial_time = ds2["time"].isel(time=0).values
    elif "valid_time" in ds2.coords:
        initial_time = ds2["valid_time"].isel(valid_time=0).values
    else:
        initial_time = np.datetime64(datetime.utcnow())

    delta_t = np.timedelta64(int(inner_steps), "h")
    n_out = ds_preds.sizes.get("time", ds_preds.sizes.get("valid_time", 0))
    valid_time = initial_time + np.arange(n_out) * delta_t
    # se ds_preds ha dim "time", rinominiamo in "valid_time"
    if "time" in ds_preds.dims:
        ds_preds = ds_preds.rename({"time": "valid_time"})
    ds_preds = ds_preds.assign_coords(valid_time=("valid_time", valid_time))
    # assign time (simulated origin)
    ds_preds = ds_preds.assign_coords(time=initial_time)
    ds_preds = ds_preds.assign_coords(step=("valid_time", np.arange(n_out) * delta_t))

    # Salva netcdf
    final_file = os.path.join(output_path, f"aurora_state-{start_time}-{end_time}-{rng_key}_regular025.nc")
    LOG.info("Salvo NetCDF in %s", final_file)
    ds_preds.to_netcdf(final_file)
    LOG.info("Wrote: %s", final_file)

if __name__ == "__main__":
    main()