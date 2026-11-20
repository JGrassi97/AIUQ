"""EC-Earth3 (CMIP6, ESGF) ICs: file search, time sampling, profiles, derivations, assembly."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
import download_ec_earth3 as ec  # noqa: E402

DAY = np.datetime64("2020-01-01T00", "ns")
HOUR = np.timedelta64(1, "h")
TIMES = ec.ic_times("2020-01-01")


def card():
    return yaml.safe_load((ROOT / "conf/cards/ics/ec-earth3_card.yml").read_text())


def standard():
    return {
        "variables": yaml.safe_load((ROOT / "conf/AIUQ-st/v100/variables.yml").read_text()),
        "pressure_levels": yaml.safe_load((ROOT / "conf/AIUQ-st/v100/pressure_levels.yml").read_text()),
    }


# ---------------------------------------------------------------- experiment, times, file names

def test_experiment_follows_the_date_unless_overridden():
    esgf = card()["esgf"]
    assert ec.experiment_for("2014-12-31", esgf) == "historical"
    assert ec.experiment_for("2015-01-01", esgf) == esgf["scenario"]
    assert ec.experiment_for("2050-06-01", esgf, "ssp245") == "ssp245"


def test_unresolved_autosubmit_placeholders_are_unset():
    assert ec._setting({"IC_MEMBER": "%MODEL.ICS_MEMBER%"}, "IC_MEMBER") is None
    assert ec._setting({"IC_MEMBER": ""}, "IC_MEMBER") is None
    assert ec._setting({"IC_MEMBER": "r1i1p1f1"}, "IC_MEMBER") == "r1i1p1f1"


def test_ic_times_and_windows():
    assert list(TIMES) == [DAY, DAY + 6 * HOUR]
    assert ec.time_window("point", TIMES) == (DAY, DAY + 6 * HOUR)
    assert ec.time_window("mean", TIMES) == (DAY - 24 * HOUR, DAY + 30 * HOUR)


def test_file_periods():
    first, last = ec.file_period("ta_6hrPlevPt_EC-Earth3_ssp585_r4i1p1f1_gr_202001010000-202012311800.nc")
    assert (first, last) == (np.datetime64("2020-01-01T00:00"), np.datetime64("2020-12-31T18:00"))
    first, last = ec.file_period("clt_day_EC-Earth3_ssp585_r4i1p1f1_gr_20200101-20201231.nc")
    assert last == np.datetime64("2020-12-31T23:59:59")
    first, last = ec.file_period("tsl_Lmon_EC-Earth3_historical_r4i1p1f1_gr_201401-201412.nc")
    assert last == np.datetime64("2014-12-31T23:59:59")
    assert ec.file_period("orog_fx_EC-Earth3_historical_r1i1p1f1_gr.nc") is None


# ---------------------------------------------------------------- ESGF search and replicas

def doc(title, node, version="20200425", opendap=True):
    urls = [f"https://{node}/thredds/fileServer/x/{title}|application/netcdf|HTTPServer"]
    if opendap:
        urls.append(f"https://{node}/thredds/dodsC/x/{title}.html|application/opendap-html|OPENDAP")
    return {"title": title, "data_node": node, "version": version, "url": urls}


def test_candidates_keep_overlapping_latest_files_and_order_replicas():
    docs = [
        doc("ta_6hrPlevPt_EC-Earth3_ssp585_r4i1p1f1_gr_201901010000-201912311800.nc", "a.org"),
        doc("ta_6hrPlevPt_EC-Earth3_ssp585_r4i1p1f1_gr_202001010000-202012311800.nc", "b.org"),
        doc("ta_6hrPlevPt_EC-Earth3_ssp585_r4i1p1f1_gr_202001010000-202012311800.nc", "a.org"),
        doc("ta_6hrPlevPt_EC-Earth3_ssp585_r4i1p1f1_gr_202001010000-202012311800.nc", "c.org", "20190101"),
        doc("ta_6hrPlevPt_EC-Earth3_ssp585_r4i1p1f1_gr_202001010000-202012311800.nc", "d.org", opendap=False),
    ]
    files = ec.opendap_candidates(docs, (DAY, DAY + 6 * HOUR), ["a.org", "b.org"])
    assert len(files) == 1
    title, urls = files[0]
    assert title.endswith("202001010000-202012311800.nc")
    assert urls == [f"https://a.org/thredds/dodsC/x/{title}", f"https://b.org/thredds/dodsC/x/{title}"]
    # A window across the new year needs both yearly files.
    across = ec.opendap_candidates(docs, (DAY - 24 * HOUR, DAY + 30 * HOUR), [])
    assert [t[-28:-3] for t, _ in across] == ["201901010000-201912311800", "202001010000-202012311800"]


def test_search_falls_back_to_the_next_index_and_pages():
    esgf = {**card()["esgf"], "indexes": ["https://broken", "https://ok"]}
    pages = {0: [doc("f1.nc", "a"), doc("f2.nc", "a")], 2: [doc("f3.nc", "a")]}
    urls = []

    def fetch(url):
        urls.append(url)
        if url.startswith("https://broken"):
            raise RuntimeError("422")
        offset = int(url.split("offset=")[1].split("&")[0])
        return {"response": {"numFound": 3, "docs": pages.get(offset, [])}}

    docs = ec.search_files(esgf, "ssp585", "3hr", "tas", fetch=fetch)
    assert [d["title"] for d in docs] == ["f1.nc", "f2.nc", "f3.nc"]
    assert not any("fields=" in u for u in urls)              # the ORNL bridge rejects it (HTTP 422)


def field(times, value=1.0, name="tas", levels=None):
    lat, lon = [-45.0, 45.0], [0.0, 180.0]
    dims, shape = ("time", "lat", "lon"), (len(times), 2, 2)
    coords = {"time": times, "lat": lat, "lon": lon}
    if levels is not None:
        dims, shape = ("time", "plev", "lat", "lon"), (len(times), len(levels), 2, 2)
        coords["plev"] = np.asarray(levels, dtype=float) * 100.0
    data = np.full(shape, value, dtype="float32") + np.arange(len(times)).reshape((-1,) + (1,) * (len(shape) - 1))
    return xr.Dataset({name: (dims, data)}, coords=coords)


def test_read_field_uses_the_next_replica_and_joins_files():
    year_end = np.array(["2019-12-31T21"], dtype="datetime64[ns]")
    year_start = np.array(["2020-01-01T03", "2020-01-01T09"], dtype="datetime64[ns]")
    datasets = {
        "https://a/f2019": field(year_end, 10.0),
        "https://b/f2020": field(year_start, 20.0),
    }

    def opener(url, **_):
        if url not in datasets:
            raise OSError("NetCDF: file not found")
        return datasets[url]

    def search(*_):
        return [{"title": "tas_6hrPlev_x_201901010300-201912312100.nc", "version": "1", "data_node": "a",
                 "url": ["https://a/f2019.html|x|OPENDAP"]},
                {"title": "tas_6hrPlev_x_202001010300-202012312100.nc", "version": "1", "data_node": "dead",
                 "url": ["https://dead/f2020.html|x|OPENDAP"]},
                {"title": "tas_6hrPlev_x_202001010300-202012312100.nc", "version": "1", "data_node": "b",
                 "url": ["https://b/f2020.html|x|OPENDAP"]}]

    esgf = {**card()["esgf"], "preferred_nodes": ["dead", "a", "b"]}
    da, used = ec.read_field(esgf, "ssp585", "6hrPlev", "tas", ec.time_window("mean", TIMES), search, opener)
    assert used == ["https://a/f2019", "https://b/f2020"]
    assert da.sizes["time"] == 3 and set(da.dims) == {"time", "latitude", "longitude"}
    with pytest.raises(ec.NotPublished):
        ec.read_field(esgf, "ssp585", "6hrPlev", "tas",
                      (np.datetime64("2030-01-01", "ns"), np.datetime64("2030-01-02", "ns")), search, opener)


# ---------------------------------------------------------------- time sampling

def test_point_values_need_exact_time_stamps():
    da = ec._standardize(field(TIMES)["tas"])
    assert ec.at_instants(da, TIMES, "point").time.values.tolist() == TIMES.tolist()
    with pytest.raises(ec.NotPublished):
        ec.at_instants(da, TIMES + 3 * HOUR, "point")


def test_means_are_interpolated_in_time_and_clamped_at_the_edges():
    centres = np.array([DAY - 12 * HOUR, DAY + 12 * HOUR], dtype="datetime64[ns]")
    da = ec._standardize(field(centres, 0.0)["tas"])            # 0 then 1
    out = ec.at_instants(da, TIMES, "mean")
    np.testing.assert_allclose(out.isel(latitude=0, longitude=0).values, [0.5, 0.75])
    late = ec.at_instants(da, np.array([DAY + 20 * HOUR], dtype="datetime64[ns]"), "mean")
    assert late.isel(latitude=0, longitude=0).values.tolist() == [1.0]


# ---------------------------------------------------------------- profiles

def column(levels, values, times=TIMES):
    values = np.asarray(values, dtype="float32")
    return xr.DataArray(np.broadcast_to(values[None, :, None, None], (len(times), len(levels), 1, 1)).copy(),
                        dims=("time", "level", "latitude", "longitude"),
                        coords={"time": times, "level": np.asarray(levels, dtype=float),
                                "latitude": [0.0], "longitude": [0.0]})


def test_anomaly_weights_interpolate_hold_and_taper():
    target = np.array([1000., 925., 700., 500., 250., 50., 10., 5., 1.])
    w = ec.anomaly_weights(target, np.array([925., 500., 50.]), taper_decades=1.0)
    np.testing.assert_allclose(w.sum(axis=1)[:6], 1.0)
    assert w[0].tolist() == [1.0, 0.0, 0.0]                     # 1000 hPa: held from 925 hPa
    assert w[3].tolist() == [0.0, 1.0, 0.0]                     # on an anchor
    assert 0 < w[2, 0] < 1 and 0 < w[2, 1] < 1                  # 700 hPa: between 925 and 500
    np.testing.assert_allclose(w[6], [0.0, 0.0, 1.0 - np.log10(5.0)])   # 10 hPa: tapered
    assert w[8].tolist() == [0.0, 0.0, 0.0]                     # 1 hPa: daily mean only


def test_anchored_profile_is_exact_on_the_six_hourly_levels():
    daily_levels = [1000., 925., 850., 700., 500., 250., 50., 10., 1.]
    daily = column(daily_levels, [280, 275, 270, 260, 250, 225, 210, 220, 260])
    instant = column([925., 500., 50.], [277, 251, 214])
    out = ec.anchor_profile(daily, instant, 1.0).isel(time=0, latitude=0, longitude=0)
    np.testing.assert_allclose(out.sel(level=[925., 500., 50.]).values, [277, 251, 214], atol=1e-4)
    assert float(out.sel(level=1000.)) == pytest.approx(282.0)  # +2 K held below 925 hPa
    assert float(out.sel(level=1.)) == pytest.approx(260.0)     # taper reached zero
    assert ec.anchor_profile(daily, None) is daily


# ---------------------------------------------------------------- fields and derivations

def test_sea_ice_fraction_detection():
    da = xr.DataArray([0.0, 0.5, 0.99])
    np.testing.assert_allclose(ec.to_fraction(da), [0.0, 0.5, 0.99])
    np.testing.assert_allclose(ec.to_fraction(da * 100.0), [0.0, 0.5, 0.99])


def test_fill_nearest_fills_sea_points():
    da = xr.DataArray(np.array([[[1.0, np.nan, np.nan, 4.0]]]), dims=("time", "latitude", "longitude"))
    assert ec.fill_nearest(da).values.ravel().tolist() == [1.0, 1.0, 4.0, 4.0]


def test_surface_pressure_dewpoint_and_column_water():
    msl, t2m = xr.DataArray([101325.0, 101325.0]), xr.DataArray([288.0, 288.0])
    sp = ec.surface_pressure(msl, t2m, xr.DataArray([0.0, 1000.0 * ec.G]))
    assert float(sp[0]) == 101325.0 and 0.87 < float(sp[1] / msl[1]) < 0.9
    # Saturated air at 280 K: the dewpoint is 280 K.
    es = 611.21 * np.exp(17.502 * (280.0 - 273.16) / (280.0 - 32.19))
    q = 0.622 * es / (1e5 - 0.378 * es)
    assert float(ec.dewpoint(xr.DataArray(q), xr.DataArray(1e5), xr.DataArray(300.0))) == pytest.approx(280.0, abs=0.01)
    assert float(ec.dewpoint(xr.DataArray(q), xr.DataArray(1e5), xr.DataArray(279.0))) == 279.0
    levels = [1, 10, 100, 500, 850, 1000]
    q = xr.DataArray(np.full((len(levels), 2), 0.01), dims=("level", "x"), coords={"level": levels})
    tcwv = ec.column_integral(q, xr.DataArray([100000.0, 50000.0], dims="x"))
    np.testing.assert_allclose(tcwv, [0.01 * (1e5 - 100) / ec.G, 0.01 * (5e4 - 100) / ec.G], rtol=1e-5)


def test_cloud_layers_reproduce_total_cover():
    tcc = xr.DataArray([0.0, 0.3, 0.9, 1.0])
    layer = ec.cloud_layers(tcc)
    np.testing.assert_allclose(1.0 - (1.0 - layer) ** 3, tcc, atol=1e-12)


def test_ocean_field_regridded_to_sea_points_only():
    j, i = np.meshgrid(np.arange(4), np.arange(8), indexing="ij")
    lat2d, lon2d = -60.0 + 40.0 * j, 45.0 * i - 180.0            # curvilinear-style, lon in -180..180
    tos = xr.DataArray(np.full((2, 4, 8), 290.0), dims=("time", "j", "i"),
                       coords={"time": TIMES, "latitude": (("j", "i"), lat2d),
                               "longitude": (("j", "i"), lon2d)})
    sea = xr.DataArray(np.array([[True, False], [True, True]]), dims=("latitude", "longitude"),
                       coords={"latitude": [30.0, -30.0], "longitude": [0.0, 180.0]})
    out = ec.ocean_to_target(tos, sea)
    assert np.isnan(out.values[:, 0, 1]).all()
    np.testing.assert_allclose(out.values[:, sea.values], 290.0)


# ---------------------------------------------------------------- end to end (synthetic)

NATIVE_LAT, NATIVE_LON = np.array([-75.0, -25.0, 25.0, 75.0]), np.arange(0.0, 360.0, 60.0)
NATIVE_LAND = NATIVE_LON < 180.0


def native(times, base, levels=None, land_only=False, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    shape = (len(times),) + ((len(levels),) if levels is not None else ()) + (NATIVE_LAT.size, NATIVE_LON.size)
    values = (base + noise * rng.random(shape)).astype("float32")
    if land_only:
        values[..., ~NATIVE_LAND] = np.nan
    dims = ("time",) + (("level",) if levels is not None else ()) + ("latitude", "longitude")
    coords = {"time": times, "latitude": NATIVE_LAT, "longitude": NATIVE_LON}
    if levels is not None:
        coords["level"] = np.asarray(levels, dtype=float)
    return xr.DataArray(values, dims=dims, coords=coords)


def synthetic_fields():
    point6 = TIMES
    point_day = DAY + np.array([0, 6, 12, 18]) * HOUR
    mean6 = DAY + np.array([-3, 3, 9]) * HOUR
    daily = DAY + np.array([-12, 12, 36]) * HOUR
    eday_levels = card()["pressure_levels"]["values"]
    pt_levels, wap_levels = [925, 850, 700, 600, 500, 250, 50], [925, 850, 500, 250]
    f = {}
    for var, base in {"ta": 250.0, "ua": 5.0, "va": 2.0, "zg": 5000.0, "hus": 1e-3}.items():
        f[("Eday", var)] = native(daily, base, eday_levels, noise=base * 0.01)
        f[("6hrPlevPt", var)] = native(point6, base, pt_levels, noise=base * 0.01, seed=1)
    f[("Eday", "wap")] = native(daily, 0.0, eday_levels, noise=0.1)
    f[("6hrPlev", "wap")] = native(mean6, 0.0, wap_levels, noise=0.1, seed=1)
    f[("6hrPlevPt", "psl")] = native(point6, 101000.0, noise=500)
    f[("6hrPlevPt", "tsl")] = native(point_day, 280.0, land_only=True)
    for var, base in {"tas": 280.0, "uas": 3.0, "vas": 1.0, "huss": 5e-3}.items():
        f[("3hr", var)] = native(point6, base, noise=base * 0.01)
    f[("6hrPlev", "ua100m")] = native(mean6, 5.0)
    f[("6hrPlev", "va100m")] = native(mean6, 2.0)
    f[("day", "clt")] = native(daily, 50.0, noise=10.0)
    f[("day", "snw")] = native(daily, 10.0, land_only=True)
    f[("day", "mrsos")] = native(daily, 25.0, land_only=True)
    f[("day", "tslsi")] = native(daily, 279.0, land_only=True)
    f[("day", "tas")] = native(daily, 279.5)
    f[("SIday", "siconca")] = native(daily, 0.0, noise=0.9)
    j, i = np.meshgrid(np.arange(6), np.arange(12), indexing="ij")
    tos = xr.DataArray(np.full((3, 6, 12), 15.0, dtype="float32"), dims=("time", "j", "i"),
                       coords={"time": daily, "latitude": (("j", "i"), -75.0 + 30.0 * j),
                               "longitude": (("j", "i"), 30.0 * i)})
    tos.values[:, :, :6] = np.nan                                 # land in the ocean model
    f[("Oday", "tos")] = tos
    return f


def support_files(tmp_path):
    lat, lon = np.arange(-90., 91., 30.), np.arange(0., 360., 10.)
    shape = (lat.size, lon.size)
    lsm = np.broadcast_to((lon < 180.0).astype(float), shape)
    static = xr.Dataset({"lsm": (("latitude", "longitude"), lsm)} |
                        {name: (("latitude", "longitude"), np.full(shape, value))
                         for name, value in {"sdor": 1., "slor": .1, "z": 500. * ec.G, "slt": 1.}.items()},
                        coords={"latitude": lat, "longitude": lon})
    levels = standard()["pressure_levels"]["values"]
    climate = xr.Dataset({name: (("hour", "level", "latitude", "longitude"),
                                 np.full((4, len(levels)) + shape, 1e-6, dtype="float32"))
                          for name in ("clwc", "ciwc")},
                         coords={"hour": [0, 6, 12, 18], "level": levels, "latitude": lat, "longitude": lon})
    static.to_netcdf(tmp_path / "static.nc")
    climate.to_netcdf(tmp_path / "climatology.nc")
    return {"HPCROOTDIR": str(ROOT), "STATIC_DATA": str(tmp_path / "static.nc"),
            "CLIMATOLOGY_DATA": str(tmp_path / "climatology.nc")}


@pytest.fixture
def coarse_target(monkeypatch):
    monkeypatch.setattr(ec, "TARGET_LATITUDES", np.array([60.0, 20.0, -20.0, -60.0]))
    monkeypatch.setattr(ec, "TARGET_LONGITUDES", np.arange(0.0, 360.0, 45.0))


def test_requests_cover_profiles_fields_and_inputs():
    requests = ec.collect_requests(card(), standard()["variables"]["data"])
    assert requests[("6hrPlevPt", "hus")] == "instant" and requests[("Eday", "hus")] == "daily"
    assert requests[("Oday", "tos")] == "field" and requests[("3hr", "tas")] == "field"
    assert len(requests) == len(synthetic_fields())


def test_missing_six_hourly_humidity_falls_back_to_daily_means():
    calls = []

    def reader(esgf, experiment, table, variable, window):
        calls.append((table, variable))
        if (table, variable) == ("6hrPlevPt", "hus"):
            raise ec.NotPublished("no hus")
        return synthetic_fields()[(table, variable)], [f"u/{table}/{variable}"]

    fields, urls = ec.read_sources(card(), standard()["variables"]["data"], card()["esgf"], "historical",
                                   TIMES, reader)
    assert ("6hrPlevPt", "hus") not in fields and ("Eday", "hus") in fields
    assert len(urls) == len(calls) - 1

    def failing(esgf, experiment, table, variable, window):
        raise ec.NotPublished(f"{table}/{variable}")

    with pytest.raises(ec.NotPublished):
        ec.read_sources(card(), standard()["variables"]["data"], card()["esgf"], "historical", TIMES, failing)


def test_prepared_ics_contain_the_whole_standard(tmp_path, coarse_target):
    std = standard()
    out = ec.prepare_initial_conditions(synthetic_fields(), support_files(tmp_path), card(), std, TIMES)
    required = {v["short_name"] for v in std["variables"]["data"].values() if not v.get("optional")}
    assert required <= set(out.data_vars)
    assert out.level.values.tolist() == std["pressure_levels"]["values"]
    assert out.latitude.values.tolist() == [60.0, 20.0, -20.0, -60.0]
    assert out.time.values.tolist() == TIMES.tolist()
    assert out.t.dims == ("time", "level", "latitude", "longitude")
    assert out.sot.dims == ("time", "SoilLevel", "latitude", "longitude")
    assert out.SoilLevel.values.tolist() == [1, 2]
    sea = out.longitude >= 180.0
    assert out.sst.where(~sea).isnull().all() and out.sst.where(sea).notnull().sum() > 0
    assert float(out.sst.where(sea).max()) == pytest.approx(288.15)
    assert (out.sd.where(sea, 0.0) == 0).all() and float(out.sd.where(~sea).max()) > 0
    assert float(out.q.min()) >= 0 and float(out.tcwv.min()) > 0
    assert float(out.sd.min()) >= 0 and float(out.tcc.min()) >= 0 and float(out.ci.min()) >= 0
    assert (out.tcw >= out.tcwv).all()
    assert (out.sp < out.msl).all()                               # 500 m static orography everywhere
    assert out.z.attrs["units"] == std["variables"]["data"][129]["units"]
    assert out.clwc.attrs["source"] == "ERA5 climatology"
    for name in out.data_vars:
        if name not in {"sst", "ci"}:
            assert np.isfinite(out[name]).all(), name
    assert all(out[n].dtype == np.float32 for n in ("t", "q", "sp", "skt", "sot", "tcw"))
