import os
import sys
import io
import re
import time as _time
import datetime as dt
import argparse
import numpy as np
import requests
import eccodes as ec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D

# ======================================================================
# Default Parameters (overridable via CLI or Environment Variables)
# ======================================================================
PLOT_DEFAULT_BUOYS = True
PLOT_POINT_FORECAST = True
SHOW_WIND_DIRECTION = True
LABEL_EVERY_H = 6
SHOW_GFS_WAVE = True
SHOW_ECMWF_WAVE = True

PRESET = "Custom (use LAT / LON)"
LAT = os.getenv("WAVE_LAT", "18.64")
LON = os.getenv("WAVE_LON", "-65.854")
NAME = os.getenv("WAVE_NAME", "Loíza")

DATE = ""
CYCLE = 12
NWPS_FILE = "Auto (CG1, then CG0)"
GFS_FALLBACK = True
REFRESH_DATA = True
SHOW_INVENTORY = True
OUT_DIR = "web_images"

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
M2FT = 3.28084
MS2KT = 1.94384
RHO, G = 1025.0, 9.81
TE_OVER_TP = 0.9
LOCAL_UTC_OFFSET = -4
LOCAL_TZ_NAME = "AST"

DEFAULT_BUOYS = [
    ("North of Puerto Rico", 18.75, -66.50),
    ("Mona Passage",         18.10, -67.50),
    ("South of Puerto Rico", 17.70, -66.50),
]

PRESETS = {
    "Rincón":                 (18.34, -67.28),
    "Aguadilla (Crash Boat)": (18.47, -67.18),
    "Arecibo":                (18.50, -66.70),
    "San Juan (Isla Verde)":  (18.47, -66.00),
    "Luquillo / Fajardo":     (18.40, -65.70),
    "Culebra":                (18.30, -65.30),
    "Vieques (south)":        (18.07, -65.45),
    "Ponce":                  (17.94, -66.62),
    "Cabo Rojo":              (17.93, -67.20),
    "St. Thomas":             (18.30, -64.93),
    "St. Croix":              (17.68, -64.80),
}

HS_CLASSES = [
    (5.0,    "Small (<5 ft)",       "#3bb143"),
    (7.0,    "Moderate (5–7 ft)",   "#f2c400"),
    (np.inf, "High (≥7 ft)",        "#e02424"),
]

# Beaufort Scale: (upper_bound_kt, force_num, label, kt_range, color)
BEAUFORT_CLASSES = [
    (1.0,    0,  "Calm",          "<1 kt",     "#94a3b8"),  # Cool Slate Gray
    (4.0,    1,  "Light air",     "1–3 kt",    "#7dd3fc"),  # Pale Sky Blue
    (7.0,    2,  "Light breeze",  "4–6 kt",    "#2563eb"),  # Royal Blue
    (11.0,   3,  "Gentle breeze", "7–10 kt",   "#0d9488"),  # Deep Teal
    (17.0,   4,  "Mod. breeze",   "11–16 kt",  "#0B4F32"),  # Vivid Emerald Green
    (22.0,   5,  "Fresh breeze",  "17–21 kt",  "#eab308"),  # Bright Gold / Yellow
    (28.0,   6,  "Strong breeze", "22–27 kt",  "#f97316"),  # Vivid Orange
    (34.0,   7,  "Near gale",     "28–33 kt",  "#dc2626"),  # Crimson Red
    (41.0,   8,  "Gale",          "34–40 kt",  "#991b1b"),  # Dark Maroon
    (48.0,   9,  "Strong gale",   "41–47 kt",  "#db2777"),  # Hot Magenta / Pink
    (56.0,   10, "Storm",         "48–55 kt",  "#7e22ce"),  # Deep Purple
    (64.0,   11, "Violent storm", "56–63 kt",  "#4338ca"),  # Dark Indigo
    (np.inf, 12, "Hurricane",     "≥64 kt",    "#111827"),  # Near-Black Charcoal
]

PARAMS = {
    (10, 0, 3):  "HTSGW",
    (10, 0, 10): "DIRPW",
    (10, 0, 11): "PERPW",
    (0, 2, 0):   "WDIR",
    (0, 2, 1):   "WIND",
    (0, 2, 2):   "UGRD",
    (0, 2, 3):   "VGRD",
}

PART_PARAMS = {
    (10, 0, 8): ("HTSGW", "swell"),
    (10, 0, 9): ("PERPW", "swell"),
    (10, 0, 7): ("DIRPW", "swell"),
    (10, 0, 5): ("HTSGW", "windsea"),
    (10, 0, 6): ("PERPW", "windsea"),
    (10, 0, 4): ("DIRPW", "windsea"),
}
SURFACE_LEVEL_TYPES = {1, 101}

COMPASS16 = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
             "S","SSW","SW","WSW","W","WNW","NW","NNW"]

def _utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

def compass(deg):
    if deg is None or not np.isfinite(deg):
        return ""
    return COMPASS16[int((deg % 360) / 22.5 + 0.5) % 16]

def _beaufort_index(kt):
    if not np.isfinite(kt):
        return -1
    for idx, (ub, _, _, _, _) in enumerate(BEAUFORT_CLASSES):
        if kt < ub:
            return idx
    return len(BEAUFORT_CLASSES) - 1

# ----------------------------------------------------------------------
# 1. Download (NWPS from NOMADS, GFS-Wave fallback)
# ----------------------------------------------------------------------
FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_srnwps.pl"
PUB = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod"
NWPS_VARS = ("DIRPW", "HTSGW", "PERPW", "WDIR", "WIND", "UGRD", "VGRD")
BOX = (19.5, -68.0, -64.0, 17.0)

NWPS_PRODUCTS = {
    "CG1": dict(grid="CG1", file="nwps.t{c}z.cg1.{wfo}.grib2"),
    "CG0": dict(grid="CG0", file="{wfo}_nwps_CG0_Trkng_{d}_{c}00.grib2"),
}

def _product_order(product="auto"):
    p = str(product or "auto").strip().upper()
    if p.startswith("AUTO"):
        return ["CG1", "CG0"]
    if p.startswith("CG1"):
        return ["CG1"]
    if p.startswith("CG0"):
        return ["CG0"]
    raise ValueError(f"Unknown NWPS file choice '{product}' (use Auto, CG1 or CG0).")

def nwps_file(date, cycle, product="CG0", wfo="sju"):
    return NWPS_PRODUCTS[product]["file"].format(c=f"{int(cycle):02d}", d=date, wfo=wfo)

def nwps_url(date, cycle, product="CG0", wfo="sju", box=BOX, variables=NWPS_VARS):
    top, left, right, bottom = box
    c = f"{int(cycle):02d}"
    q = {"dir": f"/sr.{date}/{wfo}/{c}/{NWPS_PRODUCTS[product]['grid']}",
         "file": nwps_file(date, cycle, product, wfo),
         "all_lev": "on", "subregion": "",
         "toplat": top, "leftlon": left, "rightlon": right, "bottomlat": bottom}
    for v in variables:
        q[f"var_{v}"] = "on"
    return requests.Request("GET", FILTER, params=q).prepare().url

def pub_url(date, cycle, product="CG0", wfo="sju"):
    c = f"{int(cycle):02d}"
    return f"{PUB}/sr.{date}/{wfo}/{c}/{NWPS_PRODUCTS[product]['grid']}/{nwps_file(date, cycle, product, wfo)}"

def _listing(url):
    try:
        r = requests.get(url, timeout=(10, 60))
        if not r.ok:
            return []
        return re.findall(r'href="([^"?/][^"]*)"', r.text)
    except requests.RequestException:
        return []

def find_nwps_candidates(products, wfo="sju", max_days=3, limit=4):
    days = sorted({m for h in _listing(PUB + "/")
                   for m in re.findall(r"^sr\.(\d{8})/$", h)}, reverse=True)
    if not days:
        return None
    out = []
    for d in days[:max_days]:
        cycles = sorted({m for h in _listing(f"{PUB}/sr.{d}/{wfo}/")
                         for m in re.findall(r"^(\d{2})/$", h)}, reverse=True)
        for c in cycles:
            for p in products:
                folder = f"{PUB}/sr.{d}/{wfo}/{c}/{NWPS_PRODUCTS[p]['grid']}/"
                if nwps_file(d, c, p, wfo) in _listing(folder):
                    out.append((d, int(c), p))
            if len(out) >= limit:
                return out
    return out

def _cached(path):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if fh.read(4) == b"GRIB":
                return True
        os.remove(path)
    return False

def _fetch_grib(url, out_path, timeout=(10, 120)):
    tmp = out_path + ".part"
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if not r.ok:
                return "missing"
            head = b""
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        if not head:
                            head = chunk[:4]
                        f.write(chunk)
        if head != b"GRIB":
            os.remove(tmp)
            return "bad"
        os.replace(tmp, out_path)
        return "ok"
    except requests.RequestException:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "neterr"

def download_nwps(cache_dir=".", date=None, cycle=None, product="auto", force=False,
                  wfo="sju", guess_cycles=4, cycles=(18, 12, 6, 0)):
    os.makedirs(cache_dir, exist_ok=True)
    order = _product_order(product)
    if date and cycle is not None:
        cands = [(str(date), int(cycle), p) for p in order]
    else:
        cands = find_nwps_candidates(order, wfo)
        if cands is None:
            now = _utcnow()
            cands, k = [], 0
            while len(cands) < guess_cycles * len(order) and k < 3:
                day = now - dt.timedelta(days=k)
                for c in cycles:
                    if (day.replace(hour=c, minute=0, second=0, microsecond=0) <= now
                            and len(cands) < guess_cycles * len(order)):
                        cands += [(day.strftime("%Y%m%d"), c, p) for p in order]
                k += 1
        elif cands:
            d, c, p = cands[0]
            print(f"✓ Latest NWPS run posted: {d} {c:02d}Z ({p})")
    for d, c, p in cands:
        out = os.path.join(cache_dir, f"nwps_{p.lower()}_{d}_{c:02d}z.grib2")
        if not force and _cached(out):
            print(f"✓ Using cached NWPS {p} {d} {c:02d}Z -> {out}")
            yield out, d, c, p
            continue
        label = f"NWPS {p} {d} {c:02d}Z"
        r1 = _fetch_grib(nwps_url(d, c, p, wfo), out)
        r2 = r1 if r1 in ("ok", "neterr") else _fetch_grib(pub_url(d, c, p, wfo), out)
        if "ok" in (r1, r2):
            print(f"✓ Downloaded {label} ({os.path.getsize(out) / 1e6:.1f} MB) -> {out}")
            yield out, d, c, p
            continue
        if "neterr" in (r1, r2):
            print(f"  {label}: cannot reach NOMADS — skipping other NWPS options.")
            return
        why = {"missing": "not posted", "bad": "reply is not GRIB2"}
        print(f"  {label}: filter {why.get(r1, r1)}, /pub/ file {why.get(r2, r2)}")

# --- GFS-Wave from NOMADS (last-resort main source) -------------------------
GFS_PUB = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl"
GFS_GRIDS = ("atlocn.0p16", "global.0p16", "global.0p25")
GFS_MAX_HOUR = 168

def _gfs_run_files(d, c):
    names = _listing(f"{GFS_PUB}/gfs.{d}/{c:02d}/wave/gridded/")
    for g in GFS_GRIDS:
        pat = re.compile(rf"^gfswave\.t{c:02d}z\.{re.escape(g)}\.f(\d{{3}})\.grib2$")
        hrs = {}
        for n in names:
            m = pat.match(n)
            if m:
                hrs[int(m.group(1))] = n
        if 0 in hrs and max(hrs) >= 72:
            return g, hrs
    return None

def download_gfswave_nomads(cache_dir=".", force=False, max_days=2, every_h=3,
                            polite_s=0.4, box=BOX):
    os.makedirs(cache_dir, exist_ok=True)
    days = sorted({m for h in _listing(GFS_PUB + "/")
                   for m in re.findall(r"^gfs\.(\d{8})/$", h)}, reverse=True)
    if not days:
        raise RuntimeError("cannot read the NOMADS GFS folder listing")
    top, left, right, bottom = box
    for d in days[:max_days]:
        cyc = sorted({int(m) for h in _listing(f"{GFS_PUB}/gfs.{d}/")
                      for m in re.findall(r"^(\d{2})/$", h)}, reverse=True)
        for c in cyc:
            out = os.path.join(cache_dir, f"gfswave_{d}_{c:02d}z.grib2")
            if not force and _cached(out):
                return out, d, c, "cached"
            found = _gfs_run_files(d, c)
            if not found:
                continue
            grid, hrs = found
            want = [h for h in sorted(hrs) if h <= GFS_MAX_HOUR and h % every_h == 0]
            tmp, n_ok = out + ".part", 0
            with requests.Session() as sess, open(tmp, "wb") as fo:
                for h in want:
                    q = {"dir": f"/gfs.{d}/{c:02d}/wave/gridded", "file": hrs[h],
                         "all_lev": "on", "subregion": "", "toplat": top,
                         "leftlon": left, "rightlon": right, "bottomlat": bottom,
                         "var_HTSGW": "on", "var_PERPW": "on", "var_DIRPW": "on",
                         "var_WDIR": "on", "var_WIND": "on"}
                    try:
                        r = sess.get(GFS_FILTER, params=q, timeout=(10, 60))
                        if r.ok and r.content[:4] == b"GRIB":
                            fo.write(r.content)
                            n_ok += 1
                    except requests.RequestException:
                        pass
                    _time.sleep(polite_s)
            if n_ok >= max(3, len(want) // 2):
                os.replace(tmp, out)
                return out, d, c, grid
            os.remove(tmp)
    raise RuntimeError("no usable GFS-Wave run found on NOMADS")

# ----------------------------------------------------------------------
# 2. Read GRIB2 and extract point time series
# ----------------------------------------------------------------------
def _messages(path):
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                yield gid
            finally:
                ec.codes_release(gid)

def _var_name(gid):
    try:
        key = (ec.codes_get(gid, "discipline"),
               ec.codes_get(gid, "parameterCategory"),
               ec.codes_get(gid, "parameterNumber"))
        return PARAMS.get(key, ec.codes_get(gid, "shortName"))
    except ec.CodesInternalError:
        return ec.codes_get(gid, "shortName")

def _classify(gid):
    try:
        pk = (ec.codes_get(gid, "discipline"), ec.codes_get(gid, "parameterCategory"),
              ec.codes_get(gid, "parameterNumber"))
    except ec.CodesInternalError:
        return None
    lev = ec.codes_get(gid, "level")
    if pk in PART_PARAMS:
        var, tag = PART_PARAMS[pk]
        return var, tag, lev
    var = PARAMS.get(pk)
    if var is None:
        return None
    return var, ec.codes_get(gid, "typeOfFirstFixedSurface", ktype=int), lev

def _values(gid):
    ec.codes_set(gid, "missingValue", 9.999e20)
    v = ec.codes_get_values(gid).astype("float64")
    v[v > 9e20] = np.nan
    return v

def _valid_time(gid):
    d = ec.codes_get(gid, "validityDate")
    t = ec.codes_get(gid, "validityTime")
    return dt.datetime.strptime(f"{d}{t:04d}", "%Y%m%d%H%M")

def inventory(path):
    seen, times = {}, set()
    for gid in _messages(path):
        c = _classify(gid)
        k = (f"{_var_name(gid)}" + (f"[{c[1]}]" if c and isinstance(c[1], str) else ""),
             ec.codes_get(gid, "typeOfFirstFixedSurface", ktype=int), ec.codes_get(gid, "level"))
        seen[k] = seen.get(k, 0) + 1
        times.add(_valid_time(gid))
    print(f"{'var':16s}{'levType':>8s}{'level':>7s}{'n_times':>9s}")
    for (v, lt, lev), n in sorted(seen.items(), key=lambda x: str(x[0])):
        print(f"{v:16s}{lt:>8}{lev:>7}{n:>9}")
    times = sorted(times)
    print(f"Valid times: {times[0]:%Y-%m-%d %HZ} -> {times[-1]:%Y-%m-%d %HZ} ({len(times)})")

def extract_points(path, points, max_dist_km=25.0):
    lats = lons = wet = None
    t0 = None
    for gid in _messages(path):
        c = _classify(gid)
        if not c or c[0] != "HTSGW":
            continue
        t = _valid_time(gid)
        if t0 is None:
            t0 = t
            lats = ec.codes_get_array(gid, "latitudes")
            lons = ec.codes_get_array(gid, "longitudes")
            lons = np.where(lons > 180, lons - 360, lons)
            wet = np.zeros(lats.size, bool)
        if t == t0:
            v = _values(gid)
            wet |= np.isfinite(v) & (v > 0)
    if lats is None:
        raise ValueError("No wave-height field (HTSGW / SWELL / WVHGT) found in file.")

    idx, meta = {}, {}
    coslat = np.cos(np.radians(lats))
    for name, la, lo in points:
        lo = lo - 360 if lo > 180 else lo
        d = 111.2 * np.hypot(lats - la, (lons - lo) * coslat)
        d_w = np.where(wet, d, np.inf)
        i = int(np.argmin(d_w))
        if not np.isfinite(d_w[i]) or d_w[i] > max_dist_km:
            raise ValueError(f"'{name}' ({la}, {lo}) has no wet grid point within {max_dist_km} km.")
        idx[name] = i
        meta[name] = dict(req_lat=la, req_lon=lo, grid_lat=float(lats[i]),
                          grid_lon=float(lons[i]), dist_km=float(d[i]))

    raw = {n: {} for n in idx}
    sel = np.array(list(idx.values()))
    names = list(idx.keys())
    for gid in _messages(path):
        key = _classify(gid)
        if key is None:
            continue
        t = _valid_time(gid)
        vals = _values(gid)[sel]
        for n, v in zip(names, vals):
            raw[n].setdefault(key, {})[t] = v

    out = {}
    for n in names:
        out[n] = _combine(raw[n])
        out[n].update(meta[n])
    return out

def _combine(raw):
    times = sorted({t for s in raw.values() for t in s})
    T = len(times)
    def series(key):
        s = raw.get(key, {})
        return np.array([s.get(t, np.nan) for t in times], dtype=float)

    hs_keys = sorted((k for k in raw if k[0] == "HTSGW"), key=str)
    surf = [k for k in hs_keys if k[1] in SURFACE_LEVEL_TYPES]
    parts = [k for k in hs_keys if k[1] not in SURFACE_LEVEL_TYPES]
    native = [k for k in parts if k[1] not in ("swell", "windsea")]
    if native:
        parts = native

    partitions = []
    for k in parts:
        _, lt, lev = k
        hs = series(k)
        tp = series(("PERPW", lt, lev))
        dr = series(("DIRPW", lt, lev))
        partitions.append(dict(level=lev, hs_m=hs, tp_s=tp, dir_deg=dr,
                               power_kw=_power(hs, tp)))

    if surf:
        _, lt, lev = surf[0]
        hs = series(surf[0])
        tp = series(("PERPW", lt, lev))
        dr = series(("DIRPW", lt, lev))
        if partitions:
            power = np.nansum([p["power_kw"] for p in partitions], axis=0)
        else:
            power = _power(hs, tp)
    elif partitions:
        H = np.array([p["hs_m"] for p in partitions])
        hs = np.sqrt(np.nansum(H**2, axis=0))
        hs[np.all(~np.isfinite(H), axis=0)] = np.nan
        Hf = np.where(np.isfinite(H), H, -1)
        dom = np.argmax(Hf, axis=0)
        tp = np.array([partitions[d]["tp_s"][i] for i, d in enumerate(dom)])
        dr = np.array([partitions[d]["dir_deg"][i] for i, d in enumerate(dom)])
        power = np.nansum([p["power_kw"] for p in partitions], axis=0)
    else:
        raise ValueError("No HTSGW series found")

    wd_keys = [k for k in raw if k[0] == "WDIR"]
    wdir = series(wd_keys[0]) if wd_keys else np.full(T, np.nan)

    ws_keys = [k for k in raw if k[0] == "WIND"]
    u_keys = [k for k in raw if k[0] == "UGRD"]
    v_keys = [k for k in raw if k[0] == "VGRD"]
    if ws_keys:
        wspd_kt = series(ws_keys[0]) * MS2KT
    elif u_keys and v_keys:
        u = series(u_keys[0])
        v = series(v_keys[0])
        wspd_kt = np.hypot(u, v) * MS2KT
        if not np.isfinite(wdir).any():
            wdir = (np.degrees(np.arctan2(-u, -v)) % 360.0)
    else:
        wspd_kt = np.full(T, np.nan)

    return dict(time=np.array(times), hs_m=hs, tp_s=tp, dir_deg=dr,
                wdir_deg=wdir, wspd_kt=wspd_kt, power_kw=power,
                energy_kj=_energy(hs, tp),
                partitions=partitions if len(partitions) > 1 else [])

def _energy(hs_m, tp_s):
    hs = np.asarray(hs_m, float)
    tp = np.asarray(tp_s, float)
    return RHO * (G ** 2) * (hs ** 2) * (tp ** 2) / (32.0 * np.pi) / 1000.0

def _power(hs_m, tp_s):
    te = TE_OVER_TP * tp_s
    return RHO * (G ** 2) * (hs_m ** 2) * te / (64.0 * np.pi) / 1000.0

# ----------------------------------------------------------------------
# 2b. Other wave models & winds for comparison (Open-Meteo API)
# ----------------------------------------------------------------------
OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"
OPEN_METEO_WX_URL = "https://api.open-meteo.com/v1/forecast"
COMPARE_MODELS = {
    "GFS-Wave":  (["ncep_gfswave025", "ncep_gfswave016", "gfs_wave025"], "o", "#2563eb"),
    "ECMWF WAM": (["ecmwf_wam025", "ecmwf_wam"],                       "D", "#7c3aed"),
}

class _ServiceDown(Exception):
    pass

def _om_fetch(lats, lons, model_id, variables=("wave_height",), retries=1, base_url=OPEN_METEO_URL):
    base = dict(latitude=",".join(f"{v:.4f}" for v in lats),
                longitude=",".join(f"{v:.4f}" for v in lons),
                hourly=",".join(variables), timezone="GMT",
                timeformat="iso8601")
    if model_id:
        base["models"] = model_id
    variants = [dict(base, past_days=2, forecast_days=8, cell_selection="sea"),
                dict(base, past_days=2, forecast_days=7),
                dict(base, forecast_days=7)]
    last = "no response"
    for params in variants:
        for attempt in range(retries + 1):
            try:
                r = requests.get(base_url, params=params, timeout=(6, 30))
            except (requests.ConnectionError, requests.Timeout) as e:
                raise _ServiceDown(f"cannot reach Open-Meteo ({e.__class__.__name__})")
            except requests.RequestException as e:
                last = f"request error: {e.__class__.__name__}"
                break
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code} (server busy)"
                if attempt < retries:
                    _time.sleep(2)
                    continue
                raise RuntimeError(last)
            try:
                js = r.json()
            except ValueError:
                last = f"HTTP {r.status_code}, not JSON"
                break
            if r.status_code != 200 or (isinstance(js, dict) and js.get("error")):
                last = (js.get("reason") if isinstance(js, dict) else None) or f"HTTP {r.status_code}"
                low = last.lower()
                if "model" in low or "variable" in low or "hourly" in low:
                    raise RuntimeError(last)
                break
            items = js if isinstance(js, list) else [js]
            if len(items) != len(lats):
                last = f"expected {len(lats)} locations, got {len(items)}"
                break
            out = []
            for it in items:
                it = it or {}
                hr = it.get("hourly") or {}
                times = []
                for t in hr.get("time") or []:
                    try:
                        times.append(dt.datetime.strptime(str(t)[:16], "%Y-%m-%dT%H:%M"))
                    except ValueError:
                        times.append(None)
                ok = np.array([t is not None for t in times], bool)
                rec = {"lat": it.get("latitude"), "lon": it.get("longitude"),
                       "time": np.array([t for t in times if t is not None], dtype=object)}
                for v in variables:
                    key = next((k for k in hr if k == v or k.startswith(v + "_")), None)
                    vals = (hr.get(key) or []) if key else []
                    arr = np.array([np.nan if x is None else float(x) for x in vals], float)
                    if len(arr) != len(times):
                        arr = np.full(len(times), np.nan)
                    rec[v] = arr[ok]
                out.append(rec)
            return out
    raise RuntimeError(last)

def _om_request(lats, lons, model_id, retries=1):
    res = _om_fetch(lats, lons, model_id, ("wave_height",), retries)
    return [{t: v for t, v in zip(r["time"], r["wave_height"]) if np.isfinite(v)} for r in res]

def enrich_winds_openmeteo(data, points):
    """Fill missing wind speed or direction from Open-Meteo 10m wind (knots)."""
    need = [p for p in points if not (np.isfinite(data[p[0]]["wspd_kt"]).any()
                                      and np.isfinite(data[p[0]]["wdir_deg"]).any())]
    if not need:
        return
    lats = [p[1] for p in need]
    lons = [p[2] - 360 if p[2] > 180 else p[2] for p in need]
    try:
        res = _om_fetch(lats, lons, None, ("wind_speed_10m", "wind_direction_10m"),
                        base_url=OPEN_METEO_WX_URL)
        for (nm, _, _), r in zip(need, res):
            ws_map = {t: v * MS2KT for t, v in zip(r["time"], r["wind_speed_10m"]) if np.isfinite(v)}
            wd_map = {t: v for t, v in zip(r["time"], r["wind_direction_10m"]) if np.isfinite(v)}
            times = data[nm]["time"]
            if not np.isfinite(data[nm]["wspd_kt"]).any():
                data[nm]["wspd_kt"] = np.array([ws_map.get(t, np.nan) for t in times], float)
            if not np.isfinite(data[nm]["wdir_deg"]).any():
                data[nm]["wdir_deg"] = np.array([wd_map.get(t, np.nan) for t in times], float)
    except Exception as e:
        print(f"⚠ Could not fetch fallback wind speed/direction ({str(e)[:120]})")

def gfs_points_openmeteo(points):
    lats = [p[1] for p in points]
    lons = [p[2] - 360 if p[2] > 180 else p[2] for p in points]
    full = ("wave_height", "wave_direction", "wave_period", "wave_peak_period")
    res, used, last = None, None, "no response"
    for mid in COMPARE_MODELS["GFS-Wave"][0]:
        for variables in (full, full[:3]):
            try:
                res, used = _om_fetch(lats, lons, mid, variables), mid
                break
            except _ServiceDown:
                raise
            except Exception as e:
                last = f"{mid}: {str(e)[:120]}"
                if "model" in str(e).lower():
                    break
        if res is not None:
            break
    if res is None:
        raise RuntimeError(last)
    start = _utcnow().replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=3)
    data = {}
    for (nm, la, lo), r in zip(points, res):
        keep = np.array([t >= start for t in r["time"]], bool)
        hs = r["wave_height"][keep]
        if not np.isfinite(hs).any():
            raise RuntimeError(f"no GFS-Wave values at '{nm}'")
        tp = r["wave_period"][keep]
        if "wave_peak_period" in r:
            pk = r["wave_peak_period"][keep]
            tp = np.where(np.isfinite(pk), pk, tp)
        glat = float(r["lat"]) if r["lat"] is not None else la
        glon = float(r["lon"]) if r["lon"] is not None else lo
        dist = 111.2 * np.hypot(glat - la, (glon - lo) * np.cos(np.radians(la)))
        n_keep = int(keep.sum())
        data[nm] = dict(time=r["time"][keep], hs_m=hs, tp_s=tp,
                        dir_deg=r["wave_direction"][keep],
                        wdir_deg=np.full(n_keep, np.nan),
                        wspd_kt=np.full(n_keep, np.nan),
                        power_kw=_power(hs, tp), energy_kj=_energy(hs, tp), partitions=[],
                        req_lat=la, req_lon=lo, grid_lat=glat, grid_lon=glon,
                        dist_km=float(dist))
    print(f"✓ GFS-Wave from Open-Meteo ({used}) for {len(data)} point(s)")
    return data

def load_gfs_main(points, cache_dir="nwps_cache", force=False, show_inventory=False):
    try:
        data = gfs_points_openmeteo(points)
        return data, dict(main="GFS-Wave", label="GFS-Wave (via Open-Meteo)",
                          run="latest available", fallback=True, open_meteo=True)
    except Exception as e:
        print(f"⚠ GFS-Wave from Open-Meteo failed ({str(e)[:150]}) — trying NOMADS.")
    try:
        path, d, c, grid = download_gfswave_nomads(cache_dir, force=force)
        if show_inventory:
            inventory(path)
        data = extract_points(path, points, max_dist_km=45.0)
        return data, dict(main="GFS-Wave", label=f"GFS-Wave {grid} (NOMADS)",
                          run=_run_label(d, c), fallback=True, open_meteo=False)
    except Exception as e:
        raise RuntimeError("No wave data available: NWPS and GFS-Wave all failed. "
                           f"Last error: {str(e)[:200]}") from None

def _run_label(d, c):
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{d[6:]} {months[int(d[4:6]) - 1]} {d[:4]} {int(c):02d}Z"

def add_model_comparison(data, points, use_gfs=True, use_ecmwf=True):
    wanted = [m for m, on in (("GFS-Wave", use_gfs), ("ECMWF WAM", use_ecmwf)) if on]
    loaded = []
    if not wanted or not points:
        return loaded
    names = [p[0] for p in points]
    lats = [p[1] for p in points]
    lons = [p[2] - 360 if p[2] > 180 else p[2] for p in points]
    down = None
    for model in wanted:
        ids = COMPARE_MODELS[model][0]
        result, reason = None, down or ""
        for mid in ([] if down else ids):
            try:
                result = _om_request(lats, lons, mid)
                if any(result):
                    break
                reason, result = f"{mid}: no wave data at these points", None
            except _ServiceDown as e:
                down = reason = str(e)
                result = None
                break
            except Exception as e:
                reason, result = f"{mid}: {str(e)[:120]}", None
        if result is None:
            print(f"⚠ {model} not available ({reason}) — plotting without it.")
            continue
        n_ok = 0
        for nm, series in zip(names, result):
            times = data[nm]["time"]
            arr = np.array([series.get(t, np.nan) for t in times], float)
            data[nm].setdefault("models", {})[model] = arr
            n_ok += int(np.isfinite(arr).any())
        if n_ok:
            loaded.append(model)
            print(f"✓ {model}: data for {n_ok}/{len(names)} point(s) (Open-Meteo, {mid})")
        else:
            for nm in names:
                data[nm].get("models", {}).pop(model, None)
    return loaded

# ----------------------------------------------------------------------
# 3. Plotting — "forecast strip" style + Beaufort wind arrows & side legend
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
INK, MUTED, FAINT = "#1f2937", "#6b7280", "#9ca3af"
NIGHT_C = "#eef1f5"

E_CLASSES = [
    (100.0,  "Tiny <100 kJ",       "#22c55e"),
    (300.0,  "Small 100–300",      "#facc15"),
    (600.0,  "Good 300–600",       "#f59e0b"),
    (1000.0, "Big 600–1000",       "#ef4444"),
    (np.inf, "Huge ≥1000",         "#db2777"),
]
INFO_STEP_H = 6

def _class_index(v, classes):
    if not np.isfinite(v):
        return -1
    for k, (ub, _, _) in enumerate(classes):
        if v < ub:
            return k
    return len(classes) - 1

def _local(times):
    return np.array([t + dt.timedelta(hours=LOCAL_UTC_OFFSET) for t in times])

def _step_days(tl):
    return np.median(np.diff(mdates.date2num(tl))) if len(tl) > 1 else 1 / 24

def _shade_nights(ax, tl, sunset=18.2, sunrise=6.6):
    d = tl[0].replace(hour=0, minute=0) - dt.timedelta(days=1)
    while d < tl[-1]:
        a = d + dt.timedelta(hours=sunset)
        b = d + dt.timedelta(days=1, hours=sunrise)
        if b > tl[0] and a < tl[-1]:
            ax.axvspan(max(a, tl[0]), min(b, tl[-1]), color=NIGHT_C, zorder=0, lw=0)
        d += dt.timedelta(days=1)

def _midnights(tl):
    d = tl[0].replace(hour=0, minute=0) + dt.timedelta(days=1)
    out = []
    while d <= tl[-1]:
        out.append(d)
        d += dt.timedelta(days=1)
    return out

def _clean(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def _hour_label(t):
    h = t.hour % 12
    return "12" if h == 0 else str(h)

def _info_indices(tl):
    if len(tl) == 0:
        return np.array([], int)
    x = mdates.date2num(tl)
    tol = _step_days(tl) * 0.51
    t = tl[0].replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % INFO_STEP_H)
    out = []
    while t <= tl[-1]:
        xt = mdates.date2num(t)
        k = int(np.argmin(np.abs(x - xt)))
        if abs(x[k] - xt) <= tol and (not out or out[-1] != k):
            out.append(k)
        t += dt.timedelta(hours=INFO_STEP_H)
    return np.array(out, int)

def _tick_times(tl, every_h=3):
    t = tl[0].replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % every_h)
    out = []
    while t <= tl[-1]:
        if t >= tl[0] - dt.timedelta(minutes=30):
            out.append(t)
        t += dt.timedelta(hours=every_h)
    return out

def _strip(fig, top_in, W, s, title, ymax_ft, bars_h, show_wind, show_axis=True,
           L_in=0.75, R_in=1.85):
    H = fig.get_figheight()
    L, R = L_in / W, 1.0 - (R_in / W)
    tl = _local(s["time"])
    xs = mdates.date2num(tl)
    hs = s["hs_m"] * M2FT
    tp = np.asarray(s["tp_s"], float)
    pw = np.asarray(s["energy_kj"], float)
    step = _step_days(tl)
    xlim = (xs[0] - step * 0.6, xs[-1] + step * 0.6)

    TITLE_H, INFO_H, WIND_H, E_H, AX_H = 0.42, 0.78, (0.36 if show_wind else 0), 0.42, (0.3 if show_axis else 0.05)
    y = top_in - TITLE_H

    axb = fig.add_axes([L, (y - bars_h) / H, R - L, bars_h / H])
    y -= bars_h
    _shade_nights(axb, tl)
    for m in _midnights(tl):
        axb.axvline(m, color=FAINT, lw=0.7, zorder=1)
    cols = [HS_CLASSES[_class_index(v, HS_CLASSES)][2] if np.isfinite(v) else "#e5e7eb" for v in hs]
    axb.bar(xs, hs, width=step * 0.72, color=cols, alpha=0.85, zorder=3, lw=0)
    for ub, _, c in HS_CLASSES[:-1]:
        if ub < ymax_ft:
            axb.axhline(ub, color=c, lw=0.8, ls=(0, (3, 3)), alpha=0.9, zorder=2)
    axb.set_ylim(0, ymax_ft)
    axb.set_xlim(*xlim)
    _clean(axb)
    axb.tick_params(left=True, labelleft=True, length=2, colors=FAINT, labelsize=7.5)
    tick_step = 3 if ymax_ft <= 15 else 5
    axb.set_yticks(np.arange(tick_step, ymax_ft + 0.01, tick_step))
    axb.set_ylabel("ft", color=FAINT, fontsize=8, rotation=0, labelpad=8, va="center")

    for m in [tl[0].replace(hour=0, minute=0)] + _midnights(tl):
        mid = m + dt.timedelta(hours=12)
        if tl[0] <= mid <= tl[-1]:
            axb.text(mid, ymax_ft * 0.99, f"{mid:%a %-d}", ha="center", va="top",
                     fontsize=8.5, color=MUTED, fontweight="bold", zorder=6)

    models = {m: np.asarray(v, float) * M2FT for m, v in (s.get("models") or {}).items()
              if np.isfinite(v).any()}
    if models:
        for k in _info_indices(tl):
            for m, v in models.items():
                if np.isfinite(v[k]):
                    _, mk, mc = COMPARE_MODELS[m]
                    axb.plot(xs[k], v[k], marker=mk, ms=5.5 if mk == "o" else 5,
                             mfc=mc, mec="white", mew=0.8, ls="none", zorder=8)

    grid_note = (f"grid pt {s['grid_lat']:.2f}N {abs(s['grid_lon']):.2f}W"
                 + (f" ({s['dist_km']:.1f} km)" if s["dist_km"] > 1.5 else ""))
    t_title = fig.text(L, (top_in - 0.12) / H, f"{title}", ha="left", va="top",
                       fontsize=12, fontweight="bold", color="#0b3d91")
    if np.isfinite(hs).any():
        i = int(np.nanargmax(hs))
        try:
            bb = t_title.get_window_extent(renderer=fig.canvas.get_renderer())
            x_after = (bb.x1 / fig.dpi + 0.15) / W
        except Exception:
            x_after = L + 0.18
        fig.text(x_after, (top_in - 0.15) / H,
                 f"Max {hs[i]:.1f} ft · {tl[i]:%a %-I%p}", ha="left", va="top",
                 fontsize=9, color=INK, fontweight="bold")
    fig.text(R, (top_in - 0.14) / H, f"{s['req_lat']:.2f}N {abs(s['req_lon']):.2f}W · {grid_note}",
             ha="right", va="top", fontsize=7.5, color=MUTED)

    axi = fig.add_axes([L, (y - INFO_H) / H, R - L, INFO_H / H], sharex=axb)
    y -= INFO_H
    _shade_nights(axi, tl)
    axi.set_ylim(0, 3)
    _clean(axi)
    idx = _info_indices(tl)
    dirs = np.asarray(s["dir_deg"], float)
    ok = idx[np.isfinite(dirs[idx])]
    th = np.radians(dirs[ok] + 180.0)
    if len(ok):
        axi.quiver(xs[ok], np.full(len(ok), 2.35), np.sin(th), np.cos(th), angles="uv",
                   pivot="middle", color=INK, scale=85, width=0.0024, headwidth=4.2,
                   headlength=4.2, headaxislength=3.8, zorder=5)
    for k in idx:
        if np.isfinite(hs[k]):
            axi.text(xs[k], 1.25, f"{hs[k]:.0f} ft", ha="center", va="center",
                     fontsize=9, fontweight="bold", color=INK, zorder=5)
        if np.isfinite(tp[k]):
            axi.text(xs[k], 0.5, f"{tp[k]:.0f} s", ha="center", va="center",
                     fontsize=8, color="black", zorder=5)

    if show_wind:
        axw = fig.add_axes([L, (y - WIND_H) / H, R - L, WIND_H / H], sharex=axb)
        y -= WIND_H
        _shade_nights(axw, tl)
        _clean(axw)
        axw.set_ylim(0, 1)
        wd = np.asarray(s["wdir_deg"], float)
        ws = np.asarray(s.get("wspd_kt", np.full(len(wd), np.nan)), float)
        okw = idx[np.isfinite(wd[idx])]
        thw = np.radians(wd[okw] + 180.0)
        if len(okw):
            arrow_colors = []
            for k in okw:
                b_idx = _beaufort_index(ws[k])
                arrow_colors.append(BEAUFORT_CLASSES[b_idx][4] if b_idx >= 0 else MUTED)
            axw.quiver(xs[okw], np.full(len(okw), 0.58), np.sin(thw), np.cos(thw),
                       angles="uv", pivot="middle", color=arrow_colors,
                       scale=72, width=0.0030, headwidth=4.2, headlength=4.2,
                       headaxislength=3.8, zorder=5)
            for k in okw:
                if np.isfinite(ws[k]):
                    axw.text(xs[k], 0.14, f"{ws[k]:.0f}kt", ha="center", va="center",
                             fontsize=6.5, color=MUTED, zorder=5)
        else:
            axw.text(xs[len(xs) // 2], 0.5, "wind direction not available from this source",
                     ha="center", va="center", fontsize=7, color=FAINT)
        axw.text(xs[0] - step * 0.6, 0.5, "Wind ", ha="right", va="center",
                 fontsize=7.5, color=MUTED, fontweight="bold")

    axe = fig.add_axes([L, (y - E_H) / H, R - L, E_H / H], sharex=axb)
    y -= E_H
    _clean(axe); axe.set_ylim(-1, 1.2)
    cls = np.array([_class_index(v, E_CLASSES) for v in pw])
    k0 = 0
    for k in range(1, len(cls) + 1):
        if k == len(cls) or cls[k] != cls[k0]:
            if cls[k0] >= 0:
                x0 = xs[k0] - step * 0.5 + step * 0.12
                x1 = (xs[k - 1] + step * 0.5) - step * 0.12
                axe.plot([x0, x1], [0, 0], color=E_CLASSES[cls[k0]][2], lw=5.5,
                         solid_capstyle="round", zorder=4)
            k0 = k
    axe.text(xs[0] - step * 0.6, 0, "Energy ", ha="right", va="center", fontsize=7.5,
             color=MUTED, fontweight="bold")
    if np.isfinite(pw).any():
        i = int(np.nanargmax(pw))
        axe.text(xs[i], 0.55, f"peak {pw[i]:,.0f} kJ", ha="center", va="bottom",
                 fontsize=7, color=INK)

    if show_axis:
        axe.tick_params(bottom=True, labelbottom=True, length=3, colors=FAINT, labelsize=7.5)
        ticks = _tick_times(tl, 3)
        axe.set_xticks(mdates.date2num(ticks))
        axe.set_xticklabels([_hour_label(t) for t in ticks])
        for lab in axe.get_xticklabels():
            lab.set_color(MUTED)
    y -= AX_H

    axb.set_xlim(*xlim)
    return y

def _beaufort_side_legend(fig, W, H, top_in, bottom_in, R_in=1.85):
    """Draw the Beaufort Wind Scale legend in the right margin."""
    x0 = 1.0 - (R_in - 0.18) / W
    w = (R_in - 0.30) / W
    y0 = bottom_in / H
    h = (top_in - bottom_in) / H
    if h <= 0.05:
        return

    ax = fig.add_axes([x0, y0, w, h])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(FancyBboxPatch(
        (0.01, 0.005), 0.98, 0.99,
        boxstyle="round,pad=0,rounding_size=0.03",
        fc="#f8fafc", ec="#e2e8f0", lw=1.0,
        mutation_aspect=w / h
    ))

    ax.text(0.5, 0.965, "BEAUFORT SCALE", ha="center", va="top",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.5, 0.935, "(Wind Arrows)", ha="center", va="top",
            fontsize=6.8, color=MUTED)

    n = len(BEAUFORT_CLASSES)
    y_top = 0.895
    y_bot = 0.025
    row_h = (y_top - y_bot) / n

    for i, (_, force, label, kt_rng, color) in enumerate(BEAUFORT_CLASSES):
        yc = y_top - (i + 0.5) * row_h
        # Color swatch
        ax.add_patch(FancyBboxPatch(
            (0.06, yc - row_h * 0.32), 0.12, row_h * 0.64,
            boxstyle="round,pad=0,rounding_size=0.015",
            fc=color, ec="none"
        ))
        # Force badge + description
        ax.text(0.22, yc + row_h * 0.12, f"F{force} · {label}",
                ha="left", va="center", fontsize=6.7, fontweight="bold", color=INK)
        ax.text(0.22, yc - row_h * 0.20, kt_rng,
                ha="left", va="center", fontsize=6.1, color=MUTED)

def _legend(fig, y_in, show_wind, models=(), R_in=1.85):
    H = fig.get_figheight()
    W = fig.get_figwidth()
    cx = (0.75 / W + (1.0 - R_in / W)) / 2.0
    if models:
        mh = [Line2D([], [], color="none", marker=COMPARE_MODELS[m][1],
                     mfc=COMPARE_MODELS[m][2], mec="white", ms=7, label=f"{m} wave height") for m in models]
        fig.legend(handles=mh, loc="lower center", bbox_to_anchor=(cx, (y_in + 0.66) / H),
                   ncol=len(mh), fontsize=8.5, frameon=False, handlelength=1.4,
                   columnspacing=1.6, labelcolor="black")
    h = [Patch(color=c, alpha=0.85, label=f"Height: {l}") for _, l, c in HS_CLASSES]
    h += [Line2D([], [], color=INK, marker=r"$\searrow$", ls="none", ms=11,
                 label="Wave direction · height · period")]
    if show_wind:
        h += [Line2D([], [], color="#22c55e", marker=r"$\searrow$", ls="none", ms=10,
                     label="Wind direction (Beaufort scale →)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(cx, (y_in + 0.33) / H),
               ncol=len(h), fontsize=8.2, frameon=False, handlelength=1.4, columnspacing=1.2,
               labelcolor="black")
    e = [Line2D([], [], color=c, lw=5, solid_capstyle="round", label=l)
         for _, l, c in E_CLASSES]
    fig.legend(handles=e, loc="lower center", bbox_to_anchor=(cx, y_in / H),
               ncol=len(e), fontsize=8.5, frameon=False, handlelength=1.8,
               labelcolor="black")

def _footer(fig, run_label, models=(), source=None):
    source = source or {}
    om = list(models) + (["GFS-Wave"] if source.get("open_meteo") else [])
    om = list(dict.fromkeys(om))
    extra = (" · " + " / ".join(om) + ": Open-Meteo.com (CC BY 4.0)") if om else ""
    fig.text(0.01, 0.006, f"{source.get('label', 'NWPS SJU')} · run {run_label} · times {LOCAL_TZ_NAME} "
             f"(UTC{LOCAL_UTC_OFFSET:+d}), shaded = night · wave energy E = ρg²·Hs²·Tp²/32π "
             f"(kJ per m of crest){extra}", fontsize=7.5, color=FAINT)
    fig.text(0.99, 0.006, f"Created {_utcnow():%Y-%m-%d %H:%MZ}", ha="right",
             fontsize=7.5, color=FAINT)

Y_FIXED_FT = 15.0

def _common_ymax(series_list):
    def smax(s):
        vals = [np.nanmax(s["hs_m"])] + [np.nanmax(v) for v in (s.get("models") or {}).values()
                                         if np.isfinite(v).any()]
        return max(vals) * M2FT
    m = max(smax(s) for s in series_list)
    if m <= Y_FIXED_FT:
        return Y_FIXED_FT
    return float(np.ceil((m + 1.5) / 5.0) * 5.0)

def _strip_height(bars_h, show_wind, show_axis=True):
    return 0.42 + bars_h + 0.78 + (0.36 if show_wind else 0) + 0.42 + (0.3 if show_axis else 0.05)

# ----------------------------------------------------------------------
# 4. The two figures
# ----------------------------------------------------------------------
def _models_in(series):
    found = []
    for s in series:
        for m, v in (s.get("models") or {}).items():
            if np.isfinite(v).any() and m not in found:
                found.append(m)
    return [m for m in COMPARE_MODELS if m in found]

def _titles(fig, H, what, source, R_in=1.85):
    source = source or {}
    W = fig.get_figwidth()
    cx = (0.75 / W + (1.0 - R_in / W)) / 2.0
    main = source.get("main", "NWPS")
    fig.text(cx, 1.0 - 0.18 / H, f"{main} {what}", ha="center", va="top",
             fontsize=16, fontweight="bold", color=INK)
    if source.get("fallback"):
        fig.text(cx, 1.0 - 0.5 / H, f"⚠ NWPS not available — showing {source.get('label', 'GFS-Wave')}"
                 " · wave height (bars) · direction · period · wave energy",
                 ha="center", va="top", fontsize=9.5, color="#b45309", fontweight="bold")
    else:
        fig.text(cx, 1.0 - 0.5 / H, "Wave height (bars) · direction · period · wave energy",
                 ha="center", va="top", fontsize=9.5, color=MUTED)

def plot_default_buoys(data, run_label, out="web_images/nwps_virtual_buoys.png",
                       show_wind=True, dpi=150, source=None):
    names = [b[0] for b in DEFAULT_BUOYS]
    series = [data[n] for n in names]
    ymax = _common_ymax(series)
    models = _models_in(series)
    R_in = 1.85 if show_wind else 0.35
    W, TOP, GAP, BOTTOM, BARS = 16.2, 0.75, 0.3, (1.33 if models else 1.0), 1.55
    sh = _strip_height(BARS, show_wind)
    H = TOP + len(names) * sh + (len(names) - 1) * GAP + BOTTOM
    fig = plt.figure(figsize=(W, H), facecolor="white")
    _titles(fig, H, "Virtual Wave Buoys — Puerto Rico & U.S. Virgin Islands", source, R_in=R_in)
    y = H - TOP
    for n in names:
        y = _strip(fig, y, W, data[n], n, ymax, BARS, show_wind, R_in=R_in) - GAP
    if show_wind:
        _beaufort_side_legend(fig, W, H, top_in=H - TOP - 0.15, bottom_in=BOTTOM, R_in=R_in)
    _legend(fig, 0.25, show_wind, models, R_in=R_in)
    _footer(fig, run_label, models, source)
    fig.savefig(out, dpi=dpi, facecolor="white")
    print(f"✓ saved {out}")
    return fig

def plot_custom_point(data, point, run_label, out="web_images/nwps_point_forecast.png",
                      show_wind=True, dpi=150, source=None):
    nm = point[0]
    sr = data[nm]
    ymax = _common_ymax([sr])
    models = _models_in([sr])
    R_in = 1.85 if show_wind else 0.35
    W, TOP, BOTTOM, BARS = 16.2, 0.75, (1.33 if models else 1.0), 2.8
    H = TOP + _strip_height(BARS, show_wind) + BOTTOM
    fig = plt.figure(figsize=(W, H), facecolor="white")
    _titles(fig, H, "Point Forecast", source, R_in=R_in)
    _strip(fig, H - TOP, W, sr, nm, ymax, BARS, show_wind, R_in=R_in)
    if show_wind:
        _beaufort_side_legend(fig, W, H, top_in=H - TOP - 0.15, bottom_in=BOTTOM, R_in=R_in)
    _legend(fig, 0.25, show_wind, models, R_in=R_in)
    _footer(fig, run_label, models, source)
    fig.savefig(out, dpi=dpi, facecolor="white")
    print(f"✓ saved {out}")
    return fig

def summary_table(data, names):
    models = _models_in([data[n] for n in names])
    extra = "".join(f"{(m.split()[0].split('-')[0] + ' ft'):>10s}" for m in models)
    print(f"{'Point':24s}{'Max Hs ft':>10s}{'when (AST)':>16s}{'Tp s':>6s}{'Dir':>5s}"
          f"{'Max E kJ':>10s}{extra}")
    for n in names:
        s = data[n]
        hs = s["hs_m"] * M2FT
        i = int(np.nanargmax(hs))
        tl = _local(s["time"])
        cols = ""
        for m in models:
            v = (s.get("models") or {}).get(m)
            cols += (f"{np.nanmax(v) * M2FT:10.1f}" if v is not None and np.isfinite(v).any()
                     else f"{'—':>10s}")
        print(f"{n:24s}{hs[i]:10.1f}{tl[i].strftime('%a %d %-I%p'):>16}{s['tp_s'][i]:6.0f}"
              f"{compass(s['dir_deg'][i]):>5s}{np.nanmax(s['energy_kj']):10,.0f}{cols}")

# ----------------------------------------------------------------------
# 5. One-call runner
# ----------------------------------------------------------------------
def run(lat=None, lon=None, name="", date=None, cycle=None, preset=None,
        refresh=False, cache_dir="nwps_cache",
        plot_default=True, plot_point=True, show_wind=True,
        out_default="web_images/nwps_virtual_buoys.png",
        out_point="web_images/nwps_point_forecast.png", local_grib=None,
        show_inventory=True, info_step_h=None, compare_gfs=True, compare_ecmwf=True,
        nwps_product="auto", gfs_fallback=True):
    global INFO_STEP_H
    if info_step_h:
        INFO_STEP_H = int(info_step_h)
    if not (plot_default or plot_point):
        raise ValueError("Nothing to do: turn on at least one plot.")
    os.makedirs(OUT_DIR, exist_ok=True)
    point = None
    if plot_point and preset and preset in PRESETS:
        lat, lon = PRESETS[preset]
        name = name or preset
    if plot_point:
        if lat in (None, "") or lon in (None, ""):
            raise ValueError("The point forecast needs LAT and LON.")
        lat, lon = float(lat), float(lon)
        if lon > 0:
            lon = -lon
        if not (17.0 <= lat <= 19.5 and -68.0 <= lon <= -64.0):
            raise ValueError(f"({lat}, {lon}) is outside the NWPS domain 17–19.5N, 64–68W.")
        point = ((name or "").strip() or f"Point {lat:.2f}N {abs(lon):.2f}W", lat, lon)

    points = (DEFAULT_BUOYS if plot_default else []) + ([point] if point else [])

    data, source = None, None
    if local_grib:
        if show_inventory:
            inventory(local_grib)
        data = extract_points(local_grib, points)
        source = dict(main="NWPS", label=f"NWPS ({os.path.basename(local_grib)})", run="local file")
    else:
        try:
            for path, d, c, p in download_nwps(cache_dir, date or None,
                                               cycle if date else None,
                                               product=nwps_product, force=refresh):
                try:
                    if show_inventory:
                        inventory(path)
                    data = extract_points(path, points)
                    source = dict(main="NWPS", label=f"NWPS SJU {p}", run=_run_label(d, c))
                    print(f"✓ Using NWPS {p} {d} {c:02d}Z")
                    break
                except Exception as e:
                    print(f"⚠ NWPS {p} {d} {c:02d}Z could not be used ({str(e)[:150]}) — trying next.")
        except Exception as e:
            print(f"⚠ NWPS download failed ({str(e)[:150]}).")

    if data is None:
        if not gfs_fallback:
            raise RuntimeError("NWPS data is not available and GFS-Wave fallback is off.")
        print("⚠ NWPS not available — using GFS-Wave as the main source.")
        data, source = load_gfs_main(points, cache_dir, refresh, show_inventory)
        compare_gfs = False
    run_label = source["run"]

    if show_wind:
        enrich_winds_openmeteo(data, points)

    if compare_gfs or compare_ecmwf:
        add_model_comparison(data, points, compare_gfs, compare_ecmwf)
    summary_table(data, [p[0] for p in points])

    outs = []
    if plot_default:
        plot_default_buoys(data, run_label, out=out_default, show_wind=show_wind, source=source)
        outs.append(out_default)
    if plot_point:
        plot_custom_point(data, point, run_label, out=out_point, show_wind=show_wind, source=source)
        outs.append(out_point)
    plt.close("all")
    return outs

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser(description="NWPS SJU virtual wave buoys / point forecast")
        ap.add_argument("--lat", type=float)
        ap.add_argument("--lon", type=float)
        ap.add_argument("--name", default="")
        ap.add_argument("--preset", choices=list(PRESETS))
        ap.add_argument("--date", help="YYYYMMDD (default: latest posted run)")
        ap.add_argument("--cycle", type=int, default=12, choices=[0, 6, 12, 18])
        ap.add_argument("--refresh", action="store_true")
        ap.add_argument("--no-default", action="store_true")
        ap.add_argument("--no-point", action="store_true")
        ap.add_argument("--no-wind", action="store_true")
        ap.add_argument("--grib")
        ap.add_argument("--every", type=int, default=6, choices=[3, 6, 12])
        ap.add_argument("--no-gfs", action="store_true")
        ap.add_argument("--no-ecmwf", action="store_true")
        ap.add_argument("--product", default="auto", choices=["auto", "cg1", "cg0"])
        ap.add_argument("--no-gfs-fallback", action="store_true")
        a = ap.parse_args()
        run(lat=a.lat, lon=a.lon, name=a.name, preset=a.preset, date=a.date,
            cycle=a.cycle, refresh=a.refresh, plot_default=not a.no_default,
            plot_point=not a.no_point, show_wind=not a.no_wind, local_grib=a.grib,
            info_step_h=a.every, compare_gfs=not a.no_gfs, compare_ecmwf=not a.no_ecmwf,
            nwps_product=a.product, gfs_fallback=not a.no_gfs_fallback)
    else:
        run(lat=LAT, lon=LON, name=NAME,
            preset=None if PRESET.startswith("Custom") else PRESET,
            date=DATE, cycle=CYCLE, refresh=REFRESH_DATA,
            plot_default=PLOT_DEFAULT_BUOYS, plot_point=PLOT_POINT_FORECAST,
            show_wind=SHOW_WIND_DIRECTION, show_inventory=SHOW_INVENTORY,
            info_step_h=LABEL_EVERY_H,
            compare_gfs=SHOW_GFS_WAVE, compare_ecmwf=SHOW_ECMWF_WAVE,
            nwps_product=NWPS_FILE, gfs_fallback=GFS_FALLBACK)
