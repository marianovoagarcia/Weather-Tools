# =============================================================================
# Upper-air trend dashboard (operational) - Wyoming archive via Siphon/MetPy
# =============================================================================
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units

warnings.filterwarnings("ignore")

# ----------------------------- Runtime Parameters ----------------------------
STATION_ICAO = os.getenv("STATION_ICAO", "TJSJ")
STATION_WMO = os.getenv("STATION_WMO", "78526")
USE_LATEST = os.getenv("USE_LATEST", "True").lower() in ("true", "1")
DATES_BACK = int(os.getenv("DATES_BACK", "2"))
HOURS_STR = os.getenv("HOURS", "00,12")
PW_UNITS = os.getenv("PW_UNITS", "in")
OUT_DIR = "web_images"
OUT_PATH = f"{OUT_DIR}/upper_air_trends.png"

# Flag thresholds
PW_CHANGE_FLAG_MM = 5.0
DRY_MIDLEVEL_RH = 40.0
DRY_PW_MM = 38.0
HYDROLAPSE_DTD = 10.0
INV_MIN_DT = 0.5
WIND_SHIFT_DEG = 45.0
CAPE_CHANGE_FLAG = 500.0
STABLE_LAPSE_KKM = 1.5
MIN_STABLE_DEPTH_HPA = 15.0
INV_SEARCH_TOP_HPA = 500.0

BARB_LEVELS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
P_GRID = np.arange(1010, 99, -5.0)

# ----------------------------- Time Window -----------------------------------
hours = sorted({int(h) for h in HOURS_STR.replace(" ", "").split(",") if h != ""})
now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

if USE_LATEST:
    # 12:00-23:59 UTC (8:00 AM - 7:59 PM AST) -> 12Z today
    # 00:00-11:59 UTC (8:00 PM - 7:59 AM AST) -> 00Z today
    end_time = now_utc.replace(hour=12 if now_utc.hour >= 12 else 0, minute=0, second=0, microsecond=0)
else:
    s_date = os.getenv("SOUNDING_DATE", now_utc.strftime("%Y-%m-%d"))
    s_hour = int(os.getenv("SOUNDING_HOUR", "12Z")[:2])
    d_obj = datetime.strptime(s_date, "%Y-%m-%d")
    end_time = d_obj.replace(hour=s_hour, minute=0, second=0, microsecond=0)

start_time = end_time - timedelta(days=DATES_BACK)
times = []
d = start_time.replace(hour=0)
while d <= end_time:
    for h in hours:
        tt = d.replace(hour=h)
        if start_time <= tt <= end_time:
            times.append(tt)
    d += timedelta(days=1)


# ----------------------------- Data Access -----------------------------------
def _parse_uwyo_fallback(text):
    txt = re.sub(r"<[^>]+>", "\n", text)
    lines = txt.splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().startswith("PRES") and "HGHT" in l)
    rows = []
    for l in lines[hi + 1:]:
        s = l.strip()
        if not s or set(s) <= set("-") or s[0].isalpha():
            if rows:
                break
            continue
        try:
            vals = [float(l[7 * k:7 * (k + 1)]) if l[7 * k:7 * (k + 1)].strip() else np.nan for k in range(11)]
            tok = l.split()
            if not np.isclose(vals[0], float(tok[0])):
                raise ValueError
        except (ValueError, IndexError):
            tok = l.split()
            if len(tok) != 11:
                if rows:
                    break
                continue
            try:
                vals = [float(t) for t in tok]
            except ValueError:
                if rows:
                    break
                continue
        rows.append(vals)
    cols = ["pressure", "height", "temperature", "dewpoint", "relh", "mixr", "direction", "speed", "thta", "thte", "thtv"]
    return pd.DataFrame(rows, columns=cols)


def fetch_sounding(target_dt, stn_icao, stn_wmo, retries=3):
    last_err = None
    # 1. Try Siphon WyomingUpperAir
    try:
        from siphon.simplewebservice.wyoming import WyomingUpperAir
        for k in range(retries):
            try:
                return WyomingUpperAir.request_data(target_dt, stn_icao)
            except ValueError as e:
                last_err = e
                break
            except Exception as e:
                last_err = e
                time.sleep(4 * (k + 1))
    except Exception as e:
        last_err = e

    # 2. Fallback to UWYO direct TEXT:LIST
    for src in ("BUFR", "UNKNOWN"):
        url = (
            f"https://weather.uwyo.edu/wsgi/sounding?"
            f"datetime={target_dt:%Y-%m-%d}%20{target_dt:%H}:00:00&id={stn_wmo}&src={src}&type=TEXT:LIST"
        )
        for k in range(retries):
            try:
                r = requests.get(url, timeout=30, headers={"User-Agent": "TJSJ-upper-air-trends"})
                if r.status_code == 503:
                    time.sleep(4 * (k + 1))
                    continue
                r.raise_for_status()
                df = _parse_uwyo_fallback(r.text)
                if len(df) >= 10:
                    return df
            except Exception as e:
                last_err = e
                break

    raise RuntimeError(f"Failed fetching {target_dt:%Y-%m-%d %HZ}: {last_err}")


def clean(df):
    df = df.dropna(subset=["pressure", "temperature"]).copy()
    df = df.sort_values("pressure", ascending=False).drop_duplicates("pressure")
    df = df[df["pressure"] >= 100]
    return df.reset_index(drop=True)


# ----------------------------- Diagnostics -----------------------------------
def interp_logp(p, x, pgrid):
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.full_like(pgrid, np.nan, dtype=float)
    lp, xx = np.log(p[m])[::-1], x[m][::-1]
    return np.interp(np.log(pgrid), lp, xx, left=np.nan, right=np.nan)


def detect_layers(p, z, T, Td):
    layers, i, n = [], 0, len(p)
    lapse = -np.diff(T) / (np.diff(z) / 1000.0)
    stable = (lapse <= STABLE_LAPSE_KKM) & (p[1:] >= INV_SEARCH_TOP_HPA)
    while i < n - 1:
        if stable[i]:
            j = i
            while j + 1 < n - 1 and stable[j + 1]:
                j += 1
            base, top = i, j + 1
            depth = p[base] - p[top]
            dT = T[top] - T[base]
            is_inv = dT >= INV_MIN_DT
            if is_inv or depth >= MIN_STABLE_DEPTH_HPA:
                layers.append(dict(
                    p_base=p[base], p_top=p[top], z_base=z[base], z_top=z[top],
                    dT=dT, dTd=(Td[top] - Td[base]) if np.isfinite(Td[base]) and np.isfinite(Td[top]) else np.nan,
                    kind="Inversion" if is_inv else "Stable"
                ))
            i = j + 1
        else:
            i += 1
    return layers


def diagnostics(df):
    p_all = df["pressure"].values
    T_all = df["temperature"].values
    Td_all = df["dewpoint"].values
    z_all = df["height"].values.astype(float)
    m = np.isfinite(Td_all)
    p, T, Td = p_all[m] * units.hPa, T_all[m] * units.degC, Td_all[m] * units.degC
    out = {}

    def safe(key, f):
        try:
            out[key] = f()
        except Exception:
            out[key] = np.nan

    safe("PW_mm", lambda: mpcalc.precipitable_water(p, Td).to("mm").m)
    rh = mpcalc.relative_humidity_from_dewpoint(T, Td).to("percent").m
    out["rh_profile"] = (p.m, rh)
    safe("RH850_500", lambda: np.nanmean(interp_logp(p.m, rh, np.arange(850, 499, -10.0))))
    safe("RH1000_850", lambda: np.nanmean(interp_logp(p.m, rh, np.arange(1000, 849, -10.0))))
    safe("K", lambda: mpcalc.k_index(p, T, Td).m)
    safe("TT", lambda: mpcalc.total_totals_index(p, T, Td).m)

    try:
        mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=100 * units.hPa)
        out["MLCAPE"], out["MLCIN"] = max(float(mlcape.m), 0.0), min(float(mlcin.m), 0.0)
    except Exception:
        out["MLCAPE"], out["MLCIN"] = np.nan, np.nan
    try:
        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
        out["parcel"] = (p, prof)
        out["LI"] = mpcalc.lifted_index(p, T, prof).m.item()
        sbcape, sbcin = mpcalc.cape_cin(p, T, Td, prof)
        out["SBCAPE"], out["SBCIN"] = max(float(sbcape.m), 0.0), min(float(sbcin.m), 0.0)
        lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0])
        lfc_p, _ = mpcalc.lfc(p, T, Td, prof)
        el_p, _ = mpcalc.el(p, T, Td, prof)
        out["LCL"], out["LFC"], out["EL"] = lcl_p.m, lfc_p.m, el_p.m
    except Exception:
        for k in ("parcel", "LI", "SBCAPE", "SBCIN", "LCL", "LFC", "EL"):
            out.setdefault(k, np.nan if k != "parcel" else None)

    try:
        idx = np.where((T_all[:-1] > 0) & (T_all[1:] <= 0))[0][0]
        f = T_all[idx] / (T_all[idx] - T_all[idx + 1])
        out["FZL_p"] = p_all[idx] + f * (p_all[idx + 1] - p_all[idx])
        out["FZL_ft"] = (z_all[idx] + f * (z_all[idx + 1] - z_all[idx])) * 3.28084
    except Exception:
        out["FZL_p"], out["FZL_ft"] = np.nan, np.nan

    wm = np.isfinite(df["speed"].values) & np.isfinite(df["direction"].values)
    pw_, zw = p_all[wm], z_all[wm]
    u, v = mpcalc.wind_components(df["speed"].values[wm] * units.knots, df["direction"].values[wm] * units.degrees)
    out["wind"] = (pw_, zw, u.m, v.m)
    try:
        hgt_agl = (zw - zw[0]) * units.m
        su, sv = mpcalc.bulk_shear(pw_ * units.hPa, u, v, height=hgt_agl, depth=6 * units.km)
        out["SHR06"] = np.hypot(su, sv).to("knots").m
    except Exception:
        out["SHR06"] = np.nan
    try:
        pl = np.arange(1000, 699, -25.0)
        um, vm = np.nanmean(interp_logp(pw_, u.m, pl)), np.nanmean(interp_logp(pw_, v.m, pl))
        out["steer_spd"] = np.hypot(um, vm)
        out["steer_dir"] = mpcalc.wind_direction(um * units.knots, vm * units.knots).m
    except Exception:
        out["steer_spd"], out["steer_dir"] = np.nan, np.nan
    for lev in (850, 700, 500, 250):
        uu, vv = interp_logp(pw_, u.m, np.array([lev]))[0], interp_logp(pw_, v.m, np.array([lev]))[0]
        out[f"wspd{lev}"] = np.hypot(uu, vv)
        out[f"wdir{lev}"] = mpcalc.wind_direction(uu * units.knots, vv * units.knots).m if np.isfinite(uu) else np.nan

    zz = z_all.copy()
    if not np.all(np.isfinite(zz)):
        zz = mpcalc.pressure_to_height_std(p_all * units.hPa).to("m").m
    out["layers"] = detect_layers(p_all, zz, T_all, Td_all)
    invs = [L for L in out["layers"] if L["kind"] == "Inversion"]
    main = max(invs, key=lambda L: L["dT"]) if invs else None
    out["INV_base"] = main["p_base"] if main else np.nan
    out["INV_dT"] = main["dT"] if main else np.nan

    out["T_g"] = interp_logp(p_all, T_all, P_GRID)
    out["Td_g"] = interp_logp(p_all, Td_all, P_GRID)
    out["RH_g"] = interp_logp(p.m, rh, P_GRID)
    out["u_g"] = interp_logp(pw_, u.m, P_GRID)
    out["v_g"] = interp_logp(pw_, v.m, P_GRID)
    out["psfc"] = p_all[0]

    out["HL"] = None
    w = int(50 / 5)
    best = 0.0
    for i in range(len(P_GRID) - w):
        if P_GRID[i + w] < 500:
            break
        drop = out["Td_g"][i] - np.nanmin(out["Td_g"][i:i + w + 1])
        if np.isfinite(drop) and drop > best:
            j = i + int(np.nanargmin(out["Td_g"][i:i + w + 1]))
            best, out["HL"] = drop, (P_GRID[i], P_GRID[j], drop)
    return out


def ang_diff(a, b):
    return (a - b + 180) % 360 - 180


# ----------------------------- Fetch Everything ------------------------------
data = {}
for t in times:
    try:
        df = clean(fetch_sounding(t, STATION_ICAO, STATION_WMO))
        if len(df) < 10:
            raise ValueError("too few levels")
        data[t] = (df, diagnostics(df))
        print(f"OK       {t:%Y-%m-%d %HZ}  ({len(df)} levels)")
    except Exception as e:
        print(f"MISSING  {t:%Y-%m-%d %HZ}  {str(e)[:80]}")

if not data:
    raise SystemExit("No soundings retrieved - check station / dates / Wyoming server.")

avail = sorted(data)
latest = avail[-1]
ref_target = latest - timedelta(hours=24)
ref = ref_target if ref_target in data else (avail[-2] if len(avail) > 1 else None)
missing = [t for t in times if t not in data]
dfL, DL = data[latest]
DR = data[ref][1] if ref else None


def pw_conv(mm):
    return mm / 25.4 if PW_UNITS == "in" else mm


pw_lab = "in" if PW_UNITS == "in" else "mm"

# ----------------------------- Plot Figure -----------------------------------
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": "#555", "axes.linewidth": 0.8,
    "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "legend.labelcolor": "black", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white"
})
C_T, C_TD = "#c0392b", "#1e8449"
C_T_REF, C_TD_REF = "#e59866", "#7dcea0"
C_INV, C_STB = "#f39c12", "#f7dc6f"
GREY = "#9aa0a6"

fig = plt.figure(figsize=(21, 13.5))
gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1.15, 1.15], height_ratios=[1, 1, 0.95],
              left=0.04, right=0.97, top=0.92, bottom=0.05, wspace=0.28, hspace=0.38)
ref_txt = f"{ref:%Y-%m-%d %HZ}" if ref else "none"
fig.suptitle(
    f"{STATION_ICAO}  upper-air trends   {avail[0]:%d %b %HZ} \u2192 {latest:%d %b %Y %HZ}"
    f"   |   latest: {latest:%Y-%m-%d %HZ}   reference: {ref_txt}",
    fontsize=14, fontweight="bold", x=0.04, ha="left"
)

# ============================ A. Skew-T ======================================
skew = SkewT(fig, rotation=45, subplot=gs[:, :2], aspect=96)
ax = skew.ax
skew.plot_dry_adiabats(alpha=0.12, linewidth=0.6)
skew.plot_moist_adiabats(alpha=0.12, linewidth=0.6)
skew.plot_mixing_lines(alpha=0.18, linewidth=0.6)
ax.axvline(0, color="#3498db", lw=0.8, ls="--", alpha=0.6)
for lev in (925, 850, 700, 500, 250):
    ax.axhline(lev, color=GREY, lw=0.5, alpha=0.4)

others = [t for t in avail if t not in (latest, ref)]
for k, t in enumerate(others):
    df = data[t][0]
    a = 0.15 + 0.35 * (k + 1) / max(len(others), 1)
    skew.plot(df["pressure"].values, df["temperature"].values, color=GREY, lw=0.8, alpha=a)
    skew.plot(df["pressure"].values, df["dewpoint"].values, color=GREY, lw=0.8, alpha=a, ls="-")

if ref:
    dfr = data[ref][0]
    skew.plot(dfr["pressure"].values, dfr["temperature"].values, color=C_T_REF, lw=1.6, ls="--")
    skew.plot(dfr["pressure"].values, dfr["dewpoint"].values, color=C_TD_REF, lw=1.6, ls="--")

pL, TL, TdL = dfL["pressure"].values, dfL["temperature"].values, dfL["dewpoint"].values
skew.plot(pL, TL, color=C_T, lw=2.4)
skew.plot(pL, TdL, color=C_TD, lw=2.4)

if DL.get("parcel") is not None:
    pp, prof = DL["parcel"]
    m = np.isfinite(dfL["dewpoint"].values)
    skew.plot(pp, prof, color="k", lw=1.2, ls=(0, (4, 2)))
    try:
        skew.shade_cape(pp, dfL["temperature"].values[m] * units.degC, prof, alpha=0.18)
        skew.shade_cin(pp, dfL["temperature"].values[m] * units.degC, prof,
                       dfL["dewpoint"].values[m] * units.degC, alpha=0.18)
    except Exception:
        pass

for L in DL["layers"]:
    col = C_INV if L["kind"] == "Inversion" else C_STB
    ax.axhspan(L["p_base"], L["p_top"], color=col, alpha=0.30 if L["kind"] == "Inversion" else 0.22, lw=0)
    if L["kind"] == "Inversion" or (np.isfinite(L["dTd"]) and L["dTd"] < -5):
        txt = f"{L['kind'][:3].upper()} {L['p_base']:.0f}\u2013{L['p_top']:.0f}  \u0394T {L['dT']:+.1f}"
        if np.isfinite(L["dTd"]):
            txt += f"  \u0394Td {L['dTd']:+.0f}"
        ax.text(0.015, L["p_top"], txt, transform=ax.get_yaxis_transform(), va="top", ha="left",
                fontsize=7.5, color="black", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))

if DL["HL"] and DL["HL"][2] >= HYDROLAPSE_DTD:
    pb, pt, dd = DL["HL"]
    ax.annotate("", xy=(0.012, pt), xytext=(0.012, pb), xycoords=ax.get_yaxis_transform(),
                arrowprops=dict(arrowstyle="-|>", color="#8e5a2b", lw=2))
    ax.text(0.03, (pb * pt) ** 0.5, f"Hydrolapse {pb:.0f}\u2013{pt:.0f}: Td \u2212{dd:.0f}\u00b0C",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color="black", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))

for key, lab in (("LCL", "LCL"), ("LFC", "LFC"), ("EL", "EL"), ("FZL_p", "0\u00b0C")):
    val = DL.get(key, np.nan)
    if val is not None and np.isfinite(val):
        ax.plot([0.88, 0.91], [val, val], transform=ax.get_yaxis_transform(), color="k", lw=1.5, clip_on=False)
        ax.text(0.915, val, f"{lab} {val:.0f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)


def barbs_std(D, xloc, color, alpha=1.0):
    pw_, _, u, v = D["wind"]
    lv = BARB_LEVELS[(BARB_LEVELS <= pw_[0]) & (BARB_LEVELS >= pw_[-1])]
    ub, vb = interp_logp(pw_, u, lv), interp_logp(pw_, v, lv)
    ok = np.isfinite(ub) & np.isfinite(vb)
    skew.plot_barbs(lv[ok] * units.hPa, ub[ok] * units.knots, vb[ok] * units.knots,
                    xloc=xloc, color=color, alpha=alpha, length=6.5)


barbs_std(DL, 1.0, C_T)

ax.set_ylim(1020, 100)
ax.set_xlim(-35, 40)
ax.set_xlabel("Temperature (\u00b0C)")
ax.set_ylabel("Pressure (hPa)")
ax.set_title("A  Skew-T  \u2014  latest (bold), 24 h earlier (dashed), others (grey, older = fainter)", loc="left")

handles = [Line2D([], [], color=C_T, lw=2.4, label=f"T  {latest:%d/%HZ}"),
           Line2D([], [], color=C_TD, lw=2.4, label=f"Td {latest:%d/%HZ}")]
if ref:
    handles += [Line2D([], [], color=C_T_REF, lw=1.6, ls="--", label=f"T  {ref:%d/%HZ}"),
                Line2D([], [], color=C_TD_REF, lw=1.6, ls="--", label=f"Td {ref:%d/%HZ}")]
handles += [Line2D([], [], color=GREY, lw=1, label=f"{len(others)} other launches"),
            Line2D([], [], color="k", lw=1.2, ls=(0, (4, 2)), label="SB parcel"),
            Patch(color=C_INV, alpha=0.4, label="Inversion"),
            Patch(color=C_STB, alpha=0.4, label=f"Stable (\u2264{STABLE_LAPSE_KKM} K/km)")]
ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)

# Hodograph inset
hax = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1.2)
h = Hodograph(hax, component_range=50)
h.add_grid(increment=10, alpha=0.3, lw=0.5)
for D, col, lw in ((DR, C_T_REF, 1.2), (DL, None, 2.0)):
    if D is None:
        continue
    pw_, zw, u, v = D["wind"]
    agl = (zw - zw[0]) / 1000.0
    mm = agl <= 12
    if col:
        hax.plot(u[mm], v[mm], color=col, lw=lw, ls="--")
    else:
        h.plot_colormapped(u[mm], v[mm], agl[mm], intervals=np.array([0, 1, 3, 6, 9, 12]),
                           colors=["#d73027", "#fc8d59", "#91bfdb", "#4575b4", "#313695"], lw=lw)
hax.set_xticklabels([])
hax.set_yticklabels([])
hax.set_title("Hodograph 0\u201312 km (kt)", fontsize=7.5, fontweight="normal")
hax.text(0.02, 0.02, "0-1-3-6-9-12 km", transform=hax.transAxes, fontsize=6, color="black")

# ============================ B. Time-height =================================
axB = fig.add_subplot(gs[0, 2:])
tnum = mdates.date2num(avail)
RH = np.array([data[t][1]["RH_g"] for t in avail]).T
TT = np.array([data[t][1]["T_g"] for t in avail]).T
if len(avail) > 1:
    cf = axB.contourf(tnum, P_GRID, RH, levels=np.arange(0, 101, 10), cmap="BrBG", extend="neither")
    cs = axB.contour(tnum, P_GRID, TT, levels=[0], colors="#2e86c1", linewidths=1.6)
    axB.clabel(cs, fmt="0\u00b0C", fontsize=7, colors="black")
    cs2 = axB.contour(tnum, P_GRID, TT, levels=[-40, -20], colors="#2e86c1", linewidths=0.6, linestyles=":")
    axB.clabel(cs2, fmt="%d\u00b0C", fontsize=6, colors="black")
    cb = fig.colorbar(cf, ax=axB, pad=0.01, fraction=0.03)
    cb.set_label("RH (%)")

for t in missing:
    x = mdates.date2num(t)
    axB.axvspan(x - 0.12, x + 0.12, color="#ddd", hatch="///", ec="#aaa", lw=0)
    axB.text(x, 980, "missing", rotation=90, fontsize=6, ha="center", va="bottom", color="black")

for t in avail:
    x = mdates.date2num(t)
    for L in data[t][1]["layers"]:
        if L["kind"] == "Inversion":
            axB.plot([x, x], [L["p_base"], L["p_top"]], color=C_INV, lw=6, solid_capstyle="butt", alpha=0.9)
            axB.plot([x, x], [L["p_base"], L["p_top"]], color="k", lw=0.6)

for t in avail:
    pw_, _, u, v = data[t][1]["wind"]
    lv = BARB_LEVELS[(BARB_LEVELS <= pw_[0]) & (BARB_LEVELS >= max(pw_[-1], 150))]
    ub, vb = interp_logp(pw_, u, lv), interp_logp(pw_, v, lv)
    axB.barbs(np.full(lv.shape, mdates.date2num(t)), lv, ub, vb, length=5.2, lw=0.6,
              color="k" if t == latest else "#333")

axB.set_yscale("log")
axB.set_ylim(1015, 150)
axB.set_yticks([1000, 925, 850, 700, 500, 400, 300, 200, 150])
axB.set_yticklabels(["1000", "925", "850", "700", "500", "400", "300", "200", "150"])
axB.minorticks_off()
axB.set_xlim(tnum[0] - 0.3, tnum[-1] + 0.3)
axB.xaxis.set_major_locator(mdates.HourLocator(byhour=hours))
axB.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
axB.set_ylabel("Pressure (hPa)")
axB.set_title("B  Time\u2013height: RH (fill), isotherms, inversions (orange bars), winds (kt)", loc="left")
axB.axvline(mdates.date2num(latest), color="k", lw=0.8, ls=":")

# ============================ C. Index trends ================================
gsC = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, 2:], wspace=0.32, hspace=0.55)
series = [
    ("PW_mm", f"Precipitable water ({pw_lab})", pw_conv, "#1f77b4"),
    ("RH850_500", "Mean RH 850\u2013500 hPa (%)", None, "#2ca02c"),
    ("MLCAPE", "MLCAPE (J/kg)", None, "#d62728"),
    ("K", "K-index", None, "#9467bd"),
    ("INV_base", "Strongest inversion base (hPa)", None, C_INV),
    ("SHR06", "0\u20136 km bulk shear (kt)", None, "#555555"),
]
ref_lines = {"RH850_500": DRY_MIDLEVEL_RH, "PW_mm": pw_conv(DRY_PW_MM)}
for k, (key, lab, conv, col) in enumerate(series):
    a = fig.add_subplot(gsC[k // 3, k % 3])
    x = np.array(avail)
    y = np.array([data[t][1].get(key, np.nan) for t in avail], dtype=float)
    if conv:
        y = conv(y)
    a.plot(x, y, color=col, lw=1.5, alpha=0.8)
    for hh, filled in ((12, True), (0, False)):
        sel = np.array([t.hour == hh for t in avail])
        a.plot(x[sel], y[sel], "o", ms=5.5, color=col, mfc=col if filled else "white", mew=1.4, ls="none")
    if np.isfinite(y[-1]):
        a.annotate(f"{y[-1]:.2f}" if key == "PW_mm" and PW_UNITS == "in" else f"{y[-1]:.0f}",
                   (x[-1], y[-1]), xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold")
    if key in ref_lines:
        a.axhline(ref_lines[key], color="#888", ls="--", lw=0.8)
        a.text(0.01, ref_lines[key], " dry", transform=a.get_yaxis_transform(), fontsize=6.5, color="black", va="bottom")
    if key == "INV_base":
        a.invert_yaxis()
    a.set_title(lab, fontsize=8.5, loc="left")
    a.xaxis.set_major_locator(mdates.DayLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    a.grid(alpha=0.25, lw=0.5)
    a.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.text(0.515, 0.615, "C  Index trends   \u25cf 12Z   \u25cb 00Z", fontsize=10, fontweight="bold",
         transform=fig.transFigure, ha="left")

# ============================ D. 24-h change profile =========================
axD = fig.add_subplot(gs[2, 2])
if DR is not None:
    dT = DL["T_g"] - DR["T_g"]
    dTd = DL["Td_g"] - DR["Td_g"]
    axD.axvline(0, color="#666", lw=0.8)
    axD.fill_betweenx(P_GRID, 0, dTd, where=dTd > 0, color=C_TD, alpha=0.18, lw=0)
    axD.fill_betweenx(P_GRID, 0, dTd, where=dTd < 0, color="#a0522d", alpha=0.18, lw=0)
    axD.plot(dT, P_GRID, color=C_T, lw=1.8, label="\u0394T")
    axD.plot(dTd, P_GRID, color=C_TD, lw=1.8, label="\u0394Td")
    lim = np.nanmax(np.abs(np.concatenate([dT[P_GRID >= 200], dTd[P_GRID >= 200]])))
    lim = 5 if not np.isfinite(lim) else min(max(5, np.ceil(lim)), 30)
    axD.set_xlim(-lim, lim)
    axD.text(0.02, 0.98, "drier", transform=axD.transAxes, fontsize=7, color="black", va="top")
    axD.text(0.98, 0.98, "moister / warmer", transform=axD.transAxes, fontsize=7, color="black", va="top", ha="right")
    axD.legend(loc="lower right", fontsize=7.5)
axD.set_yscale("log")
axD.set_ylim(1015, 200)
axD.set_yticks([1000, 925, 850, 700, 500, 400, 300, 200])
axD.set_yticklabels(["1000", "925", "850", "700", "500", "400", "300", "200"])
axD.minorticks_off()
axD.grid(alpha=0.25, lw=0.5)
axD.set_xlabel("\u00b0C")
axD.set_title(f"D  Change {ref:%d/%HZ}\u2192{latest:%d/%HZ}" if ref else "D  Change (no reference)", loc="left")

# ============================ E. Summary & flags ==============================
axE = fig.add_subplot(gs[2, 3])
axE.axis("off")


def fmt_row(name, key, conv=None, fmt="{:.0f}", unit=""):
    v = DL.get(key, np.nan)
    r = DR.get(key, np.nan) if DR else np.nan
    if conv:
        v, r = conv(v), conv(r)
    vs = fmt.format(v) if np.isfinite(v) else "\u2014"
    if np.isfinite(v) and np.isfinite(r):
        dv = v - r
        arrow = "\u25b2" if dv > 0 else ("\u25bc" if dv < 0 else "\u2022")
        ds = f"{arrow} {fmt.format(abs(dv))}"
    else:
        ds = ""
    return [name, f"{vs} {unit}".strip(), ds]


pwfmt = "{:.2f}" if PW_UNITS == "in" else "{:.0f}"
rows = [
    fmt_row("PW", "PW_mm", pw_conv, pwfmt, pw_lab),
    fmt_row("RH 1000\u2013850", "RH1000_850", None, "{:.0f}", "%"),
    fmt_row("RH 850\u2013500", "RH850_500", None, "{:.0f}", "%"),
    fmt_row("MLCAPE / MLCIN", "MLCAPE", None, "{:.0f}", "J/kg"),
    fmt_row("SBCAPE", "SBCAPE", None, "{:.0f}", "J/kg"),
    fmt_row("Lifted index", "LI", None, "{:.1f}", ""),
    fmt_row("K / TT", "K", None, "{:.0f}", ""),
    fmt_row("Freezing level", "FZL_ft", None, "{:.0f}", "ft"),
    fmt_row("Inversion base", "INV_base", None, "{:.0f}", "hPa"),
    fmt_row("0\u20136 km shear", "SHR06", None, "{:.0f}", "kt"),
    fmt_row("Steering 1000\u2013700", "steer_spd", None, "{:.0f}", "kt"),
]
rows[3][1] = f"{DL['MLCAPE']:.0f} / {DL['MLCIN']:.0f}" if np.isfinite(DL["MLCAPE"]) else "\u2014"
rows[6][1] = f"{DL['K']:.0f} / {DL['TT']:.0f}" if np.isfinite(DL["K"]) else "\u2014"
if np.isfinite(DL["steer_dir"]):
    rows[10][1] = f"{DL['steer_dir']:03.0f}\u00b0 {DL['steer_spd']:.0f} kt"

tbl = axE.table(cellText=rows, colLabels=["", f"{latest:%d/%HZ}", "24 h \u0394"], loc="upper left",
                cellLoc="left", colWidths=[0.40, 0.33, 0.22], bbox=[0, 0.42, 1, 0.58])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#ddd")
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#f2f2f2")
axE.set_title("E  Latest values & flags", loc="left")

flags = []
if DR:
    dpw = DL["PW_mm"] - DR["PW_mm"]
    if np.isfinite(dpw) and abs(dpw) >= PW_CHANGE_FLAG_MM:
        pw_txt = f"{pw_conv(dpw):+.2f} in" if PW_UNITS == "in" else f"{dpw:+.0f} mm"
        flags.append(("\u25b2 Moisture surge" if dpw > 0 else "\u25bc Drying") + f": PW {pw_txt} in 24 h")
    dcape = DL["MLCAPE"] - DR["MLCAPE"]
    if np.isfinite(dcape) and abs(dcape) >= CAPE_CHANGE_FLAG:
        arrow = "\u25b2" if dcape > 0 else "\u25bc"
        flags.append(f"{arrow} MLCAPE {dcape:+.0f} J/kg")
    for lev in (850, 700, 500):
        a1, a0 = DL[f"wdir{lev}"], DR[f"wdir{lev}"]
        if np.isfinite(a1) and np.isfinite(a0) and DL[f"wspd{lev}"] > 5 and abs(ang_diff(a1, a0)) >= WIND_SHIFT_DEG:
            flags.append(f"\u21bb Wind shift {lev} hPa: {a0:03.0f}\u00b0\u2192{a1:03.0f}\u00b0 ({DL[f'wspd{lev}']:.0f} kt)")
    if np.isfinite(DL["INV_base"]) and not np.isfinite(DR["INV_base"]):
        flags.append(f"\u25ac New inversion based {DL['INV_base']:.0f} hPa ({DL['INV_dT']:+.1f}\u00b0C)")
    elif np.isfinite(DL["INV_base"]) and np.isfinite(DR["INV_base"]) and abs(DL["INV_base"] - DR["INV_base"]) >= 25:
        word = "lowered" if DL["INV_base"] > DR["INV_base"] else "lifted"
        flags.append(f"\u25ac Inversion {word}: {DR['INV_base']:.0f}\u2192{DL['INV_base']:.0f} hPa")
    elif not np.isfinite(DL["INV_base"]) and np.isfinite(DR["INV_base"]):
        flags.append("\u25ac Inversion eroded since previous day")
if np.isfinite(DL["RH850_500"]) and DL["RH850_500"] < DRY_MIDLEVEL_RH:
    sal = " \u2013 possible SAL / dry-air intrusion" if DL["PW_mm"] < DRY_PW_MM else ""
    flags.append(f"\u26a0 Dry mid-levels: RH850\u2013500 {DL['RH850_500']:.0f}%{sal}")
if DL["HL"] and DL["HL"][2] >= HYDROLAPSE_DTD:
    flags.append(f"\u2193 Hydrolapse {DL['HL'][0]:.0f}\u2013{DL['HL'][1]:.0f} hPa (Td \u2212{DL['HL'][2]:.0f}\u00b0C)")
if DR and np.isfinite(DL["RH850_500"]) and np.isfinite(DR["RH850_500"]) and abs(DL["RH850_500"] - DR["RH850_500"]) >= 20:
    flags.append(("\u25b2 " if DL["RH850_500"] > DR["RH850_500"] else "\u25bc ") + f"Mid-level RH {DR['RH850_500']:.0f}\u2192{DL['RH850_500']:.0f}% in 24 h")
if missing:
    flags.append(f"\u2716 Missing: {', '.join(t.strftime('%d/%HZ') for t in missing)}")
if not flags:
    flags = ["No significant 24-h changes flagged"]

axE.text(0, 0.38, "\n".join(flags[:8]), transform=axE.transAxes, va="top", fontsize=8.3,
         linespacing=1.55, bbox=dict(fc="#fff8e5", ec="#f0c36d", boxstyle="round,pad=0.5"))

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_PATH, dpi=170, facecolor="white")
plt.close(fig)
print(f"Successfully generated graphic: {OUT_PATH}")
