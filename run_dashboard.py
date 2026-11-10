import os

# Read from environment variables passed by GitHub Actions, with fallback defaults
station = os.getenv("STATION", "TJSJ")
SOUNDING_DATE = os.getenv("SOUNDING_DATE", "2026-03-24")
SOUNDING_HOUR = os.getenv("SOUNDING_HOUR", "12")

import datetime as _dt
_d = _dt.date.fromisoformat(SOUNDING_DATE)
YEAR, MONTH, DAY, HOUR = _d.year, _d.month, _d.day, int(SOUNDING_HOUR[:2])
# Optional: path to a saved UWyo TEXT:LIST file (upload it to Colab). Leave blank to download.
SOUNDING_TEXT_FILE = ''

#@markdown ---
#@markdown **3-day outlook strip**: observed sounding + the GFS change since the sounding time (Archive covers ~3 months).
RUN_GFS = True  #@param {type:"boolean"}
SJU_LAT, SJU_LON = 18.43, -66.00     # TJSJ launch site

#@markdown ---
#@markdown **High cloud (HCDC)**: dense high cloud blocks daytime heating and limits showers and thunderstorms.
USE_HIGH_CLOUD = True  #@param {type:"boolean"}

TODAY_STYLE = 'legacy'  # ['enhanced', 'legacy']
GFS_MAX_AGE_DAYS = 92   # older soundings run on the sounding alone (Open-Meteo keeps ~3 months of past GFS runs)

#**Science options**
LAPSE_RATE_MODE = 'corrected'
APPLY_TERRAIN_CORRECTION = True
GDI_GATE_SEVERE = True
CAPE_OFFSET = 100

#@markdown ---
#@markdown **Graphic**
BRAND_NAME = 'WFO'
IMG_SCALE = 1
DOWNLOAD_IMAGE = False  #@param {type:"boolean"}
OUT_DIR = 'web_images'

# Science constants (Gálvez & Davison 2016; Miller et al. 2019)
import math, re, io, time, json, datetime as dt, warnings
import numpy as np
import pandas as pd
import requests
from IPython.display import display, HTML

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Gálvez–Davison Index constants (eqs. 2.2, 3.8–3.18)
# ---------------------------------------------------------------------------
GDI = dict(
    Lo=2.69e6,      # J kg-1, latent heat of vaporization used in eq. 2.1
    Cpd=1005.7,     # J kg-1 K-1
    alpha=-10.0,    # K, limits excessive GDI with deep moisture at/above 850 hPa (eqs. 3.9-3.10)
    beta=303.0,     # K, lower bound for boundary-layer heat/moisture (eqs. 3.11-3.12)
    gamma=6.5e-2,   # K-1, CBI scaling (eq. 3.13)
    tau=263.15,     # K, 500 hPa warm-core threshold (-10 C) (eq. 3.14)
    mu=-7.0,        # K-1, MWI scaling (eq. 3.14)
    sigma=1.5,      # K-1, Inversion Index scaling (eq. 3.15)
    P1=500.0, P2=9000.0, P3=18.0,   # terrain correction (eq. 3.18)
)

# Table 4 (Gálvez & Davison 2016): GDI value -> expected convective regime
GDI_REGIMES = [
    (-np.inf,  5, 'Strong trade-wind inversion',  'Strong TWI likely; activity limited to light rain showers.',             '#8d99ae'),
    (5,       15, 'Shallow convection',           'Isolated to scattered showers; any thunderstorm brief and isolated.',     '#48cae4'),
    (15,      25, 'Isolated thunderstorms',       'Isolated thunderstorms and/or isolated to scattered showers.',            '#52b788'),
    (25,      35, 'Isolated–scattered T-storms',  'Isolated to scattered thunderstorms and/or showers; locally heavy rain possible.', '#ffd60a'),
    (35,      45, 'Scattered thunderstorms',      'Scattered thunderstorms, scattered to widespread showers; locally heavy rain possible.', '#fb8500'),
    (45,  np.inf, 'Scattered–widespread T-storms','Scattered to widespread thunderstorms; locally heavy rainfall likely.',   '#d00000'),
]

def gdi_regime(v):
    for lo, hi, name, text, color in GDI_REGIMES:
        if lo <= v < hi:
            return dict(name=name, text=text, color=color, lo=lo, hi=hi)
    return dict(name='n/a', text='', color='#999', lo=np.nan, hi=np.nan)

# ---------------------------------------------------------------------------
# Puerto Rico GDI climatology (Miller et al. 2019, 00 UTC TJSJ, 2002-2016)
# Fig. 2: mean 1.5, sigma 18.0, day counts per sigma bin
# ---------------------------------------------------------------------------
PR_GDI_MU, PR_GDI_SD = 1.5, 18.0
PR_GDI_BINS = [(-np.inf, -2, 26), (-2, -1, 795), (-1, 0, 2052), (0, 1, 1446), (1, 2, 743), (2, np.inf, 185)]

def pr_gdi_percentile(v):
    """Empirical percentile of a GDI value within the 2002-2016 TJSJ distribution (Miller et al. Fig. 2 bins),
    linear inside each sigma bin (the open-ended tails use a 1-sigma width)."""
    z = (v - PR_GDI_MU) / PR_GDI_SD
    total = sum(n for *_, n in PR_GDI_BINS)
    cum = 0.0
    for lo, hi, n in PR_GDI_BINS:
        lo_e = hi - 1 if np.isinf(lo) else lo
        hi_e = lo + 1 if np.isinf(hi) else hi
        if z < hi or np.isinf(hi):
            frac = np.clip((z - lo_e) / (hi_e - lo_e), 0, 1)
            return 100 * (cum + frac * n) / total, z
        cum += n
    return 100.0, z

# Puerto Rico seasons (Van Beusekom et al. 2015, as used by Miller et al. 2019)
SEASONS = {'DS': (12, 1, 2, 3), 'ERS': (4, 5, 6, 7), 'LRS': (8, 9, 10, 11)}
SEASON_NAME = {'DS': 'Dry Season (Dec–Mar)', 'ERS': 'Early Rainfall Season (Apr–Jul)', 'LRS': 'Late Rainfall Season (Aug–Nov)'}
def season_of(month):
    return next(k for k, v in SEASONS.items() if month in v)

# Daily R² vs TJUA 3-km echo-top activity (Miller et al. Figs. 3 & 4).
# Used as evidence weights: an index gets a vote proportional to the variance it actually explained in PR.
R2_TSTORM = {
    'DS':  {'GDI': 0.26, 'K': 0.11, 'TT': 0.11, 'CAPE': 0.11},
    'ERS': {'GDI': 0.25, 'K': 0.10, 'TT': 0.06, 'CAPE': 0.02},
    'LRS': {'GDI': 0.27, 'K': 0.09, 'TT': 0.03, 'CAPE': 0.01},
}
R2_DAILY_RAIN = {'DS': 0.09, 'ERS': 0.08, 'LRS': 0.06}      # GDI vs island-mean daily rain (Fig. 3)

# Index -> 0-10 sub-score anchors (piecewise linear). GDI anchors follow Table 4 break points.
ANCHORS = {
    'GDI':  ([-20, 5, 15, 25, 35, 45, 55], [0, 2, 4, 5.5, 7, 8.5, 10]),
    'K':    ([15, 20, 25, 30, 35, 40, 45],  [0, 1.5, 3, 5, 7, 9, 10]),
    'TT':   ([38, 42, 44, 46, 48, 50, 52],  [0, 2, 3.5, 5, 7, 9, 10]),
    'CAPE': ([0, 500, 1000, 1500, 2500, 3500, 4500], [0, 2, 3.5, 5, 7, 9, 10]),
}
def subscore(name, v):
    x, y = ANCHORS[name]
    return float(np.interp(v, x, y))

# Risk categories (unchanged from the original script)
# Status colors (good / warning / critical) — always shown with a text label, never color alone
RISK_CATS = [(0, 20, 'Very low', '#8a8a85'), (20, 40, 'Slight', '#0ca30c'),
             (40, 70, 'Moderate', '#fab219'), (70, 101, 'High', '#d03b3b')]
def risk_cat(ri):
    for lo, hi, name, color in RISK_CATS:
        if lo <= ri < hi:
            return name, color
    return 'High', '#d03b3b'

HAZARDS = ['flood', 'tstorm', 'twind',  'hail','waterspout', 'tornado', 'streamer']

# ---------------------------------------------------------------------------
# High cloud cover (GFS HCDC) — dense cirrus/anvil shields cut solar heating, so the
# sea-breeze/orographic convection that drives PR showers and thunderstorms is limited.
# Tiers in % cloud -> risk points removed (1 point = 10 % Risk Index).
# ---------------------------------------------------------------------------
HC_TIERS = [(75, -3, 'Overcast high cloud'), (50, -2, 'Dense high cloud'), (25, -1, 'Considerable high cloud')]
HC_HAZARDS = ('tstorm', 'flood', 'hail', 'twind', 'tornado')     # convective hazards reduced by high cloud
HC_WINDOW_H = (12, 18)          # UTC hours averaged: 8 a.m.–2 p.m. AST (morning heating window)
HC_RH_RAMP = (50.0, 90.0)       # sounding check: 500–300 hPa mean RH -> 0–100 % estimated high cloud
HC_COLOR = '#86b6ef'            # highlight color for cloud-related reductions

# Display order: these two always first; the others follow by risk (ties keep this order), up to 5 rows
FIXED_HAZARDS = ['flood', 'tstorm']
OTHER_HAZARDS = [h for h in HAZARDS if h not in FIXED_HAZARDS]
# Display order in every card: highest forecast skill first (forecaster's ranking). All 7 are always shown.
SKILL_ORDER = ['flood', 'tstorm', 'twind', 'hail', 'waterspout', 'tornado', 'streamer']
def pick_hazards(ri=None, n=None, floor=None):
    return list(SKILL_ORDER)

# Forecast-skill tiers (display only; scores are not changed). 3 = higher, 2 = medium, 1 = lower.
SKILL_TIER = {'flood': 3, 'tstorm': 3, 'twind': 2, 'hail': 2, 'waterspout': 1, 'tornado': 1, 'streamer': 1}
SKILL_DOTS = {3: '●●●', 2: '●●○', 1: '●○○'}
SKILL_KEY = 'Forecast skill  ●●● higher · ●●○ medium · ●○○ lower (outlined bars)'
def low_skill(h):
    return SKILL_TIER.get(h, 3) == 1

def round10(pct):
    """Round a 0–100 % risk to the nearest 10 % (half up: 35 → 40)."""
    return float(min(100, max(0, 10 * math.floor(pct / 10 + 0.5))))
HAZARD_LABEL = {'tstorm': 'Thunderstorms / Lightning', 'flood': 'Flooding Rain', 'hail': 'Hail',
                'twind': 'Thunderstorm Wind', 'tornado': 'Tornado', 'waterspout': 'Fair-Wx Waterspout',
                'streamer': 'San Juan Streamer'}

# ---------------------------------------------------------------------------
# Local TJSJ sounding climatology (from the original script, unchanged)
# rows: [min, low_50s, high_50s, max]; hazard order HAIL, TWIND, TORNADO, WATERSPOUT, FFW
# ---------------------------------------------------------------------------
CLIMO_HZ = ['hail', 'twind', 'tornado', 'waterspout', 'flood']
CLIMO = {
    'CAPE':  [[1200,1825,2600,3225],[375,1600,2500,3575],[375,1400,2525,3850],[0,600,2075,3825],[500,2000,3500,4000]],
    'PWAT':  [[1.45,1.72,1.93,2.21],[1.41,1.78,2.17,2.7],[1.68,1.80,2.03,2.27],[0.99,1.58,2.0,2.47],[1.5,1.8,2.8,2.9]],
    'KI':    [[20,26,33,38],[14,26,34,42],[21,25,31,38],[3,19,32,39],[20,30,45,46]],
    'SHI':   [[-3.8,-1.8,1,2],[-5.2,-1.8,0.8,4],[-3.8,-1.8,1.8,3.8],[-4.1,-0.5,3.1,8.5],[-4.1,-1.5,0,5]],
    'SWEAT': [[109,162,220,255],[135,185,235,298],[198,200,210,226],[101,168,212,272],[150,200,235,275]],
    'TT':    [[39,44.5,49,51],[41,44.5,47,51],[37.5,42.8,47,50.8],[34.2,41,46,52.2],[39,43,46,52]],
}
SC = dict(  # monthly climatology, index 0 = January
    pwat=[1.5,1.47,1.5,1.7,1.96,1.96,1.9,2.05,2.12,2.15,1.95,1.8],
    lr87=[4.1,3.9,4.3,4.9,5.2,5.3,5.3,5.3,5.4,5.5,5.1,4.5],
    lr87_75=[2.9,2.5,2.9,3.75,4.3,4.6,4.6,4.6,4.8,4.7,4.1,3.1],
    t500=[-8.2,-8.6,-8.5,-7.9,-7.9,-7.9,-7.6,-7.1,-7.4,-7.3,-7.5,-7.7],
    thk75=[2691,2688,2689,2693,2696,2696,2697,2703,2705,2704,2700,2697],
)

# Thermodynamics, sounding container and the GDI
RD, CPD, LV, EPS, G0 = 287.04, 1005.7, 2.501e6, 0.622, 9.80665
_trapz = getattr(np, 'trapezoid', None) or np.trapz

def es_hpa(t_c):
    """Saturation vapour pressure over water (Bolton 1980), hPa."""
    t_c = np.asarray(t_c, float)
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))

def mixr_kgkg(p_hpa, td_c):
    e = es_hpa(td_c)
    return EPS * e / (np.asarray(p_hpa, float) - e)

def td_from_rh(t_c, rh):
    e = np.clip(np.asarray(rh, float), 0.5, 100) / 100 * es_hpa(t_c)
    x = np.log(e / 6.112)
    return 243.5 * x / (17.67 - x)

def rh_from_td(t_c, td_c):
    return 100 * es_hpa(td_c) / es_hpa(t_c)

def theta_k(t_c, p):
    return (np.asarray(t_c, float) + 273.15) * (1000.0 / np.asarray(p, float)) ** (2 / 7)

def t_lcl_k(t_c, td_c):
    """Bolton (1980) eq. 15."""
    T, Td = t_c + 273.15, td_c + 273.15
    return 1.0 / (1.0 / (Td - 56.0) + np.log(T / Td) / 800.0) + 56.0

def theta_e_bolton(p, t_c, td_c):
    """Bolton (1980) eq. 43 equivalent potential temperature, K."""
    r = mixr_kgkg(p, td_c)
    T = t_c + 273.15
    TL = t_lcl_k(t_c, td_c)
    th = T * (1000 / p) ** (0.2854 * (1 - 0.28 * r))
    return th * np.exp((3.376 / TL - 0.00254) * r * 1000 * (1 + 0.81 * r))

def _moist_dTdp(T, p):
    rs = mixr_kgkg(p, T - 273.15)
    return (RD * T + LV * rs) / (p * (CPD + LV**2 * rs * EPS / (RD * T**2)))

def moist_adiabat(T0_k, p0, p1, dp=2.0):
    """Integrate a saturated pseudo-adiabat from p0 to p1 (hPa) with RK4."""
    n = max(1, int(abs(p1 - p0) / dp))
    h = (p1 - p0) / n
    T, p = T0_k, p0
    for _ in range(n):
        k1 = _moist_dTdp(T, p); k2 = _moist_dTdp(T + h*k1/2, p + h/2)
        k3 = _moist_dTdp(T + h*k2/2, p + h/2); k4 = _moist_dTdp(T + h*k3, p + h)
        T += h * (k1 + 2*k2 + 2*k3 + k4) / 6; p += h
    return T

def parcel_temp(p_start, t_c, td_c, p_targets):
    """Parcel temperature (K) lifted from p_start: dry to the LCL, moist above."""
    TL = t_lcl_k(t_c, td_c)
    pL = p_start * (TL / (t_c + 273.15)) ** (CPD / RD)
    out, T_prev, p_prev = [], TL, pL
    for p in p_targets:
        if p >= pL:
            out.append((t_c + 273.15) * (p / p_start) ** (RD / CPD))
        else:
            T_prev = moist_adiabat(T_prev, p_prev, p); p_prev = p
            out.append(T_prev)
    return np.array(out), TL, pL

# ---------------------------------------------------------------------------
# Sounding container
# ---------------------------------------------------------------------------
COLS = ['PRES', 'HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV']
NEEDED_LEVELS = (1000, 950, 925, 850, 700, 500, 400, 300, 250)

class Sounding:
    """Vertical profile with ln(p) interpolation for any level, plus stability indices."""
    def __init__(self, df, indices=None, meta=None, source='RAOB'):
        df = df.copy()
        for c in COLS:
            if c not in df: df[c] = np.nan
        df = df.dropna(subset=['PRES']).sort_values('PRES', ascending=False).drop_duplicates('PRES')
        # fill derivable columns
        m = df.DWPT.isna() & df.RELH.notna() & df.TEMP.notna()
        df.loc[m, 'DWPT'] = td_from_rh(df.TEMP[m], df.RELH[m])
        m = df.RELH.isna() & df.DWPT.notna()
        df.loc[m, 'RELH'] = rh_from_td(df.TEMP[m], df.DWPT[m])
        m = df.MIXR.isna() & df.DWPT.notna()
        df.loc[m, 'MIXR'] = 1000 * mixr_kgkg(df.PRES[m], df.DWPT[m])
        m = df.THTE.isna() & df.DWPT.notna()
        df.loc[m, 'THTE'] = theta_e_bolton(df.PRES[m], df.TEMP[m], df.DWPT[m])
        m = df.THTA.isna() & df.TEMP.notna()
        df.loc[m, 'THTA'] = theta_k(df.TEMP[m], df.PRES[m])
        self.df = df.reset_index(drop=True)
        self.indices = dict(indices or {})
        self.meta = dict(meta or {})
        self.source = source
        self.interpolated = set()
        self.notes = []

    @property
    def psfc(self):
        pm = self.meta.get('psfc_model')
        return float(pm) if pm is not None and np.isfinite(pm) else float(self.df.PRES.max())

    def _interp(self, p, col):
        d = self.df[['PRES', col]].dropna()
        if d.empty: return np.nan
        hit = d[np.isclose(d.PRES, p)]
        if len(hit): return float(hit[col].iloc[0])
        lp = np.log(d.PRES.values[::-1]); v = d[col].values[::-1]
        if not (lp.min() <= np.log(p) <= lp.max()): return np.nan
        self.interpolated.add(p)
        return float(np.interp(np.log(p), lp, v))

    def wind(self, p):
        d = self.df[['PRES', 'DRCT', 'SKNT']].dropna()
        if d.empty: return np.nan, np.nan
        hit = d[np.isclose(d.PRES, p)]
        if len(hit): return float(hit.DRCT.iloc[0]), float(hit.SKNT.iloc[0])
        u = -d.SKNT * np.sin(np.radians(d.DRCT)); v = -d.SKNT * np.cos(np.radians(d.DRCT))
        lp = np.log(d.PRES.values[::-1])
        if not (lp.min() <= np.log(p) <= lp.max()): return np.nan, np.nan
        ui = np.interp(np.log(p), lp, u.values[::-1]); vi = np.interp(np.log(p), lp, v.values[::-1])
        return (math.degrees(math.atan2(-ui, -vi)) % 360, math.hypot(ui, vi))

    def level(self, p):
        L = {c: self._interp(p, c) for c in ('HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'THTE')}
        L['DRCT'], L['SKNT'] = self.wind(p)
        L['PRES'] = p
        return L

    def complete_indices(self):
        """Use UWyo indices when present (your boxplot climatology was built on them);
        otherwise compute them from the profile."""
        ix, L = self.indices, {p: self.level(p) for p in (850, 700, 500)}
        T = {p: L[p]['TEMP'] for p in L}; Td = {p: L[p]['DWPT'] for p in L}
        def need(k): return k not in ix or ix[k] is None or not np.isfinite(ix[k])
        if need('KI'):
            ix['KI'] = (T[850] - T[500]) + Td[850] - (T[700] - Td[700]); self.notes.append('K computed')
        if need('TT'):
            ix['TT'] = T[850] + Td[850] - 2 * T[500]; self.notes.append('TT computed')
        if need('SHI'):
            tp, *_ = parcel_temp(850, T[850], Td[850], [500])
            ix['SHI'] = T[500] - (tp[0] - 273.15); self.notes.append('Showalter computed')
        if need('SWEAT'):
            d8, f8 = self.wind(850); d5, f5 = self.wind(500)
            s = 12 * max(Td[850], 0) + 20 * max(ix['TT'] - 49, 0) + 2 * f8 + f5
            if 130 <= d8 <= 250 and 210 <= d5 <= 310 and d5 - d8 > 0 and f8 >= 15 and f5 >= 15:
                s += 125 * (math.sin(math.radians(d5 - d8)) + 0.2)
            ix['SWEAT'] = s; self.notes.append('SWEAT computed')
        if need('PWAT_MM'):
            d = self.df[['PRES', 'MIXR']].dropna(); d = d[d.PRES >= 300]
            ix['PWAT_MM'] = float(_trapz(d.MIXR.values / 1000, d.PRES.values * 100) / -G0) if len(d) > 2 else np.nan
            self.notes.append('PWAT integrated')
        if need('LCL_T') or need('LCL_P'):
            s = self.df.dropna(subset=['TEMP', 'DWPT']).iloc[0]
            TL = t_lcl_k(s.TEMP, s.DWPT)
            ix['LCL_T'] = TL - 273.15; ix['LCL_P'] = s.PRES * (TL / (s.TEMP + 273.15)) ** (CPD / RD)
        if need('CAPE') or need('LI'):
            self._parcel_cape()
        ix.setdefault('CIN', np.nan)
        return ix

    def _parcel_cape(self):
        d = self.df.dropna(subset=['TEMP', 'DWPT']).query('PRES >= 150')
        s = d.iloc[0]
        tp, *_ = parcel_temp(s.PRES, s.TEMP, s.DWPT, d.PRES.values)
        r_env = mixr_kgkg(d.PRES.values, d.DWPT.values)
        tve = (d.TEMP.values + 273.15) * (1 + 0.61 * r_env)
        r_par = np.minimum(mixr_kgkg(s.PRES, s.DWPT), mixr_kgkg(d.PRES.values, tp - 273.15))
        tvp = tp * (1 + 0.61 * r_par)
        buoy = RD * (tvp - tve) * np.gradient(-np.log(d.PRES.values))
        if not np.isfinite(self.indices.get('CAPE', np.nan)):
            self.indices['CAPE'] = float(buoy[buoy > 0].sum()); self.notes.append('CAPE computed (approx.)')
        if not np.isfinite(self.indices.get('LI', np.nan)):
            t500p, *_ = parcel_temp(s.PRES, s.TEMP, s.DWPT, [500])
            self.indices['LI'] = self.level(500)['TEMP'] - (t500p[0] - 273.15)

# ---------------------------------------------------------------------------
# Gálvez–Davison Index (vectorised: works on scalars or model grids)
# ---------------------------------------------------------------------------
def compute_gdi(T950, T850, T700, T500, r950, r850, r700, r500, psfc=None, terrain=True):
    """Temperatures in K, mixing ratios in kg/kg, psfc in hPa. Returns dict of all terms."""
    c = GDI
    T950, T850, T700, T500 = map(np.asarray, (T950, T850, T700, T500))
    thA = T950 * (1000 / 950) ** (2 / 7)                                                     # 3.2
    thB = 0.5 * (T850 * (1000 / 850) ** (2 / 7) + T700 * (1000 / 700) ** (2 / 7))            # 3.4
    thC = T500 * (1000 / 500) ** (2 / 7)                                                     # 3.6
    rA, rB, rC = np.asarray(r950), 0.5 * (np.asarray(r850) + np.asarray(r700)), np.asarray(r500)  # 3.3, 3.5, 3.7
    k = c['Lo'] / (c['Cpd'] * T850)
    EA = thA * np.exp(k * rA)                                                                # 3.8
    EB = thB * np.exp(k * rB) + c['alpha']                                                   # 3.9
    EC = thC * np.exp(k * rC) + c['alpha']                                                   # 3.10
    ME, LE = EC - c['beta'], EA - c['beta']                                                  # 3.11, 3.12
    CBI = np.where(LE > 0, c['gamma'] * LE * ME, 0.0)                                        # 3.13
    MWI = np.where(T500 - c['tau'] > 0, c['mu'] * (T500 - c['tau']), 0.0)                    # 3.14
    S = T950 - T700                                                                          # 3.16
    D = EB - EA                                                                              # 3.17
    II = np.where(S + D > 0, 0.0, c['sigma'] * (S + D))                                      # 3.15
    if terrain and psfc is not None:
        TC = c['P3'] - c['P2'] / (np.asarray(psfc, float) - c['P1'])                         # 3.18
    else:
        TC = np.zeros_like(CBI)
    total = CBI + MWI + II + TC
    out = dict(GDI=total, CBI=CBI, MWI=MWI, II=II, TC=TC, S=S, D=D, EPTPA=EA, EPTPB=EB, EPTPC=EC, LE=LE, ME=ME)
    return {k_: (float(v) if np.ndim(v) == 0 else v) for k_, v in out.items()}

def gdi_from_sounding(snd, terrain=True):
    L = {p: snd.level(p) for p in (950, 850, 700, 500)}
    T = {p: L[p]['TEMP'] + 273.15 for p in L}
    r = {p: L[p]['MIXR'] / 1000 for p in L}
    g = compute_gdi(T[950], T[850], T[700], T[500], r[950], r[850], r[700], r[500],
                    psfc=snd.psfc, terrain=terrain)
    g['interpolated_950'] = 950 in snd.interpolated
    return g

def detect_inversion(snd, pmax=980, pmin=600, stable_lapse=2.0):
    """Trade-wind-inversion detector (Gálvez & Davison Figs. 2-3): among layers between pmax and pmin
    where temperature falls < 2 °C/km (or rises), return the one with the sharpest drying (largest RH drop),
    falling back to the largest temperature increase."""
    d = snd.df.query('@pmin <= PRES <= @pmax').dropna(subset=['TEMP', 'HGHT']).reset_index(drop=True)
    if len(d) < 3: return None
    lapse = -(d.TEMP.diff() / d.HGHT.diff()) * 1000            # C/km, positive = cooling with height
    layers, cur = [], None
    for i in range(1, len(d)):
        if lapse[i] < stable_lapse:
            cur = [cur[0], i] if cur else [i - 1, i]
        elif cur:
            layers.append(cur); cur = None
    if cur: layers.append(cur)
    if not layers: return None
    def key(lr):
        b, t = d.iloc[lr[0]], d.iloc[lr[1]]
        drh = (t.RELH - b.RELH) if np.isfinite(t.RELH) and np.isfinite(b.RELH) else 0.0
        return (drh, -(t.TEMP - b.TEMP))
    b0, t0 = min(layers, key=key)
    b, t = d.iloc[b0], d.iloc[t0]
    return dict(p_base=b.PRES, p_top=t.PRES, z_base=b.HGHT, z_top=t.HGHT, dT=t.TEMP - b.TEMP,
                lapse=(b.TEMP - t.TEMP) / ((t.HGHT - b.HGHT) / 1000),
                dRH=(t.RELH - b.RELH) if np.isfinite(t.RELH) and np.isfinite(b.RELH) else np.nan,
                dTHTE=(t.THTE - b.THTE) if np.isfinite(t.THTE) and np.isfinite(b.THTE) else np.nan)


# ---------------------------------------------------------------------------
# Extra sounding diagnostics for the upper cards (Convection Playbook ingredients)
# Cached per sounding (snd._xd) because the legacy engine reuses the same profile.
# ---------------------------------------------------------------------------
def wetbulb_k(p, t_c, td_c):
    """Wet-bulb temperature (K) by Normand's rule: lift to the LCL, then descend moist-adiabatically to p."""
    if not (np.isfinite(t_c) and np.isfinite(td_c)): return np.nan
    TL = t_lcl_k(t_c, td_c); pL = p * (TL / (t_c + 273.15)) ** (CPD / RD)
    return moist_adiabat(TL, pL, p, dp=2.0) if pL < p else TL

def _base(snd):
    d = snd.df.dropna(subset=['TEMP', 'DWPT', 'HGHT']).sort_values('PRES', ascending=False)
    return d, float(d.HGHT.iloc[0])

def _fine(d, p_top=150.0, step=0.005):
    """Profile interpolated to a fine ln-p grid (≈ 0.5 % steps) from the surface to p_top."""
    pf = np.exp(np.arange(np.log(d.PRES.values[0]), np.log(p_top), -step))
    lp = np.log(d.PRES.values[::-1])
    f = lambda c: np.interp(np.log(pf), lp, d[c].values[::-1])
    return pf, f('TEMP'), f('DWPT'), f('HGHT')

def _crossing_height(z, v, thresh=0.0):
    """Height where v first falls through thresh going up (z, v ordered bottom-up)."""
    for i in range(1, len(v)):
        if v[i - 1] > thresh >= v[i]:
            return float(z[i - 1] + (z[i] - z[i - 1]) * (v[i - 1] - thresh) / (v[i - 1] - v[i]))
    return np.nan

def _uv(drct, sknt):
    return -sknt * np.sin(np.radians(drct)), -sknt * np.cos(np.radians(drct))

def _uv_at_height(snd, z_agl):
    d = snd.df.dropna(subset=['HGHT', 'DRCT', 'SKNT']).sort_values('PRES', ascending=False)
    if d.empty: return np.nan, np.nan
    z0 = float(snd.df.dropna(subset=['HGHT']).sort_values('PRES', ascending=False).HGHT.iloc[0])
    u, v = _uv(d.DRCT.values, d.SKNT.values)
    z = d.HGHT.values - z0
    return float(np.interp(z_agl, z, u)), float(np.interp(z_agl, z, v))

def _parcel_areas(pf, Tf, Tdf, zf, p0, t0, td0):
    """Positive buoyant energy (J/kg, virtual temperature) of a parcel from (p0, t0, td0) on the fine grid:
    total CAPE, 0–3 km CAPE, and CAPE where the environment is −10…−30 °C (charge-separation layer)."""
    tp, *_ = parcel_temp(p0, t0, td0, pf)
    tve = (Tf + 273.15) * (1 + 0.61 * mixr_kgkg(pf, Tdf))
    tvp = tp * (1 + 0.61 * np.minimum(mixr_kgkg(p0, td0), mixr_kgkg(pf, tp - 273.15)))
    b = np.clip(RD * (tvp - tve), 0, None)
    lnp = -np.log(pf)                                   # increases with height
    seg = lambda m: float(_trapz(np.where(m, b, 0.0), lnp))
    return seg(np.ones_like(b, bool)), seg(zf <= 3000), seg((Tf <= -10) & (Tf >= -30))

def extra_diag(snd):
    """Playbook ingredients not covered by the classic indices (cached on the sounding)."""
    if getattr(snd, '_xd', None) is not None:
        return snd._xd
    X = {}
    d, z0 = _base(snd)
    pf, Tf, Tdf, Zf = _fine(d); zf = Zf - z0
    psfc = float(d.PRES.iloc[0])
    # Mixed-layer parcel: pressure-weighted mean θ and mixing ratio of the lowest 100 hPa (MetPy convention)
    m = pf >= psfc - 100
    w = np.gradient(-pf)[m]
    th = np.average(theta_k(Tf[m], pf[m]), weights=w); rm = np.average(mixr_kgkg(pf[m], Tdf[m]), weights=w)
    t_ml = th * (psfc / 1000) ** (2 / 7) - 273.15
    e = rm * psfc / (EPS + rm); x = np.log(e / 6.112); td_ml = 243.5 * x / (17.67 - x)
    X['MLCAPE'], X['CAPE03'], X['CAPE_LTG'] = _parcel_areas(pf, Tf, Tdf, zf, psfc, t_ml, td_ml)
    # LCL (mixed-layer), freezing level, wet-bulb zero, warm cloud depth
    TL = t_lcl_k(t_ml, td_ml); pL = psfc * (TL / (t_ml + 273.15)) ** (CPD / RD)
    X['LCL_Z'] = float(np.interp(np.log(pL), np.log(pf[::-1]), zf[::-1]))
    X['FZL_Z'] = _crossing_height(zf, Tf, 0.0)
    sel = slice(None, None, 10)                          # wet bulb every ~5 % in p is plenty
    twz = np.array([wetbulb_k(p_, t_, td_) - 273.15 for p_, t_, td_ in zip(pf[sel], Tf[sel], Tdf[sel])])
    X['WBZ_Z'] = _crossing_height(zf[sel], twz, 0.0)
    X['WCD'] = (X['FZL_Z'] - X['LCL_Z']) / 1000 if np.isfinite(X['FZL_Z']) else np.nan
    # Mean RH 850–500 hPa
    lay = (pf <= 850) & (pf >= 500)
    X['RH85'] = float(np.mean(rh_from_td(Tf[lay], Tdf[lay]))) if lay.any() else np.nan
    # Corfidi et al. (1996) / Corfidi (2003): mean cloud-layer wind 850–300 hPa (Vcl), low-level jet ≈ 850 hPa wind (Vllj);
    # upwind (back-building) propagation vector = Vcl − Vllj; downwind = 2·Vcl − Vllj. Speeds in kt.
    uv = [_uv(*snd.wind(p)) for p in (850, 700, 500, 400, 300)]
    ucl = np.nanmean([a for a, _ in uv]); vcl = np.nanmean([b for _, b in uv])
    ul, vl = _uv(*snd.wind(850))
    X['CORFIDI_UP'] = float(np.hypot(ucl - ul, vcl - vl))
    X['CORFIDI_DN'] = float(np.hypot(2 * ucl - ul, 2 * vcl - vl))
    X['CL_WIND'] = float(np.hypot(ucl, vcl))
    # 0–6 km bulk shear and lowest-level wind, kt
    us, vs = _uv_at_height(snd, 0); u6, v6 = _uv_at_height(snd, 6000)
    X['SHEAR06'] = float(np.hypot(u6 - us, v6 - vs))
    X['SFC_WIND'] = float(np.hypot(us, vs))
    # DCAPE (Gilmore & Wicker 1998; MetPy convention): saturated parcel at the wet-bulb temperature of the
    # minimum-θe sounding level in 700–500 hPa, descending moist-adiabatically to the surface; area of T_env − T_parcel.
    P, TT, TD = d.PRES.values, d.TEMP.values, d.DWPT.values
    mid = (P <= 700) & (P >= 500)
    X['DCAPE'] = X['DCAPE_P'] = np.nan
    if mid.any():
        the = theta_e_bolton(P, TT, TD); i0 = np.where(mid)[0][np.argmin(the[mid])]
        p_start = P[i0]; tw = wetbulb_k(p_start, TT[i0], TD[i0])
        pd_ = np.exp(np.arange(np.log(p_start), np.log(psfc), 0.005)); pd_ = np.append(pd_, psfc)
        Te = np.interp(np.log(pd_), np.log(P[::-1]), TT[::-1]) + 273.15
        Tp = [tw]
        for k in range(1, len(pd_)):
            Tp.append(moist_adiabat(Tp[-1], pd_[k - 1], pd_[k], dp=2.0))
        X['DCAPE'] = float(_trapz(np.clip(RD * (Te - np.array(Tp)), 0, None), np.log(pd_)))
        X['DCAPE_P'] = float(p_start)
    snd._xd = X
    return X

# Data access: TJSJ RAOB (UWyo) and GFS (Open-Meteo)
UWYO_INDEX_MAP = {      # UWyo label -> our key
    'Showalter index': 'SHI', 'Lifted index': 'LI', 'SWEAT index': 'SWEAT', 'K index': 'KI',
    'Totals totals index': 'TT', 'Convective Available Potential Energy': 'CAPE',
    'Convective Inhibition': 'CIN', 'Temp [K] of the Lifted Condensation Level': 'LCL_T_K',
    'Pres [hPa] of the Lifted Condensation Level': 'LCL_P',
    'Precipitable water [mm] for entire sounding': 'PWAT_MM', 'Station elevation': 'ELEV',
}

# New UWyo wsgi pages list indices by short code (e.g. "KINX  K index  32.4  ..."); same mapping as the legacy script
UWYO_CODE_MAP = [('SHOW', 'SHI'), ('LIFT', 'LI'), ('LFVT', 'LI'), ('SWET', 'SWEAT'), ('KINX', 'KI'), ('TOTL', 'TT'),
                 ('MUCAPE', 'CAPE'), ('CAPE', 'CAPE'), ('MUCIN', 'CIN'), ('CINS', 'CIN'),
                 ('LCLT', 'LCL_T_K'), ('LCLP', 'LCL_P'), ('PWAT', 'PWAT_MM')]

def parse_uwyo_text(text):
    """Parse a UWyo TEXT:LIST page. Uses the fixed 7-character columns, so rows with blank fields
    (missing dewpoint or wind) no longer shift values into the wrong column — a bug in the old
    split()-based parser."""
    txt = re.sub(r'<[^>]+>', '\n', text)
    lines = txt.splitlines()
    try:
        hi = next(i for i, l in enumerate(lines) if l.strip().startswith('PRES') and 'HGHT' in l)
    except StopIteration:
        raise ValueError('No sounding table found in the response.')
    rows = []
    for l in lines[hi + 1:]:
        s = l.strip()
        if not s:
            if rows: break
            continue
        if set(s) <= set('-') or s[0].isalpha():
            if rows: break
            continue
        try:
            vals = [float(l[7*k:7*(k+1)]) if l[7*k:7*(k+1)].strip() else np.nan for k in range(11)]
            tok = l.split()
            if not np.isclose(vals[0], float(tok[0])):          # not 7-char aligned (newer pages): use tokens
                raise ValueError
        except (ValueError, IndexError):
            tok = l.split()
            try:
                if len(tok) != 11: raise ValueError
                vals = [float(t) for t in tok]
            except ValueError:
                if rows: break
                continue
        rows.append(vals)
    df = pd.DataFrame(rows, columns=COLS)
    indices = {}
    for l in lines:
        m = re.match(r'\s*([A-Za-z][^:]{2,80}?):\s*(-?\d+(?:\.\d+)?)\s*$', l)
        if m and m.group(1).strip() in UWYO_INDEX_MAP:
            indices[UWYO_INDEX_MAP[m.group(1).strip()]] = float(m.group(2))
    plain = '\n'.join(lines)
    for code, key in UWYO_CODE_MAP:                       # newer format; first match per key wins (MUCAPE before CAPE)
        if key in indices: continue
        m = re.search(rf'^\s*{code}\s+.*?\s+([-+]?\d*\.?\d+)', plain, re.MULTILINE)
        if m:
            indices[key] = float(m.group(1))
    if 'LCL_T_K' in indices:
        indices['LCL_T'] = indices.pop('LCL_T_K') - 273.15
    if len(df) < 8:
        raise ValueError(f'Only {len(df)} levels parsed — sounding incomplete or unavailable.')
    return df, indices

def latest_synoptic(now=None):
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    t = now - dt.timedelta(hours=3)          # allow ~3 h for the data to reach UWyo
    return t.replace(hour=0 if t.hour < 12 else 12, minute=0, second=0, microsecond=0)

def fetch_uwyo(station, when, tries=2):
    urls = [
        (f"https://weather.uwyo.edu/wsgi/sounding?datetime={when:%Y-%m-%d}%20{when:%H}:00:00"
         f"&id={station}&src=BUFR&type=TEXT:LIST"),
        (f"http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={when:%Y}"
         f"&MONTH={when:%m}&FROM={when:%d%H}&TO={when:%d%H}&STNM={station}"),
        (f"https://weather.uwyo.edu/wsgi/sounding?datetime={when:%Y-%m-%d}%20{when:%H}:00:00"
         f"&id={station}&src=UNKNOWN&type=TEXT:LIST"),
        (f"http://weather.uwyo.edu/cgi-bin/bufrraob.py?src=bufr&datetime={when:%Y-%m-%d}+{when:%H}:00:00"
         f"&id={station}&type=TEXT:LIST"),
    ]
    errors = []
    for url in urls:
        for k in range(tries):
            try:
                r = requests.get(url, timeout=40, headers={'User-Agent': 'TJSJ-risk-notebook (Colab)'})
                if r.status_code == 503:            # UWyo throttles; back off and retry
                    time.sleep(4 * (k + 1)); continue
                r.raise_for_status()
                df, ix = parse_uwyo_text(r.text)
                return df, ix, url
            except Exception as e:
                errors.append(f'{url.split("?")[0]}: {e}')
                break
    raise RuntimeError('Could not get the sounding from UWyo:\n  ' + '\n  '.join(errors))

def load_raob():
    when = latest_synoptic() if USE_LATEST_SOUNDING else dt.datetime(YEAR, MONTH, DAY, HOUR)
    if SOUNDING_TEXT_FILE:
        df, ix = parse_uwyo_text(open(SOUNDING_TEXT_FILE, encoding='utf-8', errors='ignore').read())
        src = SOUNDING_TEXT_FILE
    else:
        df, ix, src = fetch_uwyo(STATION_ID, when)
    snd = Sounding(df, ix, meta=dict(station=STATION_ID, valid=when, url=src), source='RAOB')
    snd.complete_indices()
    return snd

# ---------------------------------------------------------------------------
# GFS through Open-Meteo (0.25° pressure-level fields, no key, no GRIB libraries)
# ---------------------------------------------------------------------------
OM_URL = 'https://api.open-meteo.com/v1/gfs'
PT_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300, 250]
WIND_LEVELS = [1000, 925, 850, 700, 500, 250]

def _om_get(params, retries=3):
    for k in range(retries):
        r = requests.get(OM_URL, params=params, timeout=90)
        if r.status_code == 429:
            time.sleep(10 * (k + 1)); continue
        if r.status_code >= 400:
            raise RuntimeError(f'Open-Meteo {r.status_code}: {r.text[:300]}')
        return r.json()
    raise RuntimeError('Open-Meteo rate limit — wait a minute and re-run.')

def fetch_gfs_point(lat, lon, t_obs, days_ahead=3):
    """Hourly GFS point profiles covering the sounding time and the next `days_ahead` days.
    `past_days` pulls Open-Meteo's archive of earlier GFS runs so the sounding hour is included."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    past = int(np.clip((now - t_obs).days + 1, 0, 92))
    fwd = int(np.clip((t_obs + dt.timedelta(days=days_ahead + 1) - now).days + 1, 1, 16))
    hourly = []
    for p in PT_LEVELS:
        hourly += [f'temperature_{p}hPa', f'relative_humidity_{p}hPa', f'geopotential_height_{p}hPa']
    for p in WIND_LEVELS:
        hourly += [f'wind_speed_{p}hPa', f'wind_direction_{p}hPa']
    hourly += ['surface_pressure', 'cape', 'lifted_index', 'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m']
    js = _om_get(dict(latitude=lat, longitude=lon, hourly=','.join(hourly), past_days=past, forecast_days=fwd,
                      wind_speed_unit='kn', timezone='GMT', cell_selection='sea', elevation='nan'))
    h = pd.DataFrame(js['hourly']); h['time'] = pd.to_datetime(h['time'])
    return h

def hcdc_window(h, day_utc):
    """Mean GFS high cloud cover (HCDC, %) over the 8 a.m.–2 p.m. AST heating window (12–18Z) of `day_utc`."""
    if h is None or 'cloud_cover_high' not in h:
        return np.nan
    t0 = dt.datetime.combine(day_utc, dt.time(HC_WINDOW_H[0])); t1 = dt.datetime.combine(day_utc, dt.time(HC_WINDOW_H[1]))
    v = h.loc[(h.time >= t0) & (h.time <= t1), 'cloud_cover_high'].astype(float)
    return float(v.mean()) if v.notna().any() else np.nan

def hcdc_from_sounding(snd):
    """Sounding check: high cloud estimated from mean 500–300 hPa RH (linear ramp 50 % RH -> 0, 90 % RH -> 100 %).
    RH is with respect to water, so thin cirrus can be underestimated; it only ever raises the GFS value."""
    d = snd.df.query('300 <= PRES <= 500')['RELH'].dropna()
    if d.empty:
        return np.nan, np.nan
    rh = float(d.mean())
    lo, hi = HC_RH_RAMP
    return float(np.clip((rh - lo) / (hi - lo), 0, 1) * 100), rh

def high_cloud_for(snd, h=None, use_sounding=True):
    """High-cloud input for one assessment. Afternoon: GFS window mean for that day.
    12Z sounding / 18Z afternoon: that day's 12–18Z window. 00Z (8 p.m. AST) sounding: the next morning's window,
    since 00Z already falls on the next UTC date. The sounding time is maxed with the sounding RH estimate."""
    t = snd.meta['valid']
    day = t.date() if t.hour <= HC_WINDOW_H[1] else (t + dt.timedelta(days=1)).date()
    gfs = hcdc_window(h, day)
    est, rh = hcdc_from_sounding(snd) if use_sounding else (np.nan, np.nan)
    vals = [v for v in (gfs, est) if np.isfinite(v)]
    pct = max(vals) if vals else np.nan
    src = ('GFS HCDC' if np.isfinite(gfs) and (not np.isfinite(est) or gfs >= est) else 'est. from sounding RH 500–300 hPa') if vals else 'n/a'
    return dict(pct=pct, gfs=gfs, snd=est, rh53=rh, src=src, day=day)

# ---------------------------------------------------------------------------
# Synoptic lift from GFS: 200 hPa divergence and 700/200 hPa relative vorticity from a 5-point stencil
# (center ± LIFT_DX degrees), averaged over the 8 a.m.–2 p.m. AST window. Raw GFS (no sounding calibration).
# ---------------------------------------------------------------------------
LIFT_DX = 1.5                     # degrees between stencil points (~160 km)
LIFT_DIV_STRONG, LIFT_DIV_MOD = 2e-5, 1e-5      # s-1, 200 hPa divergence (Playbook red flag: >= 2e-5)
LIFT_VORT_MOD = 1e-5              # s-1, 700 hPa cyclonic vorticity: tropical wave / trough
TUTT_VORT = 2e-5                  # s-1, 200 hPa cyclonic vorticity: TUTT / upper trough overhead
LIFT_MULT = {'strong': 1.0, 'moderate': 1.0, 'weak': 0.85, 'none_day': 0.85, 'none_night': 0.55}
LIFT_HAZARDS = ('tstorm', 'flood', 'hail', 'twind', 'tornado')

def fetch_gfs_stencil(lat, lon, t_obs, days_ahead=3, dx=LIFT_DX):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    past = int(np.clip((now - t_obs).days + 1, 0, 92))
    fwd = int(np.clip((t_obs + dt.timedelta(days=days_ahead + 1) - now).days + 1, 1, 16))
    pts = {'C': (lat, lon), 'N': (lat + dx, lon), 'S': (lat - dx, lon), 'E': (lat, lon + dx), 'W': (lat, lon - dx)}
    hourly = ['wind_speed_200hPa', 'wind_direction_200hPa', 'wind_speed_700hPa', 'wind_direction_700hPa']
    js = _om_get(dict(latitude=','.join(str(a) for a, _ in pts.values()), longitude=','.join(str(b) for _, b in pts.values()),
                      hourly=','.join(hourly), past_days=past, forecast_days=fwd, wind_speed_unit='ms',
                      timezone='GMT', cell_selection='nearest'))
    js = js if isinstance(js, list) else [js]
    out = {}
    for k, j in zip(pts, js):
        h = pd.DataFrame(j['hourly']); h['time'] = pd.to_datetime(h['time']); out[k] = h
    out['_dx'] = dx; out['_lat'] = lat
    return out

def _window_uv(h, day, p):
    t0 = dt.datetime.combine(day, dt.time(HC_WINDOW_H[0])); t1 = dt.datetime.combine(day, dt.time(HC_WINDOW_H[1]))
    w = h[(h.time >= t0) & (h.time <= t1)]
    if w.empty: return np.nan, np.nan
    u, v = _uv(w[f'wind_direction_{p}hPa'].astype(float).values, w[f'wind_speed_{p}hPa'].astype(float).values)
    return float(np.nanmean(u)), float(np.nanmean(v))

def gfs_env_for(snd, h_center=None, stencil=None):
    """Lift class + multiplier and the GFS 10 m wind for one assessment (same window/day rule as high cloud)."""
    t = snd.meta['valid']
    day = t.date() if t.hour <= HC_WINDOW_H[1] else (t + dt.timedelta(days=1)).date()
    night = (t.hour == 0)                                   # 00Z = 8 p.m. AST sounding: evening/overnight regime
    E = dict(day=day, lift=None, mult=1.0, div200=np.nan, vort700=np.nan, vort200=np.nan, sfc_wind=np.nan)
    if h_center is not None and 'wind_speed_10m' in h_center:
        t0 = dt.datetime.combine(day, dt.time(HC_WINDOW_H[0])); t1 = dt.datetime.combine(day, dt.time(HC_WINDOW_H[1]))
        w = h_center.loc[(h_center.time >= t0) & (h_center.time <= t1), 'wind_speed_10m'].astype(float)
        E['sfc_wind'] = float(w.mean()) if w.notna().any() else np.nan
    if stencil:
        R_E = 6.371e6; dxm = R_E * math.cos(math.radians(stencil['_lat'])) * math.radians(stencil['_dx']); dym = R_E * math.radians(stencil['_dx'])
        def dd(p):
            uE, vE = _window_uv(stencil['E'], day, p); uW, vW = _window_uv(stencil['W'], day, p)
            uN, vN = _window_uv(stencil['N'], day, p); uS, vS = _window_uv(stencil['S'], day, p)
            div = (uE - uW) / (2 * dxm) + (vN - vS) / (2 * dym)
            vort = (vE - vW) / (2 * dxm) - (uN - uS) / (2 * dym)
            return div, vort
        E['div200'], E['vort200'] = dd(200); _, E['vort700'] = dd(700)
        dv, vo = E['div200'], E['vort700']
        if np.isfinite(dv):
            if dv >= LIFT_DIV_STRONG or (dv >= LIFT_DIV_MOD and vo >= LIFT_VORT_MOD): lift = 'strong'
            elif dv >= LIFT_DIV_MOD or vo >= LIFT_VORT_MOD: lift = 'moderate'
            elif dv >= 0: lift = 'weak'
            else: lift = 'none'
            E['lift'] = lift
            E['mult'] = LIFT_MULT['none_night' if night else 'none_day'] if lift == 'none' else LIFT_MULT[lift]
    return E

def gfs_sounding_at(h, when):
    """GFS point profile valid at `when` (exact hour), or None if not in the table."""
    row = h[h.time == pd.Timestamp(when)]
    if row.empty:
        return None
    row = row.iloc[0]
    recs = []
    for p in PT_LEVELS:
        rec = dict(PRES=p, HGHT=row.get(f'geopotential_height_{p}hPa'),
                   TEMP=row.get(f'temperature_{p}hPa'), RELH=row.get(f'relative_humidity_{p}hPa'))
        if p in WIND_LEVELS:
            rec['DRCT'] = row.get(f'wind_direction_{p}hPa'); rec['SKNT'] = row.get(f'wind_speed_{p}hPa')
        recs.append(rec)
    df = pd.DataFrame(recs).astype(float)
    if df.TEMP.isna().sum() > 3:
        return None
    s = Sounding(df, dict(CAPE=row.get('cape', np.nan), LI=row.get('lifted_index', np.nan)),
                 meta=dict(valid=row.time.to_pydatetime(), psfc_model=row.get('surface_pressure')), source='GFS')
    s.complete_indices()
    return s

def _td_from_mixr(p, r_gkg):
    r = np.maximum(np.asarray(r_gkg, float), 0.01) / 1000
    e = r * p / (EPS + r)
    x = np.log(e / 6.112)
    return 243.5 * x / (17.67 - x)

def _uv(drct, sknt):
    return -sknt * np.sin(np.radians(drct)), -sknt * np.cos(np.radians(drct))

CAL_INDEX_KEYS = ('KI', 'TT', 'SHI', 'SWEAT', 'PWAT_MM', 'CAPE', 'LI')

def calibrate_with_gfs(obs, g0, gt):
    """Sounding-calibrated forecast profile:  observed + [GFS(target) − GFS(sounding time)].
    Temperature, mixing ratio, height and wind deltas are interpolated in ln p onto every observed level;
    mixing ratio is kept between 0 and saturation. Stability indices get the same treatment
    (observed UWyo index + GFS index change), so they stay on the scale your climatology was built on."""
    d = obs.df.copy()
    lp_obs = np.log(d.PRES.values)
    def delta(col, src0=g0, srct=gt):
        a, b = src0.df[['PRES', col]].dropna(), srct.df[['PRES', col]].dropna()
        m = a.merge(b, on='PRES', suffixes=('0', 't')).sort_values('PRES')
        if m.empty: return np.zeros(len(d))
        return np.interp(lp_obs, np.log(m.PRES.values), (m[col + 't'] - m[col + '0']).values)
    d['TEMP'] = d.TEMP + delta('TEMP')
    d['HGHT'] = d.HGHT + delta('HGHT')
    r_new = d.MIXR + delta('MIXR')
    r_sat = 1000 * mixr_kgkg(d.PRES.values, d.TEMP.values)
    d['MIXR'] = np.clip(r_new, 0.01, r_sat)
    d['DWPT'] = _td_from_mixr(d.PRES.values, d.MIXR.values)
    # winds: add the GFS change in u and v
    w0, wt = g0.df.dropna(subset=['DRCT', 'SKNT']), gt.df.dropna(subset=['DRCT', 'SKNT'])
    m = w0.merge(wt, on='PRES', suffixes=('0', 't')).sort_values('PRES')
    if not m.empty:
        u0, v0 = _uv(m.DRCT0, m.SKNT0); ut, vt = _uv(m.DRCTt, m.SKNTt)
        du = np.interp(lp_obs, np.log(m.PRES.values), (ut - u0).values)
        dv = np.interp(lp_obs, np.log(m.PRES.values), (vt - v0).values)
        u, v = _uv(d.DRCT, d.SKNT); u, v = u + du, v + dv
        d['SKNT'] = np.hypot(u, v); d['DRCT'] = np.degrees(np.arctan2(-u, -v)) % 360
    for c in ('RELH', 'THTE', 'THTA', 'THTV'):
        d[c] = np.nan                                  # recomputed from the new T and Td
    ix = {}
    for k in CAL_INDEX_KEYS:
        a, b, o = g0.indices.get(k, np.nan), gt.indices.get(k, np.nan), obs.indices.get(k, np.nan)
        if all(np.isfinite(x) for x in (a, b, o)):
            ix[k] = o + (b - a)
    ix['CAPE'] = max(ix.get('CAPE', 0.0), 0.0)
    ix['PWAT_MM'] = max(ix.get('PWAT_MM', 0.0), 0.0)
    s = Sounding(d, ix, meta=dict(obs.meta, valid=gt.meta['valid']), source='RAOB-cal')
    s.complete_indices()
    return s

def calibrated_outlook(obs, h, days=3, hour_utc=18):
    """Calibrated profiles for the next `days` afternoons (18Z = 2 p.m. AST), starting the day of the sounding."""
    g0 = gfs_sounding_at(h, obs.meta['valid'])
    if g0 is None:
        raise RuntimeError(f"GFS has no profile at the sounding time {obs.meta['valid']:%Y-%m-%d %HZ}.")
    d0 = (obs.meta['valid'] - dt.timedelta(hours=4)).date()
    out = []
    for k in range(days + 1):
        t = dt.datetime.combine(d0 + dt.timedelta(days=k), dt.time(hour_utc))
        if t <= obs.meta['valid']:
            continue
        gt = gfs_sounding_at(h, t)
        if gt is not None:
            out.append(calibrate_with_gfs(obs, g0, gt))
        if len(out) == days:
            break
    return out

# Diagnostics and the evidence-based risk engine
def _vec_mean_wind(snd, levels):
    """Scalar-mean speed (what the legacy thresholds expect) + vector-mean direction
    (the legacy arithmetic mean of directions gave 180° for winds of 350° and 10°)."""
    w = [snd.wind(p) for p in levels]
    spd = np.nanmean([s for _, s in w])
    u = np.nansum([-s * math.sin(math.radians(d)) for d, s in w])
    v = np.nansum([-s * math.cos(math.radians(d)) for d, s in w])
    return spd, math.degrees(math.atan2(-u, -v)) % 360

def diagnostics(snd, lapse_mode='corrected', cape_offset=100):
    L = {p: snd.level(p) for p in NEEDED_LEVELS}
    T = lambda p: L[p]['TEMP']; Z = lambda p: L[p]['HGHT']
    ix = snd.indices
    D = dict(month=snd.meta['valid'].month)
    D.update(SHI=ix['SHI'], LI=ix.get('LI', np.nan), SWEAT=ix['SWEAT'], KI=ix['KI'], TT=ix['TT'],
             CAPE=ix['CAPE'] - (cape_offset if snd.source.startswith('RAOB') else 0), CIN=ix.get('CIN', np.nan),
             LCL_T=ix.get('LCL_T', np.nan), LCL_P=ix.get('LCL_P', np.nan),
             PWAT=round(ix['PWAT_MM'] / 25.4, 2))
    D['thick1000_500'] = Z(500) - Z(1000)
    D['thick700_500'] = Z(500) - Z(700)
    D['dthe'] = L[1000]['THTE'] - min(L[500]['THTE'], L[400]['THTE'])
    def lr(p1, p2): return (T(p1) - T(p2)) / ((Z(p2) - Z(p1)) / 1000.0)
    if lapse_mode == 'legacy':
        D['LR87'] = (T(850) - T(700)) / 1.6
        D['LR01'] = abs((T(1000) - T(850)) / 1.5)
        D['LR03'] = abs(((T(1000) - T(925)) + (T(850) - T(700))) / 3)   # legacy formula skips 925-850
        D['LR75'] = (T(700) - T(500)) / 2.8
        D['LR53'] = (T(500) - T(300)) / 3.8
    else:
        D['LR87'] = lr(850, 700); D['LR01'] = abs(lr(1000, 850)); D['LR03'] = abs(lr(1000, 700))
        D['LR75'] = lr(700, 500); D['LR53'] = lr(500, 300)
    D['HMI'] = D['LR87'] + ((T(850) - L[850]['DWPT']) - (T(700) - L[700]['DWPT']))
    D['WMSI'] = D['CAPE'] * D['dthe'] / 1000
    D['gust'] = 0.0475 * D['WMSI'] + 35
    D['RH87'] = (L[850]['RELH'] + L[700]['RELH']) / 2
    D['RH75'] = (L[700]['RELH'] + L[500]['RELH']) / 2
    D['RH53'] = (L[500]['RELH'] + L[300]['RELH']) / 2
    D['wspd_015'], D['wdir_015'] = _vec_mean_wind(snd, (1000, 925, 850))
    D['wspd_03'], D['wdir_03'] = _vec_mean_wind(snd, (1000, 925, 850, 700))
    D['wspd_87'], _ = _vec_mean_wind(snd, (850, 700))
    for p in (1000, 925, 850, 700, 250):
        D[f'dir{p}'], D[f'spd{p}'] = snd.wind(p)
    D['spd_shear'] = ((D['spd850'] - D['spd1000']) / 0.5) / 1400 * 1000   # legacy units, thresholds tuned to it
    for p in (1000, 850, 700):
        D[f'MixR{p}'] = L[p]['MIXR']; D[f'Te{p}'] = L[p]['THTE']
    for p in (850, 700, 500, 400):
        D[f'T{p}'] = T(p)
    D['VOTD'] = round(750 / ((abs(T(500)) + abs(T(400))) * 0.5))
    m = D['month'] - 1
    D['sc'] = {k: v[m] for k, v in SC.items()}
    D.update(extra_diag(snd))
    return D

# ---------------------------------------------------------------------------
class Ledger:
    """Every point added or removed from a hazard is recorded with its reason, so the dashboard
    can show *why* a risk is high — replacing the old free-text 'bonus' string."""
    def __init__(self):
        self.items = {h: [] for h in HAZARDS}
        self.remarks = []
    def add(self, h, delta, label, reason, src='rule'):
        if delta and np.isfinite(delta):
            self.items[h].append(dict(delta=float(delta), label=label, reason=reason, src=src))
    def total(self, h):
        return sum(i['delta'] for i in self.items[h])
    def set_to(self, h, target, label, reason, src='rule'):
        self.add(h, target - self.total(h), label, reason, src)
    def remark(self, text, kind='info'):
        self.remarks.append((kind, text))

def _box_score(v, row):
    a, b, c, d = row
    if v < a: return -0.5
    if v < b: return 0.5
    if v <= c: return 1.0
    if v < d: return 0.5
    return -0.5

def assess(D, g, season, gate=True, hc=None, env=None):
    L = Ledger()
    sc = D['sc']
    KI, CAPE, PWAT, T500 = D['KI'], D['CAPE'], D['PWAT'], D['T500']
    LR87, LR01, LR03, RH87, RH75, RH53 = D['LR87'], D['LR01'], D['LR03'], D['RH87'], D['RH75'], D['RH53']
    W03, WD03, W87, W015, WD015 = D['wspd_03'], D['wdir_03'], D['wspd_87'], D['wspd_015'], D['wdir_015']
    dthe, shear = D['dthe'], D['spd_shear']

    # ---- 1. Local boxplot climatology (unchanged logic, now itemised) ----------------
    vals = dict(CAPE=CAPE, PWAT=PWAT, KI=KI, SHI=D['SHI'], SWEAT=D['SWEAT'], TT=D['TT'])
    for name, rows in CLIMO.items():
        for h, row in zip(CLIMO_HZ, rows):
            s = _box_score(vals[name], row)
            L.add(h, s, f'{name} climo', f'{name} = {vals[name]:.1f} vs {HAZARD_LABEL[h]} boxplot {row}', src='climo')

    # ---- 2. Legacy rules (bug-fixed) ------------------------------------------------
    hz3 = ('tornado', 'twind', 'hail')
    low_conv = (((KI < 20 or CAPE < 300) and (LR87 < sc['lr87'] or RH87 < 55))
                or (PWAT < sc['pwat'] - 0.35 and KI < 20)           # FIX: was "(sc_pwat-0.35 and KI<20)", always truthy
                or (LR87 < sc['lr87'] and PWAT < sc['pwat'] and W87 > 22)
                or LR87 < sc['lr87_75'])
    if low_conv:
        if (KI > 20 or CAPE > 1000) and RH87 >= 60:
            for h, d in zip(hz3, (-3, -2, -2)):
                L.add(h, d, 'Stable 850-700', 'Weak instability/stable 850–700 hPa, but moist: isolated storms may still cause localized flooding')
            L.remark('Isolated thunderstorms may lead to localized flooding.')
        else:
            for h, d in zip(hz3, (-5, -4, -4)):
                L.add(h, d, 'Stable/dry', 'Low KI/CAPE with stable or dry 850–700 hPa layer')
            L.remark('Low risk of thunderstorms; any that develop will be isolated, brief and shallow.')
    elif W03 >= 20 and PWAT < sc['pwat'] and RH87 < 60 and T500 >= -7.0:
        for h in ('flood',) + hz3:
            L.add(h, -3, 'Strong trades', f'0–3 km mean wind {W03:.0f} kt with below-normal PWAT limits sea-breeze convection')
    # (the old 500–300 hPa RH 'High cloud' rule is replaced by the HCDC layer below)

    # Hail
    if ((KI < 20 or CAPE < 2000) and T500 > -7) or RH87 < 45 or (LR87 < sc['lr87'] and KI < 28):
        L.add('hail', -3, 'Hail unfavorable', 'Warm 500 hPa, low KI/CAPE, dry or stable 850–700 hPa')
        L.add('twind', -2, 'Hail-env penalty', 'Same unfavorable updraft environment')
        L.add('tornado', -2, 'Hail-env penalty', 'Same unfavorable updraft environment')
    elif T500 <= -8.5 and dthe > 15 and (KI > 25 or CAPE > 1250):
        L.add('hail', 3, 'Cold 500 hPa', f'T500 {T500:.1f} °C ≤ −8.5 with θe deficit {dthe:.0f} K')
    elif T500 <= -7.0 and dthe > 18 and (KI >= 18 or CAPE > 1800):
        L.add('hail', 2, 'Cool 500 hPa', f'T500 {T500:.1f} °C with θe deficit {dthe:.0f} K')
    elif T500 > -7.0 and (KI < 22 or CAPE < 1000 or dthe < 20):
        L.add('hail', -2, 'Normal T500', 'Near-normal 500 hPa temperature for hail')
    if D['thick700_500'] > sc['thk75']:
        L.add('hail', -2, 'High freezing level', f'700–500 hPa thickness {D["thick700_500"]:.0f} m > {sc["thk75"]} m')
    if 90 <= WD03 <= 150 and 230 <= D['dir250'] <= 310 and T500 <= -8.0 and KI > 22:
        L.add('hail', 1, 'Directional shear', 'Easterlies below, westerlies at 250 hPa')


    # Thunderstorm wind
    if W03 > 25 and D['WMSI'] > 35:
        L.add('twind', 5, 'Strong LL wind', f'0–3 km wind {W03:.0f} kt and WMSI {D["WMSI"]:.0f}')
    elif (W03 < 14 and LR01 < 5.25 and LR03 < 5) or (W87 < 20 and LR87 < sc['lr87'] and KI < 22):
        L.add('twind', -3, 'Weak wind/stable', 'Light flow and weak low-level lapse rates')
    elif dthe <= 15 and W87 < 20:
        L.add('twind', -1.5, 'Moist & light', 'Small θe deficit and light 850–700 hPa flow')
    elif W87 > 20 and (RH75 < 65 and RH53 < 65) and KI >= 25 and CAPE >= 1500 and LR03 > 5.5:
        L.add('twind', 1.5, 'Steep LR + flow', 'Steep lapse rates with strong 850–700 hPa flow and dry mid-levels')
    elif KI >= 20 and CAPE >= 1500 and LR03 > 5.5 and W87 > 20 and RH87 > 65:
        L.add('twind', 1.5, 'Strong 0-3 km', 'Strong 0–3 km mean wind')
    elif dthe >= 20 and KI >= 25 and CAPE >= 1500 and LR03 > 5.5 and W87 > 20:
        L.add('twind', 2, 'θe deficit', f'1000–500 hPa θe deficit {dthe:.0f} K (downdraft potential)')

    # Tornado  (FIX: 250 hPa speed, not direction, in the first test)
    if (W03 < 7 and D['spd250'] < 19 and CAPE < 1500) or shear < 6 or LR87 < sc['lr87']:
        L.add('tornado', -3, 'Low shear/stable', 'Weak speed shear or stable 850–700 hPa')
    elif (shear >= 7 and D['dir1000'] > D['dir850'] and 90 <= WD03 <= 140 and 230 <= D['dir250'] <= 310
          and LR03 >= 5.5 and LR01 > 6 and L.total('twind') >= 5):
        L.add('tornado', 3, 'Speed+dir shear', 'Strong speed and directional shear with steep low-level lapse rates')
    elif shear > 8 and L.total('twind') >= 6.5 and LR03 > 5 and D['dir1000'] > D['dir925'] > D['dir850'] > D['dir700']:
        L.add('tornado', 2, 'Speed shear', 'High speed shear with backing profile')
    elif shear < 6 and LR03 < 5.5:
        L.add('tornado', -2.5, 'Weak shear', 'Weak shear and modest lapse rates')

    # Fair-weather waterspout
    if LR01 >= 6 and shear > 15 and D['MixR1000'] > 14:
        if L.total('waterspout') < 5.5:
            L.set_to('waterspout', 7.5 if shear >= 22 else 5.5, 'Steep LR + shear',
                     f'0–1.5 km lapse rate {LR01:.1f} °C/km, shear index {shear:.0f}, MixR1000 {D["MixR1000"]:.1f} g/kg')
    elif (((LR03 < 4.8 or LR01 < 5) and (D['MixR1000'] < 15 and D['MixR850'] < 9))
          or W03 > 18 or (WD03 <= 85 and shear < 5)):
        L.add('waterspout', -4, 'Unfavorable', 'Climatologically unfavorable for waterspouts')
    elif D['T850'] > 17.5 and LR01 >= 6 and D['Te850'] > 345 and 90 <= WD03 <= 220 and abs(shear) > 5:
        L.add('waterspout', 3, 'Szilagyi', 'Szilagyi waterspout nomogram conditions met')
    elif (D['MixR1000'] > 15.8 and 9 < D['MixR850'] < 13 and 30 < WD015 < 180 and LR01 >= 5
          and RH87 > 55 and CAPE > 300):
        L.add('waterspout', 2, 'Moist unstable LL', 'Large low-level mixing ratio and instability')
    elif W015 < 15 and 60 <= WD03 <= 170 and LR03 > 5 and LR01 > LR03 and RH87 > 55:
        L.add('waterspout', 2, 'Light easterlies', 'Light easterly flow with steeper lapse rate near the surface')
    elif CAPE > 2000 and shear > 12 and RH87 > 55 and LR87 < sc['lr87']:
        L.add('waterspout', 2, 'Cb support', 'Deep convection can support waterspouts')

    # Flooding rain
    m = D['month']
    if PWAT >= sc['pwat'] - 0.1 and RH87 >= 65 and KI > 25 and 5 <= m <= 11 and LR01 > 6:
        L.add('flood', 2.5, 'High instability', 'Near/above-normal PWAT, moist 850–700 hPa, KI > 25')
        if L.total('flood') <= 5:
            L.set_to('flood', 5.5, 'Floor 5.5', 'Wet-season instability floor')
    elif PWAT <= sc['pwat'] - 0.2 and (m < 5 or m >= 11):
        if KI < 20 and LR03 < 5:
            L.add('flood', -3, 'Low PWAT', f'PWAT {PWAT:.2f} in below the {sc["pwat"]:.2f} in climatology')
        elif RH87 >= 65 and KI > 25:
            L.add('flood', 1, 'Low PWAT, weak cap', 'Below-normal PWAT but weak inversion')
        else:
            L.add('flood', -1, 'Low PWAT', 'Below-normal PWAT')
    elif (CAPE >= 1500 or KI > 25) and RH87 >= 70:
        if sc['pwat'] < PWAT < sc['pwat'] + 0.2:
            L.add('flood', 2, 'PWAT above normal', f'PWAT {PWAT:.2f} in above climatology')
        elif PWAT >= sc['pwat'] + 0.2 and D['Te850'] > 340 and D['Te700'] > 335:
            L.add('flood', 3, 'PWAT well above', 'PWAT well above normal with high θe at 850/700 hPa')
    elif W03 >= 18 and LR87 < 5.5 and RH87 < 60:
        L.add('flood', -2, 'Strong flow + cap', 'Strong flow and capping inversion')
    elif sc['pwat'] - 0.2 < PWAT <= sc['pwat'] and m < 4:
        L.remark('Normal PWAT compared with climatology.')
    if sc['pwat'] - 0.2 < PWAT <= sc['pwat'] and m >= 4 and KI > 22 and RH87 > 60:
        L.add('flood', 1, 'Near-normal PWAT', 'Near-normal PWAT with KI > 22 and moist 850–700 hPa')
    if PWAT >= sc['pwat'] - 0.25 and W03 < 10 and KI >= 25 and RH87 > 70:
        L.add('flood', 1.5, 'Slow steering', f'0–3 km mean wind {W03:.0f} kt: slow-moving, training cells')
    if W03 > 15 and (D['Te850'] <= 335 or D['Te700'] <= 335):
        L.add('flood', -1, 'Fast & low θe', 'Fast flow with modest θe at 850/700 hPa')

    # San Juan streamer
    if KI < 13:
        L.remark('Based on climatology, a San Juan streamer is not likely.')
    elif ((LR01 > 5.5 or LR87 > sc['lr87']) and RH87 >= 40 and D['MixR1000'] >= 15 and D['MixR850'] >= 7
          and 90 <= WD03 <= 160):
        if LR01 > 5.5 and W015 >= 10:
            L.add('streamer', 1, 'LL lapse rate', 'Steep 0–1.5 km lapse rate with ≥10 kt flow')
        if LR87 >= sc['lr87'] - 0.2 and CAPE > 400 and PWAT >= sc['pwat'] - 0.2:
            L.add('streamer', 4, '850-700 LR', '850–700 hPa lapse rate at/above climatology with moisture')
        if 7 <= D['MixR850'] <= 14 and 5 <= D['MixR700'] <= 7 and CAPE > 1500:
            L.add('streamer', 1.5, 'MixR 850/700', 'Favorable 850/700 hPa mixing ratios')
        if 330 <= D['Te850'] <= 342 and 325 <= D['Te700'] <= 334 and CAPE > 350:
            L.add('streamer', 1.5, 'θe 850/700', 'Favorable 850/700 hPa θe')

    # ---- 2b. Playbook ingredients (moisture depth, persistence, downdraft, hail, lightning, waterspout) ----
    env = env or {}
    def tier(v, edges, pts):
        """Points for the highest edge reached (edges ascending); NaN gives 0."""
        if v is None or not np.isfinite(v): return 0.0
        out = 0.0
        for e_, p_ in zip(edges, pts):
            if v >= e_: out = p_
        return out
    gdi_ = g['GDI']
    # Flooding rain: warm cloud depth (collision–coalescence efficiency), deep-layer RH, back-building (Corfidi)
    wcd = D['WCD']
    if np.isfinite(wcd):
        p_ = 1.0 if wcd >= 4.5 else 0.5 if wcd >= 3.5 else -0.5 if wcd < 3.0 else 0.0
        L.add('flood', p_, f'Warm cloud {wcd:.1f} km', f'Warm cloud depth (LCL→0 °C) {wcd:.1f} km: deeper = more efficient warm-rain processes', src='ingr')
    rh85 = D['RH85']
    if np.isfinite(rh85):
        p_ = 1.0 if rh85 >= 80 else 0.5 if rh85 >= 65 else -1.0 if rh85 < 50 else 0.0
        L.add('flood', p_, f'RH 850–500 {rh85:.0f}%', f'Mean 850–500 hPa RH {rh85:.0f}%', src='ingr')
    cup = D['CORFIDI_UP']
    if np.isfinite(cup) and PWAT >= sc['pwat'] - 0.2 and gdi_ >= 15:
        p_ = 1.5 if cup < 10 else 0.5 if cup < 15 else -0.5 if cup >= 20 else 0.0
        L.add('flood', p_, f'Corfidi up {cup:.0f} kt', f'Corfidi upwind (back-building) vector {cup:.0f} kt: < 10 kt favors back-building and training (Corfidi 2003)', src='ingr')
    # Thunderstorms & lightning: buoyancy in the charge-separation layer and mixed-layer CAPE
    cl = D['CAPE_LTG']
    if np.isfinite(cl):
        p_ = 1.0 if cl >= 300 else 0.5 if cl >= 150 else -0.5 if (cl < 50 and gdi_ < 25) else 0.0
        L.add('tstorm', p_, f'CAPE −10…−30 °C {cl:.0f}', f'CAPE in the −10 to −30 °C (mixed-phase, charging) layer {cl:.0f} J/kg', src='ingr')
    mlc = D['MLCAPE']
    if np.isfinite(mlc):
        L.add('tstorm', 0.5 if mlc >= 2000 else -0.5 if mlc < 300 else 0.0, f'MLCAPE {mlc:.0f}', f'100-hPa mixed-layer CAPE {mlc:.0f} J/kg', src='ingr')
    # Wet microburst: DCAPE + θe deficit (Atkins & Wakimoto 1991; Playbook), Hybrid Microburst Index (Pryor)
    dc = D['DCAPE']
    if np.isfinite(dc):
        L.add('twind', tier(dc, [500, 800, 1100], [0.5, 1.0, 1.5]), f'DCAPE {dc:.0f}', f'Downdraft CAPE {dc:.0f} J/kg from {D.get("DCAPE_P", np.nan):.0f} hPa', src='ingr')
        if dc >= 900 and dthe >= 20:
            L.add('twind', 1.0, 'Wet-microburst signal', f'DCAPE {dc:.0f} J/kg with θe deficit {dthe:.0f} K (≥ 900 and ≥ 20 K)', src='ingr')
    if np.isfinite(D.get('HMI', np.nan)) and D['HMI'] >= 8:
        L.add('twind', 0.5, f'HMI {D["HMI"]:.0f}', f'Hybrid Microburst Index {D["HMI"]:.1f} ≥ 8', src='ingr')
    # Hail: 700–500 hPa lapse rate, 0–6 km shear, wet-bulb-zero height, MLCAPE, TUTT
    L.add('hail', tier(D['LR75'], [5.5, 6.2, 6.8], [0.5, 1.0, 1.5]), f'LR 700–500 {D["LR75"]:.1f}', f'700–500 hPa lapse rate {D["LR75"]:.1f} °C/km', src='ingr')
    sh = D['SHEAR06']
    if np.isfinite(sh):
        L.add('hail', tier(sh, [15, 25, 35], [0.5, 1.0, 1.5]), f'Shear 0–6 km {sh:.0f} kt', f'0–6 km bulk shear {sh:.0f} kt', src='ingr')
    wbz = D['WBZ_Z']
    if np.isfinite(wbz):
        p_ = 1.0 if wbz <= 3400 else -0.5 if wbz >= 4300 else 0.0
        L.add('hail', p_, f'WBZ {wbz/1000:.1f} km', f'Wet-bulb-zero height {wbz/1000:.1f} km (≤ 3.4 km helps hail reach the ground; ≥ 4.3 km melts it)', src='ingr')
    if np.isfinite(mlc) and mlc >= 2500:
        L.add('hail', 0.5, f'MLCAPE {mlc:.0f}', 'Strong mixed-layer buoyancy for hail growth', src='ingr')
    if np.isfinite(env.get('vort200', np.nan)) and env['vort200'] >= TUTT_VORT and T500 <= -7.5:
        for h_ in ('hail', 'tstorm'):
            L.add(h_, 0.5, 'TUTT overhead', f'200 hPa cyclonic vorticity {env["vort200"]*1e5:.1f}×10⁻⁵ s⁻¹ with T500 {T500:.1f} °C', src='lift')
        L.remark('TUTT / upper trough overhead with cool 500 hPa: hail and frequent lightning possible.', 'info')
    # Waterspouts: light surface wind and low-level (0–3 km) CAPE along convergence lines
    sw = env.get('sfc_wind', np.nan); sw_src = 'GFS 10 m'
    if not np.isfinite(sw):
        sw, sw_src = D['SFC_WIND'], 'sounding sfc'
    if np.isfinite(sw):
        p_ = 1.5 if sw < 6 else 1.0 if sw < 10 else 0.5 if sw < 15 else -1.0 if sw > 18 else 0.0
        L.add('waterspout', p_, f'Sfc wind {sw:.0f} kt', f'{sw_src} wind {sw:.0f} kt (8 a.m.–2 p.m.)', src='ingr')
    c3 = D['CAPE03']
    if np.isfinite(c3):
        L.add('waterspout', tier(c3, [50, 100, 150], [0.5, 1.0, 1.5]), f'0–3 km CAPE {c3:.0f}', f'0–3 km CAPE {c3:.0f} J/kg (surface parcel)', src='ingr')

    # ---- 3. GDI science layer ---------------------------------------------------------
    gdi = g['GDI']; reg = gdi_regime(gdi)
    w = R2_TSTORM[season]
    subs = dict(GDI=subscore('GDI', gdi), K=subscore('K', KI), TT=subscore('TT', D['TT']),
                CAPE=subscore('CAPE', max(CAPE, 0)))
    wsum = sum(w.values())
    for k in ('GDI', 'K', 'TT', 'CAPE'):
        L.add('tstorm', subs[k] * w[k] / wsum, f'{k} ({100*w[k]/wsum:.0f}%)',
              f'{k} sub-score {subs[k]:.1f}/10 × weight {w[k]:.2f} (daily R² vs 3-km echo tops, {season}; Miller et al. 2019)',
              src='gdi' if k == 'GDI' else 'index')
    # Saharan-air / dry-intrusion signature: strong inversion term plus a dry 850-700 hPa layer
    if g['II'] <= -25 and RH87 < 50:
        for h, d in (('tstorm', -1.5), ('flood', -1.5)):
            L.add(h, d, 'SAL / dry air', f'II {g["II"]:.0f} with 850–700 hPa RH {RH87:.0f}%: inversion + dry-air entrainment (Mote et al. 2017)', src='gdi')
        L.remark('Inversion and dry 850–700 hPa layer consistent with Saharan air or a dry intrusion.', 'warn')
    # Flooding rain: GDI regime, kept modest (daily GDI explains only 6–9 % of island rain)
    fl = 1.5 if gdi >= 45 else 1.0 if gdi >= 35 else 0.5 if gdi >= 25 else -1.5 if gdi < 5 else 0
    L.add('flood', fl, 'GDI regime', f'GDI {gdi:.0f}: {reg["text"]} (Table 4)', src='gdi')
    # High cloud (HCDC): dense high cloud during 8 a.m.–2 p.m. AST limits heating and convection
    hcp = (hc or {}).get('pct', np.nan)
    if USE_HIGH_CLOUD and np.isfinite(hcp):
        tier = next(((d, name) for thr, d, name in HC_TIERS if hcp >= thr), None)
        if tier:
            d, name = tier
            why = (f"{name} {hcp:.0f}% ({hc['src']}; GFS {hc['gfs']:.0f}%" if np.isfinite(hc.get('gfs', np.nan)) else f"{name} {hcp:.0f}% ({hc['src']}") \
                  + (f", sounding est. {hc['snd']:.0f}% from RH 500–300 {hc['rh53']:.0f}%)" if np.isfinite(hc.get('snd', np.nan)) else ')') \
                  + ', 8 a.m.–2 p.m. AST: limits daytime heating and convection'
            for h in HC_HAZARDS:
                L.add(h, d, f'High cloud {hcp:.0f}%', why, src='cloud')
            L.remark(f"{name} {hcp:.0f}% (8 a.m.–2 p.m. AST) limits heating: {10*d:+d}% to convective hazards.".replace('-', '−'), 'cloud')
    # Synoptic lift (GFS 200 hPa divergence / 700 hPa vorticity): Playbook multipliers on convective hazards
    if env.get('lift'):
        lift, mult = env['lift'], env['mult']
        why = (f"GFS 8 a.m.–2 p.m.: 200 hPa div {env['div200']*1e5:+.1f}×10⁻⁵ s⁻¹, 700 hPa vort {env['vort700']*1e5:+.1f}×10⁻⁵ s⁻¹").replace('-', '−')
        if mult < 1.0:
            for h_ in LIFT_HAZARDS:
                tot = L.total(h_)
                if tot > 0:
                    L.add(h_, (mult - 1) * tot, f'Lift {lift} ×{mult:.2f}', f'Synoptic lift {lift} ({why})', src='lift')
        elif lift == 'strong':
            for h_ in ('tstorm', 'flood'):
                L.add(h_, 0.5, 'Upper support', f'Strong synoptic lift ({why})', src='lift')
    # Severe hazards need storms: Table 4 says activity below GDI 5 is light showers
    if gdi < 5:
        for h in hz3:
            L.add(h, -1, 'GDI < 5', 'Strong trade-wind inversion likely: deep convection suppressed (Table 4)', src='gdi')
    if gate:
        cap = L.total('tstorm') + 1.5
        for h in hz3:
            if L.total(h) > cap:
                L.set_to(h, cap, 'Storm gate', f'Capped at thunderstorm potential + 1.5 ({cap:.1f}): severe weather requires a storm', src='gdi')
    if season == 'ERS' and g['MWI'] <= -20 and g['CBI'] >= 50:
        L.remark('ERS: strongly negative MWI with high CBI. Miller et al. found warm 500 hPa days in the ERS '
                 'coincide with high low-level heat and moisture and more echo tops — don\'t over-weight mid-level warmth.', 'info')

    # ---- 4. Final ---------------------------------------------------------------------
    conf = confidence(season)
    res = {}
    for h in HAZARDS:
        raw = L.total(h)
        score = float(np.clip(raw, 0, 10))
        ri_exact = 10 * score
        ri = round10(ri_exact)                       # displayed and categorized at the nearest 10 %
        cat, color = risk_cat(ri)
        top = sorted(L.items[h], key=lambda i: -abs(i['delta']))[:3]
        cloud = sum(i['delta'] for i in L.items[h] if i['src'] == 'cloud')
        ri_nocloud = round10(10 * float(np.clip(raw - cloud, 0, 10)))
        res[h] = dict(score=score, raw=raw, ri=ri, ri_exact=ri_exact, cat=cat, color=color,
                      items=L.items[h], top=top, conf=conf[h],
                      cloud=cloud, cloud_cut=max(ri_nocloud - ri, 0.0))   # % removed by high cloud (rounded)
    res['_remarks'] = L.remarks
    res['_hc'] = hc or {}

    return res

def confidence(season):
    ts = {'DS': ('Medium', 'DS rain is often from fronts, troughs and orographic trade showers the GDI does not capture (Scholl et al. 2009).'),
          'ERS': ('High', 'ERS is where the GDI performs best in Puerto Rico (seasonal R² 0.65 echo tops, 0.82 rain).'),
          'LRS': ('Medium', 'Daily skill holds (R² 0.27) but tropical waves and cyclones can exceed GDI expectations.')}[season]
    rain = ('Low–Medium', f'Daily GDI explains only {100*R2_DAILY_RAIN[season]:.0f}% of island-mean rain; driven mostly by PWAT and steering flow.')
    emp = ('Empirical', 'Local TJSJ boxplot climatology and forecaster rules.')
    return dict(tstorm=ts, flood=rain, hail=emp, twind=emp, tornado=emp, waterspout=emp, streamer=emp)

# ---------------------------------------------------------------------------
# Legacy Risk Assessment (original TJSJ script): boxplot climatology + forecaster rules only,
# legacy lapse-rate formulas and UWyo CAPE − CAPE_OFFSET. Four bugs fixed:
#  1) "(sc_pwat − 0.35 and KI < 20)" was always true  -> PWAT < sc_pwat − 0.35 and KI < 20
#  2) tornado test used the 250 hPa wind direction as a speed -> 250 hPa speed
#  3) wind directions were averaged arithmetically (350° & 10° -> 180°) -> vector mean
#  4) 0–3 km lapse rate skipped the 925–850 hPa layer -> |T1000 − T700| / 3
# ---------------------------------------------------------------------------
LEGACY_HZ = ['flood', 'twind', 'hail', 'waterspout', 'tornado', 'streamer']     # skill order (no T-storm bar in legacy)
LEGACY_LABEL = {'flood': 'Flooding\nrain', 'hail': 'Hail', 'twind': 'T-storm\nwind', 'tornado': 'Tornado',
                'waterspout': 'Fair-wx\nwaterspout', 'streamer': 'SJ\nstreamer'}

def legacy_color(score):
    """Legacy bar colors: >5 red, >=4 yellow, >2 green, otherwise gray."""
    return '#FF0000' if score > 5 else '#EAEA00' if score >= 4 else '#33CC00' if score > 2 else '#999999'

def legacy_assess(snd):
    D = diagnostics(snd, 'legacy', CAPE_OFFSET)
    D['LR03'] = abs(snd.level(1000)['TEMP'] - snd.level(700)['TEMP']) / 3            # fix 4
    sc = D['sc']; m = D['month']
    KI, CAPE, PWAT, T500 = D['KI'], D['CAPE'], D['PWAT'], D['T500']
    LR87, LR01, LR03, RH87, RH75, RH53 = D['LR87'], D['LR01'], D['LR03'], D['RH87'], D['RH75'], D['RH53']
    W03, WD03, W87, W015, WD015 = D['wspd_03'], D['wdir_03'], D['wspd_87'], D['wspd_015'], D['wdir_015']
    dthe, shear = D['dthe'], D['spd_shear']
    S = {h: 0.0 for h in LEGACY_HZ}; rem = []
    vals = dict(CAPE=CAPE, PWAT=PWAT, KI=KI, SHI=D['SHI'], SWEAT=D['SWEAT'], TT=D['TT'])
    for name, rows in CLIMO.items():
        for h, row in zip(CLIMO_HZ, rows):
            S[h] += _box_score(vals[name], row)
    # ALL: diminish hail, T-wind, tornado
    if (((KI < 20 or CAPE < 300) and (LR87 < sc['lr87'] or RH87 < 55)) or (PWAT < sc['pwat'] - 0.35 and KI < 20)
            or (LR87 < sc['lr87'] and PWAT < sc['pwat'] and W87 > 22) or LR87 < sc['lr87_75']):
        if (KI > 20 or CAPE > 1000) and RH87 >= 60:
            S['tornado'] -= 3; S['twind'] -= 2; S['hail'] -= 2
            rem.append('Isolated thunderstorms may lead to localized flooding')
        else:
            S['tornado'] -= 5; S['twind'] -= 4; S['hail'] -= 4
            rem.append('Low risk of thunderstorms; any will be isolated, brief and shallow')
    elif W03 >= 20 and PWAT < sc['pwat'] and RH87 < 60 and T500 >= -7.0:
        for h in ('flood', 'tornado', 'twind', 'hail'): S[h] -= 3
        rem.append('−30% (tornado, hail, T-wind): strong 0–3 km winds limit sea-breeze formation')
    elif RH53 >= 70 and (KI < 18 or CAPE < 1500) and PWAT < sc['pwat']:
        for h in ('tornado', 'hail', 'twind'): S[h] -= 3
        rem.append('−30% (tornado, hail, T-wind): dense high cloud cover')
    # Hail
    if ((KI < 20 or CAPE < 2000) and T500 > -7) or RH87 < 45 or (LR87 < sc['lr87'] and KI < 28):
        S['hail'] -= 3; S['twind'] -= 2; S['tornado'] -= 2
        rem.append('−30% (hail): warm 500 hPa, low KI/CAPE, or dry/stable 850–700 hPa')
    elif T500 <= -8.5 and dthe > 15 and (KI > 25 or CAPE > 1250):
        S['hail'] += 3; rem.append('+30% (hail): very cold 500 hPa (≤ −8.5 °C)')
    elif T500 <= -7.0 and dthe > 18 and (KI >= 18 or CAPE > 1800):
        S['hail'] += 2; rem.append('+20% (hail): cold 500 hPa')
    elif T500 > -7.0 and (KI < 22 or CAPE < 1000 or dthe < 20):
        S['hail'] -= 2; rem.append('−15% (hail): near-normal 500 hPa temperature')
    if D['thick700_500'] > sc['thk75']:
        S['hail'] -= 2; rem.append('−20% (hail): high freezing level')
    if 90 <= WD03 <= 150 and 230 <= D['dir250'] <= 310 and T500 <= -8.0 and KI > 22:
        S['hail'] += 1; rem.append('+10% (hail): favorable directional shear')
    # Thunderstorm wind
    if W03 > 25 and D['WMSI'] > 35:
        S['twind'] += 5; rem.append('+30% (T-wind): strong low-level winds')
    elif (W03 < 14 and LR01 < 5.25 and LR03 < 5) or (W87 < 20 and LR87 < sc['lr87'] and KI < 22):
        S['twind'] -= 3; rem.append('−30% (T-wind): stable 850–700 hPa / light flow')
    elif dthe <= 15 and W87 < 20:
        S['twind'] -= 1.5; rem.append('−15% (T-wind): moist environment, light winds')
    elif W87 > 20 and RH75 < 65 and RH53 < 65 and KI >= 25 and CAPE >= 1500 and LR03 > 5.5:
        S['twind'] += 1.5; rem.append('+15% (T-wind): steep lapse rates, strong 850–700 hPa flow')
    elif KI >= 20 and CAPE >= 1500 and LR03 > 5.5 and W87 > 20 and RH87 > 65:
        S['twind'] += 1.5; rem.append('+15% (T-wind): strong 0–3 km mean wind')
    elif dthe >= 20 and KI >= 25 and CAPE >= 1500 and LR03 > 5.5 and W87 > 20:
        S['twind'] += 2; rem.append('+20% (T-wind): large θe deficit')
    # Tornado (fix 2)
    if (W03 < 7 and D['spd250'] < 19 and CAPE < 1500) or shear < 6 or LR87 < sc['lr87']:
        S['tornado'] -= 3
    elif (shear >= 7 and D['dir1000'] > D['dir850'] and 90 <= WD03 <= 140 and 230 <= D['dir250'] <= 310
          and LR03 >= 5.5 and LR01 > 6 and S['twind'] >= 5):
        S['tornado'] += 3; rem.append('+30% (tornado): speed and directional shear')
    elif shear > 8 and S['twind'] >= 6.5 and LR03 > 5 and D['dir1000'] > D['dir925'] > D['dir850'] > D['dir700']:
        S['tornado'] += 2; rem.append('+20% (tornado): high speed shear')
    elif shear < 6 and LR03 < 5.5:
        S['tornado'] -= 2.5
    # Fair-weather waterspout
    if LR01 >= 6 and shear > 15 and D['MixR1000'] > 14:
        if S['waterspout'] < 5.5:
            S['waterspout'] = 7.5 if shear >= 22 else 5.5
    elif ((LR03 < 4.8 or LR01 < 5) and D['MixR1000'] < 15 and D['MixR850'] < 9) or W03 > 18 or (WD03 <= 85 and shear < 5):
        S['waterspout'] -= 4; rem.append('−40% (waterspout): climatologically unfavorable')
    elif D['T850'] > 17.5 and LR01 >= 6 and D['Te850'] > 345 and 90 <= WD03 <= 220 and abs(shear) > 5:
        S['waterspout'] += 3; rem.append('+30% (waterspout): Szilagyi technique')
    elif D['MixR1000'] > 15.8 and 9 < D['MixR850'] < 13 and 30 < WD015 < 180 and LR01 >= 5 and RH87 > 55 and CAPE > 300:
        S['waterspout'] += 2; rem.append('+20% (waterspout): large low-level mixing ratio and instability')
    elif W015 < 15 and 60 <= WD03 <= 170 and LR03 > 5 and LR01 > LR03 and RH87 > 55:
        S['waterspout'] += 2; rem.append('+15% (waterspout): light easterly flow')
    elif CAPE > 2000 and shear > 12 and RH87 > 55 and LR87 < sc['lr87']:
        S['waterspout'] += 2; rem.append('+20% (waterspout): Cb/TSRA support')
    # Flooding rain
    if PWAT >= sc['pwat'] - 0.1 and RH87 >= 65 and KI > 25 and 5 <= m <= 11 and LR01 > 6:
        S['flood'] += 2.5; rem.append('+25% (flood): high convective instability')
        if S['flood'] <= 5: S['flood'] = 5.5
    elif PWAT <= sc['pwat'] - 0.2 and (m < 5 or m >= 11):
        if KI < 20 and LR03 < 5:
            S['flood'] -= 3; rem.append('−30% (flood): below-normal PWAT')
        elif RH87 >= 65 and KI > 25:
            S['flood'] += 1; rem.append('+10% (flood): below-normal PWAT but weak inversion')
        else:
            S['flood'] -= 1; rem.append('−10% (flood): below-normal PWAT')
    elif (CAPE >= 1500 or KI > 25) and RH87 >= 70:
        if sc['pwat'] < PWAT < sc['pwat'] + 0.2:
            S['flood'] += 2; rem.append('+20% (flood): above-normal PWAT')
        elif PWAT >= sc['pwat'] + 0.2 and D['Te850'] > 340 and D['Te700'] > 335:
            S['flood'] += 3; rem.append('+30% (flood): PWAT well above normal')
    elif W03 >= 18 and LR87 < 5.5 and RH87 < 60:
        S['flood'] -= 2; rem.append('−20% (flood): strong flow and capping inversion')
    if PWAT >= sc['pwat'] - 0.25 and W03 < 10 and KI >= 25 and RH87 > 70:
        S['flood'] += 1.5; rem.append('+15% (flood): light steering flow')
    if W03 > 15 and (D['Te850'] <= 335 or D['Te700'] <= 335):
        S['flood'] -= 1
    # San Juan streamer
    if KI >= 13 and ((LR01 > 5.5 or LR87 > sc['lr87']) and RH87 >= 40 and D['MixR1000'] >= 15
                     and D['MixR850'] >= 7 and 90 <= WD03 <= 160):
        if LR01 > 5.5 and W015 >= 10: S['streamer'] += 1
        if LR87 >= sc['lr87'] - 0.2 and CAPE > 400 and PWAT >= sc['pwat'] - 0.2: S['streamer'] += 4
        if 7 <= D['MixR850'] <= 14 and 5 <= D['MixR700'] <= 7 and CAPE > 1500: S['streamer'] += 1.5
        if 330 <= D['Te850'] <= 342 and 325 <= D['Te700'] <= 334 and CAPE > 350: S['streamer'] += 1.5
    S = {h: round10(10 * max(v, 0.0)) / 10 for h, v in S.items()}       # nearest 10 %
    return dict(scores=S, remarks=rem, gust=D['gust'], D=D)

def run_engine(snd, hc=None, env=None):
    season = season_of(snd.meta['valid'].month)
    D = diagnostics(snd, LAPSE_RATE_MODE, CAPE_OFFSET)
    g = gdi_from_sounding(snd, APPLY_TERRAIN_CORRECTION)
    res = assess(D, g, season, GDI_GATE_SEVERE, hc, env)
    return dict(snd=snd, D=D, g=g, res=res, season=season, hc=hc or {}, env=env or {}, legacy=legacy_assess(snd))

# Overview graphic (1600×1000, dark)
import os, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

BG, PANEL, INK_, INK2_, MUTED_, LINE_ = '#1a1a19', '#242422', '#f0efec', '#c3c2b7', '#8f8e87', '#3a3a37'
BLUE, RED, GREEN, AMBER = '#3987e5', '#d03b3b', '#0ca30c', '#fab219'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK_, 'axes.labelcolor': INK2_,
                     'xtick.color': INK2_, 'ytick.color': INK2_, 'axes.edgecolor': LINE_})

PUBLIC_HAZARD = {'flood': 'Flooding rain', 'tstorm': 'Thunderstorms & lightning','hail': 'Hail',
                 'twind': 'Thunderstorm winds', 'waterspout': 'Waterspouts', 'tornado': 'Tornado',
                 'streamer': 'San Juan streamer showers'}
SHORT_HAZARD = {'flood': 'Flooding rain', 'tstorm': 'T-storms-Lightning', 'hail': 'Hail', 'twind': 'T-storm winds',
                 'waterspout': 'Waterspouts', 'tornado': 'Tornado', 'streamer': 'SJ streamer'}
PUBLIC_REGIME = [  # Gálvez & Davison Table 4, in plain words
    (-np.inf, 5, 'Mostly stable', 'Only brief, light showers expected.'),
    (5, 15,      'Passing showers', "Showers at times; a brief thunderstorm can't be ruled out."),
    (15, 25,     'Isolated T-storms', 'A few thunderstorms possible, mainly in the afternoon.'),
    (25, 35,     'Isolated to scattered T-storms', 'Thunderstorms likely in spots; locally heavy rain possible.'),
    (35, 45,     'Scattered T-storms', 'Scattered thunderstorms; locally heavy rain possible.'),
    (45, np.inf, 'Widespread T-storms', 'Numerous thunderstorms; heavy rain likely in places.'),
]
def public_regime(v):
    for (lo, hi, name, text), reg in zip(PUBLIC_REGIME, GDI_REGIMES):
        if lo <= v < hi:
            return name, text, reg[4]

def _ast(t_utc):
    return t_utc - dt.timedelta(hours=4)        # Atlantic Standard Time, no daylight saving

def _num(v):
    s_ = f'{v:+.1f}' if abs(v) < 1 else f'{v:+.0f}'
    return s_.replace('-', '−')

def _m(v, fmt='.1f'):
    return format(v, fmt).replace('-', '−')

def _ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

def _compass(deg):
    return ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int((deg % 360 + 22.5) // 45) % 8]

def _panel(fig, rect, title=None, right=None, fc=PANEL, ec='none', lw=0):
    ax = fig.add_axes(rect); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.002, 0.004), 0.996, 0.992, boxstyle='round,pad=0,rounding_size=0.02', fc=fc, ec=ec, lw=lw,
                                mutation_aspect=rect[2] / rect[3] * 1.6))
    if title:
        fig.text(rect[0] + 0.013, rect[1] + rect[3] - 0.03, title.upper(), fontsize=10.5, color=MUTED_, weight='bold')
    if right:
        fig.text(rect[0] + rect[2] - 0.013, rect[1] + rect[3] - 0.03, right, fontsize=10.5, color=MUTED_, ha='right')
    return ax

plt.rcParams['hatch.linewidth'] = 1.2

def _risk_bar(fig, rect, rr, outline=False):
    """Risk bar: gray track, colored risk, and a hatched light-blue segment for the part removed by high cloud.
    outline=True (lower forecast skill): the risk is drawn as a hollow bar with a faint fill."""
    bx = fig.add_axes(rect); bx.axis('off'); bx.set_xlim(0, 100); bx.set_ylim(0, 1)
    bx.barh(0.5, 100, height=1, color=LINE_, lw=0)
    if outline:
        bx.barh(0.5, max(rr['ri'], 1), height=0.86, color=rr['color'], alpha=0.22, lw=0)
        bx.barh(0.5, max(rr['ri'], 1), height=0.86, facecolor='none', edgecolor=rr['color'], lw=1.3)
    else:
        bx.barh(0.5, max(rr['ri'], 1), height=1, color=rr['color'], lw=0)
    cut = rr.get('cloud_cut', 0)
    if cut >= 1:
        bx.barh(0.5, cut, left=rr['ri'], height=1, color='none', edgecolor=HC_COLOR, hatch='////', lw=0)
    return bx

def _hc_txt(hc):
    p = (hc or {}).get('pct', np.nan)
    return f"☁ {p:.0f}%" if np.isfinite(p) else '☁ n/a'

def _hc_col(hc):
    p = (hc or {}).get('pct', np.nan)
    return HC_COLOR if np.isfinite(p) and p >= HC_TIERS[-1][0] else MUTED_

def _drivers(r, n=2):
    """Top contributors to a hazard score, in the engine's own (technical) labels."""
    items = [i for i in r['items'] if abs(i['delta']) >= 0.5 and i['src'] != 'climo']
    if not items:
        items = [i for i in r['items'] if abs(i['delta']) >= 0.5]
    items = sorted(items, key=lambda i: -abs(i['delta']))[:n]
    return ' · '.join(f"{i['label']} {_num(i['delta'])}" for i in items)

def _tech_notes(R, inv, runs=None, gfs_note=None):
    """Technical bulletin lines: warnings and remarks first, then GDI anatomy, inversion and confidence."""
    g, D, res, s = R['g'], R['D'], R['res'], R['season']
    pct, z = pr_gdi_percentile(g['GDI'])
    warn = [('⚠', t) for k, t in res['_remarks'] if k == 'warn']
    cloud = [('☁', t) for k, t in res['_remarks'] if k == 'cloud']
    if runs:
        parts = [f"{_ast(r['snd'].meta['valid']):%a} {r['hc']['pct']:.0f}% ({10*r['res']['tstorm']['cloud']:+.0f}%)".replace('-', '−')
                 for r in runs if any(k == 'cloud' for k, _ in r['res']['_remarks'])]
        if parts:
            cloud.append(('☁', 'High cloud at 2 p.m.: ' + ' · '.join(parts) + ' to convective hazards.'))
    info = [('•', t) for k, t in res['_remarks'] if k not in ('warn', 'cloud')]
    lines = ([('ℹ', gfs_note)] if gfs_note else []) + warn + cloud + info
    X = D; env = R.get('env', {})
    sw = env.get('sfc_wind', np.nan); sw = sw if np.isfinite(sw) else X['SFC_WIND']
    lines.append(('•', (f"Ingredients: WCD {X['WCD']:.1f} km · RH 850–500 {X['RH85']:.0f}% · Corfidi up {X['CORFIDI_UP']:.0f} kt · "
                        f"MLCAPE {X['MLCAPE']:.0f} · CAPE −10/−30 °C {X['CAPE_LTG']:.0f} · CAPE 0–3 km {X['CAPE03']:.0f} · DCAPE {X['DCAPE']:.0f} · "
                        f"shear 0–6 km {X['SHEAR06']:.0f} kt · WBZ {X['WBZ_Z']/1000:.1f} km · sfc wind {sw:.0f} kt").replace('nan', 'n/a')))
    if env.get('lift'):
        lines.append(('•', (f"Lift (GFS 8 a.m.–2 p.m.): {env['lift']} ×{env['mult']:.2f} — 200 hPa divergence {env['div200']*1e5:+.1f}, "
                            f"700 hPa vorticity {env['vort700']*1e5:+.1f}, 200 hPa vorticity {env['vort200']*1e5:+.1f} (×10⁻⁵ s⁻¹).").replace('-', '−')))
    lines.append(('•', f"GDI {_m(g['GDI'])} = CBI {_num(g['CBI'])} + MWI {_num(g['MWI'])} + II {_num(g['II'])} + TC {_num(g['TC'])}; "
                       f"{_ordinal(round(pct))} pct of TJSJ 2002–16 (z {_num(z)}). Table 4: {gdi_regime(g['GDI'])['text']}"))
    if inv:
        extra = f", ΔRH {inv['dRH']:+.0f}%" if np.isfinite(inv.get('dRH', np.nan)) else ''
        lines.append(('•', f"Stable layer {inv['p_base']:.0f}–{inv['p_top']:.0f} hPa ({inv['z_base']/1000:.1f}–{inv['z_top']/1000:.1f} km), "
                           f"lapse {_m(inv['lapse'])} °C/km{extra.replace('-', '−')}; S {_m(g['S'])} K, D {_m(g['D'])} K."))
    if runs:
        lines.append(('•', f"Outlook: observed profile + GFS change from {R['snd'].meta['valid']:%d/%HZ} to each 18Z (ΔT, Δr, Δwind, ΔZ in ln p; "
                           f"r capped at saturation); indices = UWyo value + GFS index change."))
    conf, why = res['tstorm']['conf']
    lines.append(('•', f"Confidence ({s}): {conf}. {why}"))

    return lines

def _today_legacy(fig, R, pm, rect):
    """TODAY card drawn as a legacy-style vertical bar chart from the new engine: for each hazard a solid bar
    for Now (sounding) and, when available, a second bar for 2 p.m. (sounding + GFS). New-threshold colors,
    % labels on top, hatched light-blue caps for what high cloud removed."""
    xa, ya, wa, ha = rect
    g, res = R['g'], R['res']; t_obs = R['snd'].meta['valid']
    gname, gtext, gcol = public_regime(g['GDI'])
    _panel(fig, rect, fc='#2b2b28', ec=gcol, lw=2.2)
    top = ya + ha
    fig.text(xa + 0.013, top - 0.032, ' TODAY ', fontsize=11, weight='bold', color='#111' if gcol in ('#ffd60a', '#52b788', '#48cae4') else '#fff',
             bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.6', fc=gcol, ec='none'), va='center')
    fig.text(xa + 0.075, top - 0.032, 'HAZARDS TO WATCH', fontsize=10.5, color=MUTED_, weight='bold', va='center')
    gtxt = f"GDI {_num(g['GDI'])}" + (f"  →  {_num(pm['g']['GDI'])} at 2 p.m." if pm else '')
    fig.text(xa + wa - 0.013, top - 0.032, gtxt, fontsize=12, weight='bold', color=INK_, ha='right', va='center')
    sub = f"Now: {gname}" + (f"  ·  2 p.m.: {public_regime(pm['g']['GDI'])[0]}" if pm else f"  ·  {gtext}")
    fig.text(xa + 0.013, top - 0.062, sub, fontsize=11, color=INK2_, va='center')
    hc_now = R.get('hc', {})
    if USE_HIGH_CLOUD:
        htxt = 'High cloud ' + _hc_txt(hc_now).replace('☁ ', '') + (f"  →  {_hc_txt(pm['hc']).replace('☁ ', '')} for 2 p.m." if pm else '')
        fig.text(xa + wa - 0.013, top - 0.062, '☁ ' + htxt, fontsize=11, weight='bold',
                 color=HC_COLOR if (_hc_col(hc_now) == HC_COLOR or (pm and _hc_col(pm['hc']) == HC_COLOR)) else MUTED_,
                 ha='right', va='center')
    key = {h: max([res[h]['ri']] + ([pm['res'][h]['ri']] if pm else [])) for h in HAZARDS}
    order = list(SKILL_ORDER)
    ax = fig.add_axes([xa + 0.02, ya + 0.085, wa - 0.04, ha - 0.175]); ax.set_facecolor('#2b2b28')
    bw = 0.36 if pm else 0.6
    cols = [(res, 0.0, 'Now')] + ([(pm['res'], bw + 0.04, '2 p.m.')] if pm else [])
    off0 = -(bw + 0.04) / 2 if pm else 0.0
    for i, h in enumerate(order):
        for rr_all, dx, lab in cols:
            rr = rr_all[h]; x = i + off0 + dx
            if low_skill(h):
                ax.bar(x, max(rr['ri'], 0.8), width=bw, color=rr['color'], alpha=0.2, lw=0)
                ax.bar(x, max(rr['ri'], 0.8), width=bw, facecolor='none', edgecolor=rr['color'], lw=1.6,
                       ls='-' if lab == 'Now' else (0, (3, 2)))
            else:
                ax.bar(x, max(rr['ri'], 0.8), width=bw, color=rr['color'], alpha=1.0 if lab == 'Now' else 0.6, lw=0)
            cut = rr.get('cloud_cut', 0)
            if cut >= 1:
                ax.bar(x, cut, bottom=rr['ri'], width=bw, color='none', edgecolor=HC_COLOR, hatch='////', lw=0)
            ax.text(x, rr['ri'] + max(cut, 0) + 1.5, f"{rr['ri']:.0f}%", ha='center', va='bottom', fontsize=10 if pm else 11,
                    weight='bold', color=HC_COLOR if cut >= 1 else INK_)
    for lo, hi, nm, c in RISK_CATS[1:]:
        ax.axhline(lo, color=LINE_, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(-1.25, lo + 1, nm, fontsize=8.5, color=MUTED_, ha='left', va='bottom')
    ax.set_ylim(0, 112); ax.set_xlim(-1.3, len(order) - 0.4); ax.set_yticks([])
    TL = {'flood': 'Flooding\nrain', 'tstorm': 'T-storms &\nlightning', 'twind': 'T-storm\nwinds', 'hail': 'Hail',
          'waterspout': 'Waterspouts', 'tornado': 'Tornado', 'streamer': 'SJ\nstreamer'}
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{SKILL_DOTS[SKILL_TIER[h]]}\n{TL[h]}" for h in order], fontsize=10, color=INK_)
    for t_, h in zip(ax.get_xticklabels(), order):
        if low_skill(h): t_.set_color(INK2_)
    k1 = next(i for i, h in enumerate(order) if low_skill(h))
    ax.axvline(k1 - 0.5, color=MUTED_, lw=1, ls=(0, (2, 3)))
    ax.text(k1 - 0.42, 108, 'LOWER FORECAST SKILL', fontsize=8.5, color=MUTED_, weight='bold', va='top')
    fig.text(xa + 0.013, top - 0.092, SKILL_KEY, fontsize=9, color=MUTED_)
    ax.tick_params(axis='x', length=0, pad=4)
    for s_ in ('top', 'right', 'left'): ax.spines[s_].set_visible(False)
    ax.spines['bottom'].set_color(LINE_)
    # legend: Now / 2 p.m. / high cloud
    lx = xa + 0.015; ly = ya + 0.016
    hr = _ast(t_obs); now_lbl = f"Now · {hr.hour % 12 or 12} {'a.m.' if hr.hour < 12 else 'p.m.'} sounding"
    fig.add_artist(Rectangle((lx, ly), 0.012, 0.012, color=INK2_, lw=0, transform=fig.transFigure))
    fig.text(lx + 0.016, ly + 0.006, now_lbl, fontsize=9, color=INK2_, va='center')
    if pm:
        fig.add_artist(Rectangle((lx + 0.15, ly), 0.012, 0.012, color=INK2_, alpha=0.6, lw=0, transform=fig.transFigure))
        fig.text(lx + 0.166, ly + 0.006, '2 p.m. · sounding + GFS (lighter)', fontsize=9, color=INK2_, va='center')
    if any(x_[h].get('cloud_cut', 0) >= 1 for x_ in [res] + ([pm['res']] if pm else []) for h in HAZARDS):
        rx = xa + wa - 0.013
        fig.text(rx, ly + 0.006, 'reduced by high cloud', fontsize=9, color=HC_COLOR, ha='right', va='center', weight='bold')
        leg = fig.add_axes([rx - 0.128, ly, 0.02, 0.012]); leg.axis('off'); leg.set_xlim(0, 1); leg.set_ylim(0, 1)
        leg.barh(0.5, 1, height=1, color='none', edgecolor=HC_COLOR, hatch='////', lw=0)

def make_overview(R, inv, runs, path, gfs_note=None):
    g, res, D = R['g'], R['res'], R['D']
    t_obs = R['snd'].meta['valid']
    fig = plt.figure(figsize=(16, 10), dpi=100); fig.patch.set_facecolor(BG)

    # Header
    fig.text(0.04, 0.965, 'Weather Forecast Office', fontsize=11.5, color=MUTED_, weight='bold')
    fig.text(0.04, 0.918, 'San Juan Risk Assessment', fontsize=28, weight='bold', color=INK_)
    a = _ast(t_obs)
    fig.text(0.04, 0.885, f"{a:%A, %B} {a.day}, {a.year} · from the {a.hour % 12 or 12} {'a.m.' if a.hour < 12 else 'p.m.'} AST "
             f"San Juan weather balloon · {SEASON_NAME[R['season']]}", fontsize=13, color=INK2_)

    TOP, R1B = 0.855, 0.50      # row 1 top/bottom
    R2T, R2B = 0.485, 0.20      # row 2
    R3T, R3B = 0.185, 0.035     # row 3 (technical notes)

    # Split the calibrated afternoons: today's 2 p.m. pairs with the sounding; then the next 2 afternoons.
    d0 = _ast(t_obs).date()
    pm = next((r for r in (runs or []) if _ast(r['snd'].meta['valid']).date() == d0), None)
    days = [r for r in (runs or []) if _ast(r['snd'].meta['valid']).date() > d0][:2]

    # Hazard rows per card: Flooding rain and Thunderstorms & lightning always first, then up to 3 more
    # that reach Slight (20%) or higher, highest risk first (ties keep the HAZARDS order).
    today_ri = {h: max([res[h]['ri']] + ([pm['res'][h]['ri']] if pm else [])) for h in HAZARDS}   # higher of now / 2 p.m.
    shown = pick_hazards(today_ri)
    others_today = [h for h in OTHER_HAZARDS if today_ri[h] >= 20]
    extra = [h for h in others_today if h not in shown]
    quiet = [PUBLIC_HAZARD[h].lower().replace('san juan', 'San Juan') for h in OTHER_HAZARDS if today_ri[h] < 20]
    day_rows = [pick_hazards({h: r['res'][h]['ri'] for h in HAZARDS}) for r in days]

    # A0. No GFS (old date / download failed): TODAY card spans the full width, all 7 hazards in two columns
    wide = pm is None and not days
    if TODAY_STYLE == 'legacy':
        _today_legacy(fig, R, pm, [0.04, R1B, 0.92 if wide else 0.56, TOP - R1B])
    if wide and TODAY_STYLE != 'legacy':
        xa, wa = 0.04, 0.92
        gname, gtext, gcol = public_regime(g['GDI'])
        _panel(fig, [xa, R1B, wa, TOP - R1B], fc='#2b2b28', ec=gcol, lw=2.2)
        fig.text(xa + 0.013, TOP - 0.032, ' TODAY ', fontsize=11, weight='bold', color='#111' if gcol in ('#ffd60a', '#52b788', '#48cae4') else '#fff',
                 bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.6', fc=gcol, ec='none'), va='center')
        fig.text(xa + 0.075, TOP - 0.032, 'HAZARDS TO WATCH', fontsize=10.5, color=MUTED_, weight='bold', va='center')
        fig.text(xa + wa - 0.013, TOP - 0.032, f"GDI {_num(g['GDI'])}", fontsize=12, weight='bold', color=INK_, ha='right', va='center')
        fig.text(xa + 0.013, TOP - 0.062, f"Now: {gname}  ·  {gtext}", fontsize=11, color=INK2_, va='center')
        hc_now = R.get('hc', {})
        if USE_HIGH_CLOUD:
            est = not np.isfinite(hc_now.get('gfs', np.nan))
            fig.text(xa + wa - 0.013, TOP - 0.062, f"{_hc_txt(hc_now).replace('☁', '☁ High cloud')}" + (' (est. from sounding)' if est else ''),
                     fontsize=11, weight='bold', color=_hc_col(hc_now), ha='right', va='center')
        hr = _ast(t_obs); now_lbl = f"NOW · {hr.hour % 12 or 12} {'A.M.' if hr.hour < 12 else 'P.M.'} SOUNDING"
        allrows = list(SKILL_ORDER)
        colw = (wa - 0.026 - 0.03) / 2
        for c in range(2):
            fig.text(xa + 0.013 + c * (colw + 0.03) + 0.215, TOP - 0.092, now_lbl, fontsize=9.5, color=MUTED_, weight='bold')
        fig.text(xa + 0.013, TOP - 0.092, 'HIGHER / MEDIUM FORECAST SKILL', fontsize=9.5, color=MUTED_, weight='bold')
        fig.text(xa + 0.013 + colw + 0.03, TOP - 0.092, 'LOWER FORECAST SKILL', fontsize=9.5, color=MUTED_, weight='bold')
        fig.text(xa + wa - 0.013, R1B + 0.012, SKILL_KEY, fontsize=8.5, color=MUTED_, ha='right', va='bottom')
        fig.add_artist(plt.Line2D([xa + 0.013 + colw + 0.015] * 2, [R1B + 0.04, TOP - 0.085], color=MUTED_, lw=0.7, ls=(0, (2, 3))))
        for k, h in enumerate(allrows):
            c, j = divmod(k, 4)
            cx = xa + 0.013 + c * (colw + 0.03); y = TOP - 0.122 - j * 0.058
            r = res[h]; hot = r['ri'] >= 20; cut = r.get('cloud_cut', 0) >= 1
            fig.text(cx, y, SKILL_DOTS[SKILL_TIER[h]], fontsize=8.5, color=MUTED_ if low_skill(h) else INK2_, va='center')
            fig.text(cx + 0.03, y, PUBLIC_HAZARD[h], fontsize=13, color=INK2_ if low_skill(h) else INK_, va='center')
            _risk_bar(fig, [cx + 0.215, y - 0.008, 0.08, 0.016], r, outline=low_skill(h))
            fig.text(cx + 0.303, y, r['cat'], fontsize=12, weight='bold', color=INK_ if hot else MUTED_, va='center')
            fig.text(cx + colw, y, f"{r['ri']:.0f}%", fontsize=12, weight='bold' if cut else 'normal',
                     color=HC_COLOR if cut else (INK2_ if hot else MUTED_), va='center', ha='right')
            fig.text(cx + 0.03, y - 0.022, _drivers(r, 3), fontsize=9.5, color=MUTED_, va='center')
        if any(res[h].get('cloud_cut', 0) >= 1 for h in HAZARDS):
            lx = xa + wa - 0.38
            fig.text(lx, R1B + 0.012, 'reduced by high cloud', fontsize=9.5, color=HC_COLOR, ha='right', va='bottom', weight='bold')
            leg = fig.add_axes([lx - 0.142, R1B + 0.013, 0.022, 0.012]); leg.axis('off'); leg.set_xlim(0, 1); leg.set_ylim(0, 1)
            leg.barh(0.5, 1, height=1, color='none', edgecolor=HC_COLOR, hatch='////', lw=0)

    if not wide:
      if TODAY_STYLE != 'legacy':
        # A. TODAY block (sounding now + 2 p.m. forecast), highlighted -------------------
        xa, wa = 0.04, 0.56
        gname, _, gcol = public_regime(g['GDI'])
        _panel(fig, [xa, R1B, wa, TOP - R1B], fc='#2b2b28', ec=gcol, lw=2.2)
        fig.text(xa + 0.013, TOP - 0.032, ' TODAY ', fontsize=11, weight='bold', color='#111' if gcol in ('#ffd60a', '#52b788', '#48cae4') else '#fff',
                 bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.6', fc=gcol, ec='none'), va='center')
        fig.text(xa + 0.075, TOP - 0.032, 'HAZARDS TO WATCH', fontsize=10.5, color=MUTED_, weight='bold', va='center')
        gtxt = f"GDI {_num(g['GDI'])}" + (f"  →  {_num(pm['g']['GDI'])} at 2 p.m." if pm else '')
        fig.text(xa + wa - 0.013, TOP - 0.032, gtxt, fontsize=12, weight='bold', color=INK_, ha='right', va='center')
        sub = f"Now: {gname}" + (f"  ·  2 p.m.: {public_regime(pm['g']['GDI'])[0]}" if pm else '')
        fig.text(xa + 0.013, TOP - 0.062, sub, fontsize=11, color=INK2_, va='center')
        hc_now = R.get('hc', {})
        if USE_HIGH_CLOUD:
            htxt = 'High cloud ' + _hc_txt(hc_now).replace('☁ ', '') + (f"  →  {_hc_txt(pm['hc']).replace('☁ ', '')} for 2 p.m." if pm else '')
            fig.text(xa + wa - 0.013, TOP - 0.062, '☁ ' + htxt, fontsize=11, weight='bold',
                     color=HC_COLOR if (_hc_col(hc_now) == HC_COLOR or (pm and _hc_col(pm['hc']) == HC_COLOR)) else MUTED_,
                     ha='right', va='center')
        hr = _ast(t_obs); now_lbl = f"NOW · {hr.hour % 12 or 12} {'A.M.' if hr.hour < 12 else 'P.M.'} SOUNDING"
        cN, cP = xa + 0.20, xa + 0.385            # column starts: now, 2 p.m.
        fig.text(cN, TOP - 0.092, now_lbl, fontsize=9.5, color=MUTED_, weight='bold')
        fig.text(cP, TOP - 0.092, '2 P.M. · SOUNDING + GFS', fontsize=9.5, color=MUTED_, weight='bold')
        row = 0.032
        fig.text(xa + 0.013, TOP - 0.092, 'Skill  ●●● high · ●●○ med · ●○○ low', fontsize=7.5, color=MUTED_)
        gap_lo = 0.012
        drawn_div = False
        for k, h in enumerate(shown):
            lo_ = low_skill(h)
            y = TOP - 0.114 - k * row - (gap_lo if lo_ else 0)
            if lo_ and not drawn_div:
                yd = y + row + gap_lo - 0.026
                fig.add_artist(plt.Line2D([xa + 0.013, xa + wa - 0.013], [yd, yd], color=MUTED_, lw=0.8, ls=(0, (2, 3))))
                fig.text(xa + wa - 0.013, yd + 0.003, 'LOWER FORECAST SKILL', fontsize=7.5, color=MUTED_, weight='bold', ha='right', va='bottom')
                drawn_div = True
            r = res[h]
            fig.text(xa + 0.013, y, SKILL_DOTS[SKILL_TIER[h]], fontsize=8, color=INK2_ if not lo_ else MUTED_, va='center')
            fig.text(xa + 0.042, y, PUBLIC_HAZARD[h], fontsize=12.5, color=INK2_ if lo_ else INK_, va='center')
            for cx_, rr, wbar in ((cN, r, 0.055), (cP, pm['res'][h] if pm else None, 0.055)):
                if rr is None:
                    fig.text(cx_, y, 'not applicable', fontsize=11, color=MUTED_, va='center'); continue
                _risk_bar(fig, [cx_, y - 0.007, wbar, 0.014], rr, outline=low_skill(h))
                hot = rr['ri'] >= 20
                cut = rr.get('cloud_cut', 0) >= 1
                fig.text(cx_ + wbar + 0.007, y, rr['cat'], fontsize=11.5, weight='bold', color=INK_ if hot else MUTED_, va='center')
                fig.text(cx_ + wbar + 0.10, y, f"{rr['ri']:.0f}%", fontsize=11.5, weight='bold' if cut else 'normal',
                         color=HC_COLOR if cut else (INK2_ if hot else MUTED_), va='center', ha='right')
            if pm:
                dlt = pm['res'][h]['ri'] - r['ri']
                arrow, acol = ('▲', INK_) if dlt >= 10 else ('▼', INK_) if dlt <= -10 else ('→', MUTED_)
                fig.text(cP - 0.012, y, arrow, fontsize=11, color=acol, va='center', ha='center')
            dtx = _drivers(r); dtx = dtx if len(dtx) <= 40 else dtx[:39].rstrip(' ·') + '…'
            fig.text(xa + 0.042, y - 0.0165, dtx, fontsize=8, color=MUTED_, va='center')
        foot = []
        if extra:
            foot.append('Also Slight+ today: ' + ', '.join(PUBLIC_HAZARD[h].lower().replace('san juan', 'San Juan') for h in extra) + '.')
        if quiet:
            foot.append('Very low today: ' + ', '.join(quiet) + '.')
        any_cut = (any(x[h].get('cloud_cut', 0) >= 1 for x in [res] + ([pm['res']] if pm else []) for h in shown)
                   or any(d_['res'][h].get('cloud_cut', 0) >= 1 for d_, rows_d in zip(days, day_rows) for h in rows_d))
        if any_cut:
            lx = xa + wa - 0.013
            fig.text(lx, R1B + 0.012, 'reduced by high cloud', fontsize=9.5, color=HC_COLOR, ha='right', va='bottom', weight='bold')
            leg = fig.add_axes([lx - 0.142, R1B + 0.013, 0.022, 0.012]); leg.axis('off'); leg.set_xlim(0, 1); leg.set_ylim(0, 1)
            leg.barh(0.5, 1, height=1, color='none', edgecolor=HC_COLOR, hatch='////', lw=0)

      # B. Next 2 afternoons (sounding + GFS change) — right ---------------------------
      xb, wb = 0.615, 0.345
      _panel(fig, [xb, R1B, wb, TOP - R1B], 'Next 2 afternoons · 2 p.m.', 'sounding + GFS change')
      if days:
          gap = 0.01
          cw = (wb - 0.026 - gap) / 2
          cy0, ch = R1B + 0.014, TOP - R1B - 0.062
          for k, (r, rows_d) in enumerate(zip(days, day_rows)):
              nm, _, col = public_regime(r['g']['GDI'])
              cx = xb + 0.013 + k * (cw + gap)
              card = fig.add_axes([cx, cy0, cw, ch]); card.axis('off'); card.set_xlim(0, 1); card.set_ylim(0, 1)
              card.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0,rounding_size=0.03', fc='#2c2c2a', ec='none',
                                            mutation_aspect=cw / ch * 1.6))
              card.add_patch(Rectangle((0, 0.975), 1, 0.025, color=col, lw=0))
              av = _ast(r['snd'].meta['valid'])
              ytop = cy0 + ch - 0.035
              fig.text(cx + 0.01, ytop, f'{av:%a} {av:%b} {av.day}', fontsize=13, weight='bold', color=INK_)
              fig.text(cx + cw - 0.01, ytop, f"GDI {_num(r['g']['GDI'])}", fontsize=10, color=MUTED_, ha='right')
              if USE_HIGH_CLOUD:
                  cl = r['res']['tstorm'].get('cloud', 0)
                  fig.text(cx + 0.01, ytop - 0.051, f"☁ High cloud {r['hc']['pct']:.0f}%" + (f"  ·  {10*cl:+.0f}%".replace('-', '−') if cl else ''),
                           fontsize=9.5, weight='bold', color=_hc_col(r['hc']))
              fig.text(cx + 0.01, ytop - 0.027, nm, fontsize=11, color=INK2_)
              drawn_div = False
              for j, h in enumerate(rows_d):
                  lo_ = low_skill(h)
                  rr = r['res'][h]; y = ytop - (0.078 if USE_HIGH_CLOUD else 0.062) - j * 0.0265 - (0.006 if lo_ else 0)
                  if lo_ and not drawn_div:
                      yd = y + 0.0175
                      fig.add_artist(plt.Line2D([cx + 0.008, cx + cw - 0.008], [yd, yd], color=MUTED_, lw=0.7, ls=(0, (2, 3))))
                      drawn_div = True
                  hot = rr['ri'] >= 20
                  cut = rr.get('cloud_cut', 0) >= 1
                  fig.text(cx + 0.008, y, SKILL_DOTS[SKILL_TIER[h]], fontsize=6, color=MUTED_ if lo_ else INK2_)
                  fig.text(cx + 0.027, y, SHORT_HAZARD[h], fontsize=8, color=MUTED_ if lo_ else INK2_)
                  fig.text(cx + cw - 0.01, y, ('☁ ' if cut else '') + f"{rr['ri']:.0f}% {rr['cat']}", fontsize=8, weight='bold',
                           color=HC_COLOR if cut else (INK_ if hot else MUTED_), ha='right')
                  _risk_bar(fig, [cx + 0.01, y - 0.0095, cw - 0.02, 0.0045], rr, outline=lo_)
      else:
          fig.text(xb + 0.013, TOP - 0.09, '\n'.join(textwrap.wrap(
              'Outlook unavailable (no GFS profile at the sounding time, or no connection).', 50)), fontsize=11, color=MUTED_, va='top')

    # D. GDI components ---------------------------------------------------------
    xc, wc = 0.67, 0.29
    _panel(fig, [xc, R2B, wc, R2T - R2B], 'What helps or holds storms back')
    parts = [('Heat & moisture (CBI)', g['CBI']), ('Warm air aloft (MWI)', g['MWI']), ('Dry-air lid (II)', g['II'])]
    bx = fig.add_axes([xc + 0.13, R2B + 0.045, wc - 0.17, R2T - R2B - 0.105]); bx.set_facecolor(PANEL)
    lim = max(50, max(abs(v) for _, v in parts) + 20)
    for i, (lbl, v) in enumerate(parts):
        y = len(parts) - i
        bx.barh(y, v, height=0.55, color=BLUE if v >= 0 else RED, lw=0)
        bx.text(v + (3 if v >= 0 else -3), y, _num(v), va='center', ha='left' if v >= 0 else 'right',
                fontsize=12, weight='bold', color=INK_)
    bx.axvline(0, color=MUTED_, lw=1); bx.set_xlim(-lim, lim); bx.set_ylim(0.4, len(parts) + 0.6)
    bx.set_yticks(range(len(parts), 0, -1)); bx.set_yticklabels([p[0] for p in parts], fontsize=10.5, color=INK_)
    bx.tick_params(axis='y', length=0); bx.set_xticks([])
    for s_ in bx.spines.values(): s_.set_visible(False)
    bx.set_xlabel('◀ holds storms back      helps storms ▶', fontsize=9, color=MUTED_, labelpad=3)

    # E. Mini sounding ----------------------------------------------------------
    xd, wd = 0.365, 0.29
    _panel(fig, [xd, R2B, wd, R2T - R2B], 'Air above San Juan')
    d = R['snd'].df.query('PRES >= 300').dropna(subset=['TEMP'])
    sx = fig.add_axes([xd + 0.05, R2B + 0.045, wd - 0.065, R2T - R2B - 0.1]); sx.set_facecolor(PANEL)
    sx.set_yscale('log'); sx.set_ylim(1030, 300); sx.minorticks_off()
    sx.set_yticks([1000, 850, 700, 500, 300]); sx.set_yticklabels(['sfc', '1.5 km', '3 km', '5.8 km', '9.6 km'], fontsize=9)
    if inv:
        sx.axhspan(inv['p_base'], inv['p_top'], color=AMBER, alpha=0.30, lw=0)
        sx.text(0.02, inv['p_top'], ' stable layer', transform=sx.get_yaxis_transform(), fontsize=9, color=AMBER,
                va='bottom', weight='bold')
    dd = d.dropna(subset=['DWPT'])
    sx.fill_betweenx(dd.PRES, dd.DWPT, np.interp(dd.PRES[::-1], d.PRES[::-1], d.TEMP[::-1])[::-1], color=INK2_, alpha=0.07)
    sx.plot(d.TEMP, d.PRES, color=RED, lw=2.3); sx.plot(dd.DWPT, dd.PRES, color=GREEN, lw=2.3)
    sx.text(0.98, 0.94, 'Temperature', color=RED, ha='right', transform=sx.transAxes, fontsize=9.5, weight='bold')
    sx.text(0.98, 0.86, 'Dew point', color=GREEN, ha='right', transform=sx.transAxes, fontsize=9.5, weight='bold')
    sx.tick_params(labelsize=9, length=0); sx.grid(color=LINE_, lw=0.5)
    for s_ in ('top', 'right'): sx.spines[s_].set_visible(False)
    sx.set_xlabel('°C  (wider gap = drier air)', fontsize=9, color=MUTED_, labelpad=2)

    # F. Legacy Risk Assessment chart (original TJSJ rules, 4 bugs fixed) ------------------
    xe, we = 0.04, 0.31
    LG = R['legacy']; ls = LG['scores']
    _panel(fig, [xe, R2B, we, R2T - R2B], 'Legacy risk assessment', 'TJSJ RAOB rules')
    lx = fig.add_axes([xe + 0.012, R2B + 0.09, we - 0.024, R2T - R2B - 0.14]); lx.set_facecolor(PANEL)
    for i, h in enumerate(LEGACY_HZ):
        v = ls[h]
        if low_skill(h):
            lx.bar(i, max(v, 0.06), width=0.62, color=legacy_color(v), alpha=0.2, lw=0)
            lx.bar(i, max(v, 0.06), width=0.62, facecolor='none', edgecolor=legacy_color(v), lw=1.5)
        else:
            lx.bar(i, max(v, 0.06), width=0.62, color=legacy_color(v), lw=0)
        lx.text(i, min(v, 10.0) + 0.25, f"{round10(10 * v):.0f}%", ha='center', va='bottom', fontsize=10, weight='bold', color=INK_)
    if ls['twind'] > 5:
        lx.text(LEGACY_HZ.index('twind'), min(ls['twind'], 10) / 2, f"gust\n{LG['gust']:.0f} kt", ha='center', va='center', fontsize=8, color='#111', weight='bold')
    lx.set_ylim(0, 11.5); lx.set_xlim(-0.6, len(LEGACY_HZ) - 0.4); lx.set_yticks([])
    lx.set_xticks(range(len(LEGACY_HZ))); lx.set_xticklabels([f"{LEGACY_LABEL[h]}\n{SKILL_DOTS[SKILL_TIER[h]]}" for h in LEGACY_HZ], fontsize=8.5, color=INK2_)
    k1 = next(i for i, h in enumerate(LEGACY_HZ) if low_skill(h))
    lx.axvline(k1 - 0.5, color=MUTED_, lw=0.8, ls=(0, (2, 3)))
    lx.text(k1 - 0.42, 11.3, 'lower skill', fontsize=8, color=MUTED_, weight='bold', va='top')
    lx.tick_params(axis='x', length=0, pad=3)
    for s_ in ('top', 'right', 'left'): lx.spines[s_].set_visible(False)
    lx.spines['bottom'].set_color(LINE_)
    legend_items = [('#999999', '≤20%'), ('#33CC00', '20–40%'), ('#EAEA00', '40–50%'), ('#FF0000', '>50%')]
    for k, (c, t) in enumerate(legend_items):
        bx_ = xe + 0.016 + k * 0.072
        fig.add_artist(Rectangle((bx_, R2B + 0.016), 0.012, 0.012, color=c, lw=0, transform=fig.transFigure))
        fig.text(bx_ + 0.016, R2B + 0.022, t, fontsize=8.5, color=INK2_, va='center')

    # G. Footer: Key numbers (⅓) + Technical notes (⅔) ------------------------------------
    fx, fw = 0.04, 0.30
    pw, pwn = D['PWAT'], D['sc']['pwat']
    _panel(fig, [fx, R3B, fw, R3T - R3B], 'Key numbers', f"PWAT climo {pwn:.2f} in · LR °C/km")
    def _f(v, fmt='.0f', unit=''):
        return (format(v, fmt) + unit).replace('-', '−') if v is not None and np.isfinite(v) else 'n/a'
    LD = R['legacy']['D']
    kn = [('PWAT in', f"{pw:.2f}"), ('CAPE', _f(D['CAPE'])), ('LI', _f(D['LI'], '.1f')),
          ('CIN', _f(D.get('CIN'))), ('K / TT', f"{_f(D['KI'])}/{_f(D['TT'])}"),
          ('T500 °C', _f(D['T500'], '.1f')), ('LR 85–70', _f(D['LR87'], '.1f')), ('LR 70–50', _f(D['LR75'], '.1f')),
          ('LR 50–30', _f(D['LR53'], '.1f')), ('θe def. K', _f(D['dthe'])),
          ('RH 85–70', _f(D['RH87'], '.0f', '%')), ('RH 70–50', _f(D['RH75'], '.0f', '%')), ('RH 50–30', _f(D['RH53'], '.0f', '%')),
          ('VIL-OTD', _f(D['VOTD'])), ('0–3 km', f"{_f(D['wspd_03'])} kt {_compass(D['wdir_03'])}")]
    ncol, nrow = 3, 5
    cw_ = (fw - 0.02) / ncol; rh_ = (R3T - R3B - 0.052) / nrow
    for k, (lbl, val) in enumerate(kn):
        c, r_ = k // nrow, k % nrow
        tx = fx + 0.01 + c * cw_; ty = R3T - 0.058 - r_ * rh_
        fig.text(tx, ty, lbl, fontsize=8.5, color=MUTED_, va='center')
        fig.text(tx + cw_ - 0.012, ty, val, fontsize=9, weight='bold', color=INK_, va='center', ha='right')

    nx, nw = 0.355, 0.605
    _panel(fig, [nx, R3B, nw, R3T - R3B], 'Technical notes')
    y = R3T - 0.052
    for mark, line in _tech_notes(R, inv, ([pm] if pm else []) + days, gfs_note if not (pm or days) else None):
        wrapped = textwrap.wrap(line, 126)
        if y - 0.019 * (len(wrapped) - 1) < R3B + 0.004:
            continue                      # skip a note that no longer fits; a shorter one after it may still fit
        for j, w_ in enumerate(wrapped):
            mcol = AMBER if mark == '⚠' else HC_COLOR if mark == '☁' else INK2_ if mark == 'ℹ' else MUTED_
            fig.text(nx + 0.013, y, mark if j == 0 else '', fontsize=9.5, color=mcol)
            fig.text(nx + 0.026, y, w_, fontsize=9.5, color=INK_ if mark in ('⚠', '☁') else INK2_)
            y -= 0.019

    fig.savefig(path, dpi=100 * IMG_SCALE, facecolor=BG); plt.close(fig)
    return path

# Run: sounding → risk engine → overview graphic
SND = load_raob()
INV = detect_inversion(SND)
H_GFS, GFS_NOTE = None, None
_age = (dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - SND.meta['valid']).days
if not RUN_GFS:
    GFS_NOTE = 'GFS outlook turned off (RUN_GFS = False): sounding only; high cloud estimated from 500–300 hPa RH.'
elif _age > GFS_MAX_AGE_DAYS:
    GFS_NOTE = f'No GFS for this date (sounding {_age} days old, archive ~{GFS_MAX_AGE_DAYS} days): sounding only; high cloud from 500–300 hPa RH.'
else:
    try:
        H_GFS = fetch_gfs_point(SJU_LAT, SJU_LON, SND.meta['valid'])
    except Exception as e:
        print('⚠️ GFS download failed:', e)
        GFS_NOTE = 'GFS download failed: sounding only; high cloud from 500–300 hPa RH.'
STENCIL = None
if H_GFS is not None:
    try:
        STENCIL = fetch_gfs_stencil(SJU_LAT, SJU_LON, SND.meta['valid'])
    except Exception as e:
        print('⚠️ GFS lift (200/700 hPa stencil) skipped:', e)

HC_NOW = high_cloud_for(SND, H_GFS, use_sounding=True)
ENV_NOW = gfs_env_for(SND, H_GFS, STENCIL)
R = run_engine(SND, HC_NOW, ENV_NOW)
print(f"TJSJ {SND.meta['valid']:%Y-%m-%d %HZ} | GDI {R['g']['GDI']:.1f} | source: {SND.meta['url']}")
print(f"  High cloud now: {HC_NOW['pct']:.0f}% ({HC_NOW['src']}; GFS {HC_NOW['gfs']:.0f}%, sounding est. {HC_NOW['snd']:.0f}%)".replace('nan%', 'n/a'))
print(f"  Lift now: {ENV_NOW['lift'] or 'n/a'} (×{ENV_NOW['mult']:.2f}) | 200 hPa div {ENV_NOW['div200']:.2e} | 700 hPa vort {ENV_NOW['vort700']:.2e} | 10 m wind {ENV_NOW['sfc_wind']:.0f} kt")
_X = R['D']; print(f"  WCD {_X['WCD']:.1f} km | RH850-500 {_X['RH85']:.0f}% | Corfidi up {_X['CORFIDI_UP']:.0f} kt | MLCAPE {_X['MLCAPE']:.0f} | "
                   f"CAPE(-10..-30) {_X['CAPE_LTG']:.0f} | CAPE0-3 {_X['CAPE03']:.0f} | DCAPE {_X['DCAPE']:.0f} | shear0-6 {_X['SHEAR06']:.0f} kt | WBZ {_X['WBZ_Z']/1000:.1f} km")

GFS_RUNS = []
if H_GFS is not None:
    try:
        for s in calibrated_outlook(SND, H_GFS, days=3):
            GFS_RUNS.append(run_engine(s, high_cloud_for(s, H_GFS, use_sounding=False), gfs_env_for(s, H_GFS, STENCIL)))
        for r in GFS_RUNS:
            print(f"  {r['snd'].meta['valid']:%a %d %HZ}: calibrated GDI {r['g']['GDI']:.1f} | high cloud {r['hc']['pct']:.0f}% | lift {r['env']['lift'] or 'n/a'}")
    except Exception as e:
        print('⚠️ GFS outlook skipped:', e)
        GFS_NOTE = f"GFS not available for this date ({e}): sounding only."
if not GFS_RUNS and GFS_NOTE is None:
    GFS_NOTE = 'No GFS afternoons available for this date: sounding only.'
if GFS_NOTE:
    print('ℹ️', GFS_NOTE)


os.makedirs(OUT_DIR, exist_ok=True)
# IMAGE = make_overview(R, INV, GFS_RUNS, f"{OUT_DIR}/tjsj_{R['snd'].meta['valid']:%Y%m%d%H}_overview.png", GFS_NOTE)
IMAGE = make_overview(R, INV, GFS_RUNS, f"{OUT_DIR}/risk_latest.png", GFS_NOTE)
print('Saved', IMAGE)
from IPython.display import Image as _Img
os.makedirs(OUT_DIR, exist_ok=True)
IMAGE = make_overview(R, INV, GFS_RUNS, f"{OUT_DIR}/risk_latest.png", GFS_NOTE)
print(f"Successfully generated graphic: {IMAGE}")
# if DOWNLOAD_IMAGE:
#     try:
#         from google.colab import files; files.download(IMAGE)
#     except ImportError:
#         pass
