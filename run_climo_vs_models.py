"""
Sounding climatology vs. GFS / ECMWF at San Juan (TJSJ) — Single 10-day outlook plot.

GFS is downloaded once (3-hourly to 168 h, 6-hourly to 360 h) and ECMWF is requested once.
The figure outputs to web_images/climo_vs_models.png.
"""
from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import cfgrib  # noqa: F401  (engine for xarray)
    import metpy.calc as mpcalc
    from metpy.units import units
except ImportError as e:
    raise SystemExit(f"Missing required scientific package '{e.name}'.")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.plotting.register_matplotlib_converters()

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SITE_NAME   = "San Juan, PR (TJSJ)"
LAT, LON    = 18.44, 294.0                 # 294°E = 66.0°W
MODEL       = "Both"                       # ["GFS", "ECMWF", "Both"]
COLOR_STYLE = "Enhanced"                   # ["Legacy", "Enhanced"]
MAX_WORKERS = 6                            # NOMADS rate-limiting buffer
LOCAL_TZ    = ZoneInfo("America/Puerto_Rico")
GRIB_FILE   = "gfs_point.grib2"
LEVELS      = [1000, 925, 850, 700, 500, 250]

# Single target window: 10-Day Outlook (Days 0–10)
OUTPUT_FILE = "web_images/climo_vs_models.png"
WINDOW = dict(
    name="10-Day Outlook",
    span="Days 0–10",
    start=0,
    end=240,
    step=6,
    out=OUTPUT_FILE
)

FETCH_END = 360
GFS_HOURS = list(range(0, 169, 3)) + list(range(174, FETCH_END + 1, 6))
EC_HOURS  = [h for h in range(0, FETCH_END + 1, 3) if h <= 144 or h % 6 == 0]

NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
NOMADS_PROD   = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

ECMWF_API    = "https://single-runs-api.open-meteo.com/v1/forecast"
ECMWF_MODELS = ["ecmwf_ifs025", "ecmwf_ifs"]
ECMWF_DELAY  = timedelta(hours=8)

# ──────────────────────────────────────────────────────────────
# Monthly Sounding Climatology
# ──────────────────────────────────────────────────────────────
CLIMO = {
    "pwat": {
        1: [0.548, 0.876, 1.189, 1.337, 1.502, 1.947, 2.218],
        2: [0.568, 0.837, 1.147, 1.303, 1.465, 1.894, 2.569],
        3: [0.702, 0.862, 1.167, 1.325, 1.490, 1.984, 2.252],
        4: [0.707, 0.930, 1.290, 1.480, 1.688, 2.192, 2.628],
        5: [0.898, 1.139, 1.527, 1.762, 1.981, 2.456, 2.679],
        6: [0.902, 1.266, 1.600, 1.774, 1.981, 2.394, 2.558],
        7: [0.989, 1.250, 1.590, 1.754, 1.959, 2.412, 2.632],
        8: [0.938, 1.291, 1.644, 1.828, 2.026, 2.494, 2.746],
        9: [0.991, 1.381, 1.720, 1.910, 2.081, 2.447, 2.731],
        10: [1.033, 1.345, 1.714, 1.905, 2.106, 2.503, 2.744],
        11: [0.704, 1.088, 1.516, 1.748, 1.985, 2.443, 2.674],
        12: [0.644, 0.887, 1.256, 1.446, 1.656, 2.262, 2.701],
    },
    "z250": {
        1:  [10740, 10800, 10870, 10900, 10930, 11000, 11010, 11020],
        2:  [10710, 10780, 10860, 10900, 10930, 11000, 11010, 11020],
        3:  [10710, 10780, 10860, 10900, 10930, 11000, 11010, 11020],
        4:  [10760, 10800, 10880, 10910, 10930, 11010, 11020, 11040],
        5:  [10780, 10850, 10900, 10930, 10960, 11010, 11020, 11030],
        6:  [10822, 10870, 10920, 10940, 10960, 11020, 11030, 11050],
        7:  [10810, 10870, 10920, 10940, 10960, 11000, 11010, 11020],
        8:  [10850, 10900, 10940, 10960, 10970, 11010, 11030, 11060],
        9:  [10820, 10880, 10930, 10950, 10970, 11010, 11020, 11030],
        10: [10800, 10850, 10910, 10940, 10960, 11010, 11020, 11030],
        11: [10780, 10840, 10890, 10920, 10950, 11000, 11010, 11030],
        12: [10710, 10810, 10880, 10920, 10940, 11000, 11010, 11020],
    },
    "thk1000_500": {
        1:  [5592, 5673.48, 5706, 5721, 5738, 5768, 5767.91, 5777],
        2:  [5627, 5662.18, 5698, 5718, 5737, 5777, 5773.11, 5803],
        3:  [5633, 5662.92, 5701, 5719, 5737, 5783, 5774.68, 5808],
        4:  [5651, 5684.08, 5715, 5730, 5744, 5783, 5774.87, 5805],
        5:  [5675, 5705.55, 5732, 5745, 5758, 5792, 5784.38, 5811],
        6:  [5699, 5721.53, 5743, 5753, 5764, 5791, 5784.84, 5809],
        7:  [5703, 5725.67, 5745, 5755, 5765, 5785, 5783.17, 5799],
        8:  [5688, 5734.36, 5754, 5763, 5772, 5797, 5791.56, 5837],
        9:  [5710, 5736.55, 5757, 5766, 5774, 5797, 5793.71, 5843],
        10: [5673, 5728.46, 5751, 5762, 5772, 5800, 5795.07, 5842],
        11: [5680, 5710.37, 5737, 5750, 5761, 5789, 5786.23, 5802],
        12: [5642, 5691.65, 5721, 5735, 5747, 5776, 5775.22, 5796],
    },
    "thk1000_850": {
        1:  [1348, 1385.94, 1395, 1398, 1402, 1415, 1411.16, 1423],
        2:  [1363, 1382.5,  1392, 1397, 1401, 1417, 1410.94, 1447],
        3:  [1362, 1382.19, 1393, 1397, 1402, 1420, 1413.47, 1438],
        4:  [1367, 1386.4,  1398, 1402, 1407, 1423, 1418.73, 1445],
        5:  [1380, 1393.82, 1404, 1409, 1414, 1427, 1423.95, 1437],
        6:  [1373, 1399.52, 1409, 1413, 1417, 1430, 1426.46, 1449],
        7:  [1389, 1400.93, 1410, 1414, 1417, 1430, 1426.72, 1444],
        8:  [1373, 1402.6,  1413, 1416, 1419, 1432, 1429.14, 1450],
        9:  [1368, 1402.71, 1413, 1417, 1420, 1435, 1430.44, 1451],
        10: [1376, 1401.01, 1411, 1415, 1419, 1432, 1428.84, 1452],
        11: [1383, 1395.38, 1405, 1410, 1414, 1427, 1423.57, 1434],
        12: [1365, 1388.2,  1398, 1403, 1407, 1419, 1416.64, 1434],
    },
    "rh700_500": {
        1:  [1.0, 0.80798,  9.0, 20.0, 35.0, 72.5, 58.9598, 90.0],
        2:  [1.5, 2.4456,   9.5, 18.5, 33.0, 83.0, 60.9731, 95.0],
        3:  [1.0, 4.98839, 12.0, 21.5, 36.0, 86.5, 61.4515, 96.0],
        4:  [1.5, 6.1214,  17.5, 29.5, 46.5, 89.5, 72.0869, 100.0],
        5:  [2.0, 6.110963, 27.5, 43.0, 62.0, 93.5, 90.6016, 99.0],
        6:  [5.5, 6.63565, 30.0, 41.5, 57.5, 88.5, 81.5105, 98.0],
        7:  [5.5, 6.10043, 25.0, 35.0, 49.5, 88.5, 76.067,  96.0],
        8:  [2.0, 6.94979, 27.0, 38.0, 53.5, 89.0, 79.3716, 98.0],
        9:  [2.0, 8.39737, 32.5, 44.0, 60.0, 91.0, 84.4477, 99.0],
        10: [1.5, 6.26088, 32.5, 47.0, 63.0, 93.0, 90.1364, 99.5],
        11: [1.5, 4.98167, 22.5, 38.0, 56.5, 93.5, 86.9409, 98.5],
        12: [1.5, 1.2597,  11.5, 22.0, 40.0, 83.0, 66.5825, 97.0],
    },
    "lr700_500": {
        1:  [2.25926, 3.96635, 5.07407, 5.59259, 6.18519, 7.22222, 7.21252, 8.62963],
        2:  [2.85185, 3.98327, 5.14815, 5.74074, 6.25926, 7.51852, 7.35217, 8.03704],
        3:  [3.22222, 3.96973, 5.00000, 5.59259, 6.11111, 7.29630, 7.15856, 8.25926],
        4:  [2.48148, 3.90554, 4.92593, 5.37037, 5.88889, 7.07407, 6.86038, 7.81481],
        5:  [3.14815, 4.51736, 5.29630, 5.66667, 6.03704, 7.07407, 6.82812, 7.81481],
        6:  [3.44444, 4.66974, 5.44444, 5.88889, 6.25926, 7.22222, 7.02786, 8.25926],
        7:  [3.00000, 4.56614, 5.44444, 5.81481, 6.25926, 7.44444, 7.13185, 8.03704],
        8:  [3.66667, 4.61008, 5.37037, 5.74074, 6.11111, 7.22222, 6.89934, 8.62963],
        9:  [2.81481, 4.82058, 5.51852, 5.81481, 6.11111, 6.92593, 6.77528, 7.74074],
        10: [3.88889, 4.69751, 5.37037, 5.74074, 6.03704, 6.77778, 6.71206, 7.37037],
        11: [2.55556, 4.42885, 5.29630, 5.59259, 5.96296, 7.00000, 6.78526, 7.81481],
        12: [3.00000, 4.09806, 5.14815, 5.59259, 6.11111, 7.29630, 7.10413, 7.96296],
    },
    "t500": {
        1:  [-15.9, -10.8289, -8.3, -7.1, -5.9, -3.5, -3.6021, -2.1],
        2:  [-14.9, -11.5172, -8.7, -7.1, -5.9, -3.1, -3.21974, -1.1],
        3:  [-14.9, -11.0457, -8.5, -7.1, -5.9, -3.5, -3.4684, -2.1],
        4:  [-13.3, -10.3915, -7.9, -6.7, -5.7, -2.3, -3.22301, -0.7],
        5:  [-11.5, -9.69815, -7.9, -6.9, -5.9, -3.9, -4.16782, -1.3],
        6:  [-11.5, -9.57936, -7.9, -6.9, -6.1, -3.9, -4.41946, -2.5],
        7:  [-12.7, -9.47666, -7.7, -6.7, -5.9, -3.9, -4.15369, -0.7],
        8:  [-11.5, -8.82237, -7.1, -6.3, -5.5, -3.5, -3.84134, -2.3],
        9:  [-13.1, -8.83121, -7.3, -6.5, -5.7, -3.7, -4.10235, -0.7],
        10: [-11.1, -9.0155,  -7.3, -6.3, -5.5, -3.5, -3.75062, -1.2],
        11: [-12.3, -9.48001, -7.5, -6.5, -5.5, -3.1, -3.57947, -1.5],
        12: [-14.9, -10.1007, -7.7, -6.7, -5.7, -3.3, -3.45135, -2.3],
    },
    "wind925": {
        1:  [1, 2.98599, 11, 16, 21, 30, 28.7626, 36],
        2:  [1, 3.32364, 11, 16, 20, 29, 27.7135, 35],
        3:  [0, 2.18878,  9, 14, 18, 27, 25.4318, 34],
        4:  [0, 1.25297,  9, 13, 17, 27, 24.5808, 33],
        5:  [0, 1.61853,  9, 13, 18, 26, 24.9221, 36],
        6:  [1, 5.6178,  12, 17, 20, 27, 26.5768, 31],
        7:  [2, 8.23691, 15, 18, 21, 32, 28.2389, 42],
        8:  [0, 4.61231, 13, 16, 19, 37, 28.265,  47],
        9:  [1, 1.06612, 10, 14, 18, 36, 27.1652, 59],
        10: [1, 1.82111,  9, 13, 16, 27, 23.7305, 32],
        11: [1, 2.26207, 10, 14, 19, 29, 26.7422, 43],
        12: [1, 3.45999, 12, 16, 20, 31, 28.2134, 43],
    },
    "rh850_700": {
        1:  [7.5,  21.1201, 45.0, 52.0, 64.5, 96.5, 89.7719, 98.5],
        2:  [7.0,  21.4326, 43.5, 51.5, 62.0, 95.0, 86.6059, 100.0],
        3:  [10.0, 21.3743, 44.0, 52.5, 64.5, 95.0, 89.2347, 100.0],
        4:  [10.5, 25.5079, 47.5, 58.5, 74.0, 95.5, 95.8836, 99.5],
        5:  [10.5, 33.1571, 53.5, 66.5, 78.0, 96.5, 98.0617, 100.0],
        6:  [11.0, 32.7324, 52.0, 62.0, 72.5, 93.5, 91.8842, 98.5],
        7:  [14.5, 33.9037, 53.0, 61.0, 71.0, 94.0, 89.7682, 99.0],
        8:  [3.5,  35.7567, 54.5, 64.5, 74.0, 93.5, 92.3529, 100.0],
        9:  [10.5, 38.3247, 57.0, 67.0, 76.0, 94.5, 93.4722, 99.5],
        10: [15.0, 40.2236, 60.0, 70.0, 79.0, 95.0, 96.8554, 100.0],
        11: [10.0, 32.9980, 53.0, 67.5, 80.0, 97.0, 100.3300, 100.0],
        12: [4.0,  24.2038, 46.5, 54.5, 70.5, 96.5, 92.5367, 100.0],
    },
    "lr850_700": {
        1:  [4.00, 4.60, 5.10, 5.40, 5.70, 6.40, 6.80, 7.50],
        2:  [4.10, 4.60, 5.10, 5.40, 5.70, 6.40, 6.80, 7.40],
        3:  [4.10, 4.70, 5.10, 5.40, 5.70, 6.40, 6.80, 7.70],
        4:  [4.20, 4.70, 5.20, 5.40, 5.70, 6.40, 6.80, 7.60],
        5:  [4.30, 4.80, 5.30, 5.50, 5.80, 6.40, 6.80, 7.60],
        6:  [4.30, 4.90, 5.40, 5.60, 5.90, 6.50, 6.90, 7.70],
        7:  [4.40, 4.90, 5.40, 5.60, 5.90, 6.50, 6.90, 7.60],
        8:  [4.40, 5.00, 5.50, 5.70, 6.00, 6.60, 6.90, 7.70],
        9:  [4.40, 5.00, 5.50, 5.70, 6.00, 6.60, 6.90, 7.50],
        10: [4.40, 5.00, 5.50, 5.70, 6.00, 6.60, 6.90, 7.40],
        11: [4.30, 4.90, 5.40, 5.60, 5.90, 6.50, 6.90, 7.40],
        12: [4.10, 4.80, 5.30, 5.50, 5.80, 6.40, 6.90, 7.50],
    },
    "thetae850": {
        1:  [306.3, 317.01,  325.4, 329.6, 333.5, 339.7, 341.035, 344.5],
        2:  [304.8, 315.825, 324.1, 328.3, 332.2, 340.2, 339.954, 346.9],
        3:  [302.3, 316.397, 324.7, 328.8, 332.8, 341.0, 340.579, 349.0],
        4:  [305.8, 318.395, 327.4, 331.8, 335.9, 345.7, 344.552, 351.6],
        5:  [306.8, 323.608, 332.2, 336.6, 340.5, 348.5, 348.541, 353.7],
        6:  [310.4, 324.567, 332.9, 337.7, 341.4, 348.6, 349.18,  353.3],
        7:  [313.9, 325.953, 334.8, 338.8, 342.4, 350.1, 350.424, 353.8],
        8:  [310.6, 327.198, 336.0, 340.0, 343.9, 351.1, 351.841, 353.3],
        9:  [307.1, 327.972, 336.5, 340.3, 343.9, 350.0, 351.411, 352.9],
        10: [310.0, 327.971, 336.3, 340.1, 343.4, 350.1, 350.997, 354.7],
        11: [309.4, 324.683, 333.0, 337.5, 341.0, 348.2, 348.813, 354.3],
        12: [305.3, 319.181, 328.1, 332.5, 336.4, 344.5, 344.563, 349.1],
    },
    "t925": {
        1:  [14.0, 16.4488, 17.8, 18.6, 19.2, 20.8, 20.6441, 22.2],
        2:  [15.4, 15.9146, 17.5, 18.2, 19.0, 21.2, 20.5920, 23.0],
        3:  [15.0, 15.8986, 17.6, 18.4, 19.4, 21.6, 21.0856, 23.2],
        4:  [15.6, 16.8364, 18.5, 19.2, 20.0, 22.4, 21.7859, 23.4],
        5:  [17.0, 18.1507, 19.6, 20.6, 21.2, 23.2, 22.8149, 25.2],
        6:  [18.4, 19.5600, 20.6, 21.2, 21.8, 23.4, 22.9340, 24.2],
        7:  [18.8, 19.7600, 20.8, 21.2, 21.8, 23.0, 22.7807, 24.4],
        8:  [19.0, 20.2523, 21.2, 21.6, 22.2, 23.6, 23.1544, 25.8],
        9:  [18.8, 20.2765, 21.4, 21.8, 22.4, 24.2, 23.5421, 25.6],
        10: [19.2, 20.0319, 21.0, 21.6, 22.2, 23.4, 23.1652, 24.4],
        11: [16.8, 18.4922, 19.8, 20.6, 21.2, 22.8, 22.5767, 25.0],
        12: [16.4, 17.2316, 18.6, 19.4, 20.2, 21.6, 21.4710, 23.0],
    },
}

PANELS = [
    ("pwat",        "Precipitable Water",       "in",    2),
    ("rh850_700",   "850–700 mb RH",            "%",     0),
    ("rh700_500",   "700–500 mb RH",            "%",     0),
    ("thetae850",   "850 mb θe",                "K",     1),
    ("z250",        "250 mb Height",            "m",     0),
    ("thk1000_500", "1000–500 mb Thickness",    "m",     0),
    ("thk1000_850", "1000–850 mb Thickness",    "m",     0),
    ("t500",        "500 mb Temperature",       "°C",    1),
    ("lr850_700",   "850–700 mb Lapse Rate",    "°C/km", 1),
    ("lr700_500",   "700–500 mb Lapse Rate",    "°C/km", 1),
    ("t925",        "925 mb Temperature",       "°C",    1),
    ("wind925",     "925 mb Wind Speed",        "kt",    0),
]
ROW_LABELS = ["MOISTURE", "HEIGHTS & TEMPERATURE", "STABILITY & LOW LEVEL"]


def climo_stats(vals):
    if vals is None:
        return None
    keys8 = ["min", "lo2s", "p25", "p50", "p75", "p975", "hi2s", "max"]
    keys7 = ["min", "lo2s", "p25", "p50", "p75", "hi2s", "max"]
    keys = keys8 if len(vals) == 8 else keys7 if len(vals) == 7 else None
    return dict(zip(keys, vals)) if keys else None


def climo_for(key, month):
    return climo_stats(CLIMO[key].get(month))


# ──────────────────────────────────────────────────────────────
# Data Access — GFS (NOMADS)
# ──────────────────────────────────────────────────────────────
def latest_cycle(lookback: int = 4) -> datetime:
    t = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    last = max(GFS_HOURS)
    for _ in range(lookback):
        url = (f"{NOMADS_PROD}/gfs.{t:%Y%m%d}/{t:%H}/atmos/"
               f"gfs.t{t:%H}z.pgrb2.0p25.f{last:03d}.idx")
        try:
            if requests.head(url, timeout=15).status_code == 200:
                return t
        except requests.RequestException:
            pass
        t -= timedelta(hours=6)
    raise RuntimeError("No complete GFS cycle found on NOMADS in the last 24 h.")


def _filter_url(cycle: datetime, fh: int) -> str:
    hh = f"{cycle:%H}"
    parts = [f"file=gfs.t{hh}z.pgrb2.0p25.f{fh:03d}"]
    parts += [f"lev_{p}_mb=on" for p in LEVELS]
    parts += ["lev_entire_atmosphere_(considered_as_a_single_layer)=on"]
    parts += [f"var_{v}=on" for v in ("HGT", "TMP", "RH", "UGRD", "VGRD", "PWAT")]
    parts += ["subregion=",
              f"leftlon={LON - 0.5}", f"rightlon={LON + 0.5}",
              f"toplat={LAT + 0.5}", f"bottomlat={LAT - 0.5}",
              f"dir=/gfs.{cycle:%Y%m%d}/{hh}/atmos"]
    return f"{NOMADS_FILTER}?{'&'.join(parts)}"


def _fetch_hour(cycle: datetime, fh: int, retries: int = 3):
    url = _filter_url(cycle, fh)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content[:4] == b"GRIB":
                return fh, r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return fh, None


def download(cycle: datetime) -> str:
    t0 = time.time()
    chunks, missing = {}, []
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = [pool.submit(_fetch_hour, cycle, fh) for fh in GFS_HOURS]
        for fut in as_completed(futures):
            fh, data = fut.result()
            if data:
                chunks[fh] = data
            else:
                missing.append(fh)
    if not chunks:
        raise RuntimeError("Every GFS download failed — NOMADS may be down or rate-limiting.")
    with open(GRIB_FILE, "wb") as f:
        for fh in sorted(chunks):
            f.write(chunks[fh])
    print(f"Downloaded {len(chunks)}/{len(GFS_HOURS)} GFS forecast hours in {time.time() - t0:.0f}s")
    if missing:
        print(f"⚠ Missing GFS hours: {sorted(missing)}")
    return GRIB_FILE


def load_point(path: str, cycle: datetime):
    common = {"indexpath": ""}

    def _open(keys):
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={**common, "filter_by_keys": keys})
        ds = ds.sel(latitude=LAT, longitude=LON, method="nearest")
        ds = ds.swap_dims({"step": "valid_time"}) if "step" in ds.dims else ds
        return ds

    prs = _open({"typeOfLevel": "isobaricInhPa"})
    try:
        pwat = _open({"typeOfLevel": "atmosphereSingleLayer", "shortName": "pwat"})["pwat"]
    except Exception as e:
        print(f"⚠ PWAT unavailable ({e}); filling with NaN.")
        pwat = xr.full_like(prs["t"].isel(isobaricInhPa=0), np.nan)

    expected = pd.DatetimeIndex([pd.Timestamp(cycle.replace(tzinfo=None))
                                 + pd.Timedelta(hours=h) for h in GFS_HOURS])
    return prs.reindex(valid_time=expected), pwat.reindex(valid_time=expected)


# ──────────────────────────────────────────────────────────────
# Data Access — ECMWF (Open-Meteo)
# ──────────────────────────────────────────────────────────────
def _ecmwf_vars():
    v  = [f"geopotential_height_{p}hPa" for p in (1000, 850, 700, 500, 250)]
    v += [f"temperature_{p}hPa"         for p in (925, 850, 700, 500)]
    v += [f"relative_humidity_{p}hPa"   for p in (850, 700, 500)]
    v += ["wind_speed_925hPa", "total_column_integrated_water_vapour"]
    return v


def _ecmwf_request(model, run, variables):
    err = "no response"
    for days in (16, 15):
        params = {
            "latitude": LAT, "longitude": LON - 360, "models": model,
            "hourly": ",".join(variables), "run": f"{run:%Y-%m-%dT%H:%M}",
            "forecast_days": days, "wind_speed_unit": "ms", "timezone": "GMT",
        }
        try:
            js = requests.get(ECMWF_API, params=params, timeout=90).json()
        except (requests.RequestException, ValueError) as e:
            err = str(e)
            continue
        if js.get("error"):
            err = js.get("reason", "API error")
            continue
        return js, None
    return None, err


def load_ecmwf(lookback: int = 4):
    now = datetime.now(timezone.utc) - ECMWF_DELAY
    run = now.replace(minute=0, second=0, microsecond=0)
    run -= timedelta(hours=run.hour % 12)
    variables = _ecmwf_vars()
    last_err = "no response"

    for _ in range(lookback):
        for model in ECMWF_MODELS:
            js, err = _ecmwf_request(model, run, variables)
            if js is None:
                last_err = err
                continue
            h = js["hourly"]
            idx = pd.to_datetime(h["time"])
            want = pd.DatetimeIndex([pd.Timestamp(run.replace(tzinfo=None))
                                     + pd.Timedelta(hours=x) for x in EC_HOURS])
            df = pd.DataFrame({k: h[k] for k in variables}, index=idx, dtype=float).reindex(want)
            if df["temperature_500hPa"].iloc[:4].isna().all():
                last_err = f"{model} {run:%HZ %d %b}: no data yet"
                continue

            n = len(want)

            def stack(prefix, conv=lambda x: x):
                out = np.full((n, len(LEVELS)), np.nan)
                for j, p in enumerate(LEVELS):
                    col = f"{prefix}_{p}hPa"
                    if col in df:
                        out[:, j] = conv(df[col].to_numpy())
                return out

            dims = ("valid_time", "isobaricInhPa")
            speed = stack("wind_speed")
            prs = xr.Dataset(
                {"gh": (dims, stack("geopotential_height")),
                 "t":  (dims, stack("temperature", lambda c: c + 273.15)),
                 "r":  (dims, stack("relative_humidity")),
                 "u":  (dims, speed),
                 "v":  (dims, np.zeros_like(speed))},
                coords={"valid_time": want, "isobaricInhPa": LEVELS})
            pwat = xr.DataArray(df["total_column_integrated_water_vapour"].to_numpy(),
                                dims="valid_time", coords={"valid_time": want})
            print(f"ECMWF {run:%Y-%m-%d %HZ} via Open-Meteo ({model}): "
                  f"{int(df.notna().all(axis=1).sum())}/{n} steps")
            return prs, pwat, run
        run -= timedelta(hours=12)
    raise RuntimeError(f"No recent ECMWF run found; last error: {last_err}")


def derive(prs: xr.Dataset, pwat: xr.DataArray) -> dict[str, np.ndarray]:
    lev = lambda name, p: prs[name].sel(isobaricInhPa=p).values.astype(float)
    T  = {p: lev("t", p)  for p in LEVELS}
    Z  = {p: lev("gh", p) for p in LEVELS}
    RH = {p: np.clip(lev("r", p), 1, 100) for p in (850, 700, 500)}

    def lapse(lo, hi):
        return (T[lo] - T[hi]) / ((Z[hi] - Z[lo]) / 1000.0)

    td850 = mpcalc.dewpoint_from_relative_humidity(T[850] * units.K, RH[850] * units.percent)
    thetae = mpcalc.equivalent_potential_temperature(850 * units.hPa, T[850] * units.K, td850)
    wind = mpcalc.wind_speed(lev("u", 925) * units("m/s"), lev("v", 925) * units("m/s"))

    return {
        "pwat":        pwat.values.astype(float) / 25.4,
        "z250":        Z[250],
        "thk1000_500": Z[500] - Z[1000],
        "thk1000_850": Z[850] - Z[1000],
        "rh700_500":   (lev("r", 700) + lev("r", 500)) / 2,
        "lr700_500":   lapse(700, 500),
        "t500":        T[500] - 273.15,
        "wind925":     wind.to("knot").magnitude,
        "rh850_700":   (lev("r", 850) + lev("r", 700)) / 2,
        "lr850_700":   lapse(850, 700),
        "thetae850":   thetae.to("K").magnitude,
        "t925":        T[925] - 273.15,
    }


# ──────────────────────────────────────────────────────────────
# Windowing & Statistics
# ──────────────────────────────────────────────────────────────
def window_series(series, w, t_ref):
    t0 = t_ref + pd.Timedelta(hours=w["start"])
    t1 = t_ref + pd.Timedelta(hours=w["end"])
    out = []
    for s in series:
        t = s["times"]
        m = (t >= t0) & (t <= t1) & (t.hour % w["step"] == 0)
        out.append({**s, "times": t[m], "data": {k: v[m] for k, v in s["data"].items()}})
    return out, t0, t1


def month_segments(t0, t1):
    segs, a = [], t0
    while a < t1:
        nxt = (a.to_period("M") + 1).to_timestamp()
        b = min(nxt, t1)
        segs.append((a, b, a.month))
        a = b
    return segs or [(t0, t1, t0.month)]


def thresholds(key, times, stat):
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        st = climo_for(key, t.month)
        if st:
            out[i] = st[stat]
    return out


# ──────────────────────────────────────────────────────────────
# Plotting Configuration
# ──────────────────────────────────────────────────────────────
C = dict(
    bg="#FFFFFF", panel="#FFFFFF", ink="#1F2933", muted="#6B7785", grid="#E6E8EB",
    normal_band="#ECEEF1", median="#9AA3AD", plume="#9AA3AD", now="#111827",
)
PLUME_ALPHA = 0.30
BAND_ALPHA = 0.10

SERIES_STYLE = {
    "GFS":   dict(short="GFS", color="#1D3FD8", ls="-"),
    "ECMWF": dict(short="EC",  color="#157A33", ls="-"),
}
MODEL_ALPHA = 0.45

CLASSES = ["below −2σ", "−2σ to 25th", "25th–75th", "75th to +2σ", "above +2σ"]
NORMAL = "#3F4650"
PALETTES = {
    "moisture":    ["#8C510A", "#BF812D", NORMAL, "#35978F", "#01665E"],
    "temperature": ["#2166AC", "#4393C3", NORMAL, "#D6604D", "#B2182B"],
    "dynamics":    ["#542788", "#8073AC", NORMAL, "#E08214", "#B35806"],
}
PALETTE_LABELS = {
    "moisture": "Moisture (dry ↔ wet)",
    "temperature": "All other variables (low ↔ high)"
}
VAR_PALETTE = {
    "pwat": "moisture", "rh850_700": "moisture", "rh700_500": "moisture", "thetae850": "moisture",
    "z250": "temperature", "thk1000_500": "temperature", "thk1000_850": "temperature",
    "t500": "temperature", "t925": "temperature",
    "lr850_700": "temperature", "lr700_500": "temperature", "wind925": "temperature",
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#C9CED4", "axes.labelcolor": C["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "figure.facecolor": C["bg"], "savefig.facecolor": C["bg"],
    })


def _badges(ax, rows):
    prev = None
    for txt, col, bold in reversed(rows):
        kw = dict(ha="right", va="bottom", fontsize=8, color=col,
                  fontweight="bold" if bold else "normal", annotation_clip=False)
        if prev is None:
            prev = ax.annotate(txt, xy=(1, 1.03), xycoords="axes fraction", **kw)
        else:
            prev = ax.annotate(txt, xy=(0, 0), xycoords=prev, xytext=(-7, 0),
                               textcoords="offset points", **kw)


def _header_left(ax, parts):
    prev = None
    for txt, col, bold in parts:
        kw = dict(ha="left", va="bottom", fontsize=8, color=col,
                  fontweight="bold" if bold else "normal", annotation_clip=False)
        if prev is None:
            prev = ax.annotate(txt, xy=(0, 1.03), xycoords="axes fraction", **kw)
        else:
            prev = ax.annotate(txt, xy=(1, 0), xycoords=prev, xytext=(6, 0),
                               textcoords="offset points", **kw)


def _on_grid(s, key, idx):
    ser = pd.Series(s["data"][key], index=s["times"]).reindex(idx)
    return ser.interpolate(method="time", limit_area="inside", limit=2).to_numpy()


def classify(key, times, y):
    lo, p25 = thresholds(key, times, "lo2s"), thresholds(key, times, "p25")
    p75, hi = thresholds(key, times, "p75"), thresholds(key, times, "hi2s")
    cls = np.full(len(y), -1)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(y)
        cls[ok] = 2
        cls[ok & (y < p25)] = 1
        cls[ok & (y < lo)] = 0
        cls[ok & (y > p75)] = 3
        cls[ok & (y > hi)] = 4
    return cls


def _colored_line(ax, times, y, key, palette, lw=2.8):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return
    dense = pd.date_range(times[0], times[-1], freq="15min")
    yd = (pd.Series(y, index=times).reindex(times.union(dense))
          .interpolate(method="time", limit_area="inside").reindex(dense).to_numpy(copy=True))
    raw_ok = pd.Series(ok.astype(float), index=times).reindex(times.union(dense)) \
        .interpolate(method="time").reindex(dense).to_numpy() == 1
    yd[~raw_ok] = np.nan
    x = mdates.date2num(dense.to_pydatetime())
    mid_t = dense[:-1] + (dense[1:] - dense[:-1]) / 2
    mid_y = (yd[:-1] + yd[1:]) / 2
    cls = classify(key, mid_t, mid_y)
    segs = np.stack([np.column_stack([x[:-1], yd[:-1]]), np.column_stack([x[1:], yd[1:]])], axis=1)
    keep = cls >= 0
    colors = [palette[c] for c in cls[keep]]
    ax.add_collection(LineCollection(segs[keep], colors="white", linewidths=lw + 2.2,
                                     capstyle="round", zorder=4.4))
    ax.add_collection(LineCollection(segs[keep], colors=colors, linewidths=lw,
                                     capstyle="round", joinstyle="round", zorder=4.5))


L = dict(
    b_out="#F2F2F2", b_lo="#FFF5E6", b_mid="#FFE6E6", b_hi="#E6E6F5",
    hi2s="#339999", lo2s="#996633",
)
LEGACY_PALETTE = ["#996633", "#FF9900", "#CC0000", "#000099", "#339999"]
LEGACY_SERIES = {
    "GFS":   dict(short="GFS", color="#0000FF", ls="-"),
    "ECMWF": dict(short="EC",  color="#1A7F37", ls="-"),
}


def _legacy():
    return COLOR_STYLE == "Legacy"


def _series_style(name):
    return (LEGACY_SERIES if _legacy() else SERIES_STYLE)[name]


def _palette(key):
    return LEGACY_PALETTE if _legacy() else PALETTES[VAR_PALETTE[key]]


def _background(ax, key, palette, t0, t1):
    pool = []
    for k, (a, b, mon) in enumerate(month_segments(t0, t1)):
        st = climo_for(key, mon)
        if not st:
            continue
        lo, hi = st["lo2s"], st["hi2s"]
        if _legacy():
            for y0, y1, col in [(st["min"], lo, L["b_out"]), (lo, st["p25"], L["b_lo"]),
                                (st["p25"], st["p75"], L["b_mid"]), (st["p75"], hi, L["b_hi"]),
                                (hi, st["max"], L["b_out"])]:
                if y0 != y1:
                    ax.fill_between([a, b], min(y0, y1), max(y0, y1), color=col, lw=0, zorder=1)
            ax.hlines(hi, a, b, color=L["hi2s"], lw=1.4, ls="--", zorder=2)
            ax.hlines(lo, a, b, color=L["lo2s"], lw=1.4, ls="--", zorder=2)
        else:
            for y0, y1, col in [(lo, st["p25"], palette[1]), (st["p25"], st["p75"], palette[2]),
                                (st["p75"], hi, palette[3])]:
                if y0 != y1:
                    ax.fill_between([a, b], min(y0, y1), max(y0, y1), color=col,
                                    alpha=BAND_ALPHA, lw=0, zorder=1)
            ax.hlines(hi, a, b, color=palette[4], lw=1.1, ls=(0, (5, 3)), alpha=0.75, zorder=2)
            ax.hlines(lo, a, b, color=palette[0], lw=1.1, ls=(0, (5, 3)), alpha=0.75, zorder=2)
        pool += [lo, hi]
        if k > 0:
            ax.axvline(a, color=C["muted"], lw=0.8, ls=":", zorder=2.2)
            ax.text(a, 0.98, f" {a:%b}", transform=ax.get_xaxis_transform(),
                    fontsize=7, color=C["muted"], va="top", ha="left", zorder=6)
    return pool


def _panel(ax, series, key, title, unit, dec, w, t0, t1, now):
    ax.set_facecolor(C["panel"])
    palette = _palette(key)
    pool = _background(ax, key, palette, t0, t1)

    idx = series[0]["times"]
    for s in series[1:]:
        idx = idx.union(s["times"])
    vals = np.vstack([_on_grid(s, key, idx) for s in series])
    with np.errstate(invalid="ignore"):
        all_ok = np.isfinite(vals).all(axis=0)
        mean = np.where(all_ok, np.nanmean(vals, axis=0), np.nan)
    spread = None
    if len(series) == 2:
        yg, ye = vals
        ax.fill_between(idx, yg, ye, where=all_ok, interpolate=True, lw=0,
                        color=C["plume"], alpha=PLUME_ALPHA, zorder=2.5)
        if all_ok.any():
            diff = np.where(all_ok, yg - ye, 0.0)
            i = int(np.argmax(np.abs(diff)))
            leader = series[0] if diff[i] >= 0 else series[1]
            spread = (abs(diff[i]), idx[i], leader)

    rows = []
    for s in series:
        y = s["data"][key]
        st = _series_style(s["name"])
        if len(series) > 1:
            ax.plot(s["times"], y, color=st["color"], ls=st["ls"], lw=1.0, alpha=MODEL_ALPHA,
                    zorder=3)
        finite = y[np.isfinite(y)]
        pool += list(finite)
        first = f"{finite[0]:.{dec}f}" if finite.size else "—"
        rows.append((f"{st['short']} {first}", st["color"], False))

    _colored_line(ax, idx, mean, key, palette)
    cls = classify(key, idx, mean)
    n_hi, n_lo = int(np.sum(cls == 4)), int(np.sum(cls == 0))
    if n_hi:
        rows.append((f"▲{n_hi}", palette[4], True))
    if n_lo:
        rows.append((f"▼{n_lo}", palette[0], True))

    if t0 <= now <= t1:
        ax.axvline(now, color=C["now"], lw=1.0, ls=(0, (2, 2)), alpha=0.7, zorder=3.5)

    if pool:
        lo_, hi_ = np.nanmin(pool), np.nanmax(pool)
        pad = (hi_ - lo_) * 0.10 or 1
        ax.set_ylim(lo_ - pad, hi_ + pad)

    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=C["ink"], pad=17)
    left = [(unit, C["muted"], False)]
    if spread:
        val, t, leader = spread
        left.append((f"max Δ {val:.{dec}f} {_series_style(leader['name'])['short']}↑ "
                     f"{t:%a %HZ}", C["muted"], False))
    _header_left(ax, left)
    _badges(ax, rows)
    _axes_common(ax, dec, w, t0, t1)


def _axes_common(ax, dec, w, t0, t1):
    ax.grid(axis="y", color=C["grid"], lw=0.6, zorder=0)
    ax.grid(axis="x", which="major", color=C["grid"], lw=0.6, zorder=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{dec}f"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18] if w["step"] == 3 else [12]))
    ax.set_xlim(t0, t1)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.tick_params(axis="x", which="minor", length=2.5, color="#B8BEC6")
    ax.tick_params(axis="x", which="major", length=4, color="#B8BEC6", labelbottom=True)


def _class_key(fig, x0, y_top, rows, lab_w=0.175):
    sw, sh, gap, row_h = 0.045, 0.011, 0.004, 0.019
    for j, name in enumerate(CLASSES):
        fig.text(x0 + lab_w + j * (sw + gap) + sw / 2, y_top + 0.004, name,
                 ha="center", va="bottom", fontsize=8, color=C["muted"])
    for r, (label, pal) in enumerate(rows):
        y = y_top - (r + 1) * row_h
        fig.text(x0 + lab_w - 0.006, y + sh / 2, label, ha="right", va="center",
                 fontsize=8.5, color=C["ink"])
        for j, col in enumerate(pal):
            fig.add_artist(Rectangle((x0 + lab_w + j * (sw + gap), y), sw, sh,
                                     transform=fig.transFigure, fc=col, ec="none"))


def plot(series, w, t0, t1, now, note=None):
    _style()
    legacy = _legacy()
    fig, axes = plt.subplots(3, 4, figsize=(20, 13.5), sharex=False)
    fig.subplots_adjust(left=0.058, right=0.985, top=0.855 if note else 0.87, bottom=0.145,
                        hspace=0.58, wspace=0.18)
    for ax, (key, title, unit, dec) in zip(axes.flat, PANELS):
        _panel(ax, series, key, title, unit, dec, w, t0, t1, now)

    for r, label in enumerate(ROW_LABELS):
        box = axes[r, 0].get_position()
        y0, y1 = box.y0, box.y1 + 0.03
        fig.add_artist(Line2D([0.016, 0.016], [y0, y1], transform=fig.transFigure,
                              color=C["muted"], lw=2.2, alpha=0.55, solid_capstyle="round"))
        fig.text(0.008, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=C["muted"])

    runs = "  &  ".join(f"{s['name']} {s['cycle']:%HZ %d %b}" for s in series)
    fig.text(0.058, 0.957, f"Sounding Climatology vs. Models — {w['name']} ({w['span']})",
             fontsize=19, fontweight="bold", color=C["ink"])

    gen = datetime.now(LOCAL_TZ).strftime("%a %d %b %Y %H:%M AST")
    months = month_segments(t0, t1)
    climo_note = ("climatology: " + " → ".join(f"{pd.Timestamp(2000, m, 1):%b}" for *_, m in months)
                  if len(months) > 1 else f"climatology: {t0:%B}")
    fig.text(0.058, 0.930,
             f"{SITE_NAME}  ·  {LAT:.2f}°N {360 - LON:.2f}°W  ·  runs: {runs}  ·  "
             f"valid {t0:%a %d %b %HZ} → {t1:%a %d %b %HZ} (UTC), {w['step']}-hourly  ·  "
             f"{climo_note}",
             fontsize=10.5, color=C["muted"])
    fig.text(0.058, 0.912, f"generated {gen}", fontsize=9, color=C["muted"])
    if note:
        fig.text(0.058, 0.893, f"⚠ {note}", fontsize=10, color="#B45309")
    mean_lbl = "model mean" if len(series) > 1 else f"{series[0]['name']}"
    fig.text(0.985, 0.912,
             f"thick line = {mean_lbl}, colored by climatological class  ·  "
             + ("max Δ = largest GFS–ECMWF gap  ·  " if len(series) > 1 else "")
             + "▲/▼ = steps beyond ±2σ",
             ha="right", fontsize=9, color=C["muted"])
    if any(s["name"] == "ECMWF" for s in series):
        fig.text(0.985, 0.004, "ECMWF open data (CC-BY-4.0) via Open-Meteo  ·  GFS via NOMADS",
                 ha="right", va="bottom", fontsize=8, color=C["muted"])

    mean_color = LEGACY_PALETTE[2] if legacy else NORMAL
    handles = [Line2D([], [], color=mean_color, lw=2.8,
                      label="Model mean" if len(series) > 1 else f"{series[0]['name']} forecast")]
    for s in (series if len(series) > 1 else []):
        st = _series_style(s["name"])
        handles.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=1.2, alpha=0.7,
                              label=s["name"]))
    if len(series) == 2:
        handles.append(Patch(fc=C["plume"], alpha=PLUME_ALPHA, lw=0, label="GFS–ECMWF spread"))
    if legacy:
        handles += [
            Patch(fc=L["b_lo"],  ec="#E8D6BC", lw=0.6, label="Climo −2σ to 25th"),
            Patch(fc=L["b_mid"], ec="#EBC4C4", lw=0.6, label="Climo 25th–75th"),
            Patch(fc=L["b_hi"],  ec="#C4C4E0", lw=0.6, label="Climo 75th to +2σ"),
            Patch(fc=L["b_out"], ec="#D9D9D9", lw=0.6, label="Beyond ±2σ to record"),
            Line2D([], [], color=L["hi2s"], lw=1.4, ls="--", label="+2σ"),
            Line2D([], [], color=L["lo2s"], lw=1.4, ls="--", label="−2σ"),
        ]
        key_rows = [("All variables", LEGACY_PALETTE)]
    else:
        handles += [
            Patch(fc=NORMAL, alpha=BAND_ALPHA, lw=0,
                  label="Climo bands: class colors at 10%"),
            Line2D([], [], color=C["muted"], lw=1.1, ls=(0, (5, 3)), label="±2σ (class colors)"),
        ]
        key_rows = [(PALETTE_LABELS[k], PALETTES[k]) for k in PALETTE_LABELS]
    if t0 <= now <= t1:
        handles.append(Line2D([], [], color=C["now"], lw=1.0, ls=(0, (2, 2)), label="Now"))
    fig.legend(handles=handles, loc="lower left", ncol=int(np.ceil(len(handles) / 2)),
               frameon=False, fontsize=10 if not legacy else 9.5, bbox_to_anchor=(0.05, 0.022),
               handlelength=2.2, columnspacing=1.5)
    if legacy:
        _class_key(fig, 0.66, 0.088, key_rows, lab_w=0.07)
    else:
        _class_key(fig, 0.55, 0.088, key_rows)

    os.makedirs(os.path.dirname(w["out"]), exist_ok=True)
    fig.savefig(w["out"], dpi=150)
    print(f"Successfully generated and saved graphic to: {w['out']}")
    return fig


# ──────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────
def _gfs_series():
    cycle = latest_cycle()
    print(f"Using GFS {cycle:%Y-%m-%d %HZ}")
    prs, pwat = load_point(download(cycle), cycle)
    return dict(name="GFS", times=pd.DatetimeIndex(prs["valid_time"].values),
                data=derive(prs, pwat), cycle=cycle)


def _ecmwf_series():
    prs, pwat, run = load_ecmwf()
    return dict(name="ECMWF", times=pd.DatetimeIndex(prs["valid_time"].values),
                data=derive(prs, pwat), cycle=run)


def fetch_all():
    want_gfs = MODEL in ("GFS", "Both")
    want_ec  = MODEL in ("ECMWF", "Both")
    series, note = [], None

    if want_ec:
        try:
            series.append(_ecmwf_series())
        except Exception as e:
            note = f"ECMWF unavailable ({e}) — showing GFS only"
            print(f"⚠ {note}")
            want_gfs = True

    if want_gfs:
        try:
            series.insert(0, _gfs_series())
        except Exception as e:
            if not series:
                raise
            note = f"GFS unavailable ({e}) — showing ECMWF only"
            print(f"⚠ {note}")
    return series, note


def main():
    series, note = fetch_all()
    t_ref = pd.Timestamp(max(s["cycle"] for s in series).replace(tzinfo=None))
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    sub, t0, t1 = window_series(series, WINDOW, t_ref)
    fig = plot(sub, WINDOW, t0, t1, now, note)
    plt.close(fig)


if __name__ == "__main__":
    main()
