# Gljivar daily pipeline

Three times a day (GitHub Actions, see the cron comment in the workflow) this job stitches
observed and forecast weather, scores four mushroom groups for today + 7 days, and publishes:

| Grid | Countries | Output | Published to | For |
|---|---|---|---|---|
| `hrsi1` (default) | HR, SI | `out/` | `.../gljivar/data/` | 0.4.x apps (never change this one) |
| `eu9v1` (`run_daily.py --v2`) | HR SI IT AT DE CH RS BA HU | `out/v2/` (+ `rain-<date>.bin`) | `.../gljivar/data/v2/` | 0.5.0+ apps, contract in `../docs/PLAN-0.5.0.md` |

The grid is picked by `GLJIVAR_GRID` in `config.py`; each has its own static grid, ERA5 caches and
archive (`archive/<tag>/`). v2 runs after v1 is published and may fail without touching v1.

| File | Role |
|---|---|
| `run_daily.py` | the job: ERA5-Land + IMERG + ECMWF/ICON-EU → `out/forecast-<date>.bin`, `latest.json`, `grid.json` |
| `era5land.py`, `imerg.py`, `forecast.py` | data sources |
| `model.py`, `out/model.json` | features and fitted weights (fitted locally by `export_model.py` from `../backtest`) |
| `preview.py` | PNG of all groups × days; `--map vrganj`, `--rain 7` one big map (`--v2` for the v2 grid) |
| `publish.sh` | FTPS upload + HTTP check (`./publish.sh v2` for out/v2) |
| `notify.py` | Telegram message (failures) |
| `archive/` | per-day inputs ERA5-Land does not cover yet; committed by the job, do not delete |
| `static/` | grids (`grid.npz` = hrsi1, `grid_eu9v1.npz`: cells incl. coast, borrowed weather cells, forest share, mediterranean zone), season multipliers, elevation, borders, IMERG monthly scale factors (v1 only) |

Local-only (need `../backtest` data, plus rasterio/shapely/pyshp for the grid): `export_model.py`, `grid.py` (build; `GLJIVAR_GRID=eu9v1 python grid.py` for v2), `calib_imerg.py`, `imerg_factors.py`, `replay.py`.
Locally everything runs with `../backtest/.venv/bin/python`.

Secrets: `CDSAPI_KEY`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `FTP_USER`, `FTP_PASS`,
`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

Data: Contains modified Copernicus Climate Change Service information (ERA5-Land, ERA5); NASA GPM IMERG
(v1 only); DWD ICON-EU (CC BY 4.0); ECMWF open data (CC BY 4.0); ESA WorldCover 2021 (CC BY 4.0);
Natural Earth.
