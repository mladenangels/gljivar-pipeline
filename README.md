# Gljivar daily pipeline

Every day at 06:17 UTC (GitHub Actions) this job stitches observed and forecast weather for
Croatia, scores four mushroom groups for today + 7 days, and publishes the result to
`https://designer2.org/designer2/gljivar/data/`.

| File | Role |
|---|---|
| `run_daily.py` | the job: ERA5-Land + IMERG + ECMWF/ICON-EU → `out/forecast-<date>.bin`, `latest.json`, `grid.json` |
| `era5land.py`, `imerg.py`, `forecast.py` | data sources |
| `model.py`, `out/model.json` | features and fitted weights (fitted locally by `export_model.py` from `../backtest`) |
| `preview.py` | PNG of all groups × days |
| `publish.sh` | FTPS upload + HTTP check |
| `notify.py` | Telegram message (failures) |
| `archive/` | per-day inputs ERA5-Land does not cover yet; committed by the job, do not delete |
| `static/` | grid (Croatian cells incl. coast, borrowed weather cells, forest share, mediterranean zone), season multipliers, elevation, borders, IMERG monthly scale factors |

Local-only (need `../backtest` data, plus rasterio/shapely/pyshp for the grid): `export_model.py`, `grid.py` (build), `calib_imerg.py`, `imerg_factors.py`, `replay.py`.

Secrets: `CDSAPI_KEY`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `FTP_USER`, `FTP_PASS`,
`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

Data: Contains modified Copernicus Climate Change Service information (ERA5-Land); NASA GPM IMERG;
DWD ICON-EU (CC BY 4.0); ECMWF open data (CC BY 4.0).
