"""CDS (Copernicus) downloads with a time budget.

The CDS queue can hold a request for most of an hour (2026-10-25: a month of ERA5 dew sat
there 30+ min and took the job past its timeout). Every caller has a fallback - a cached
month, or the ECMWF archive - so after BUDGET seconds we give up and let it use that.
"""
import signal
import time

BUDGET = 6 * 60


class _Timeout(BaseException):  # not Exception: cdsapi's own retry loops must not swallow it
    pass


def retrieve(dataset, req, target, label):
    """Download `dataset` to `target` (via a .part file, replacing an older target)."""
    import cdsapi

    def alarm(*_):
        raise _Timeout()

    tmp = target.with_suffix(".part")
    old = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(BUDGET)
    err = None
    try:
        for k in range(4):
            try:
                cdsapi.Client(quiet=True, progress=False).retrieve(dataset, req, str(tmp))
                tmp.rename(target)
                return
            except Exception as e:  # CDS queues fail now and then
                err = e
                time.sleep(30 * (k + 1))
    except _Timeout:
        err = f"no result within {BUDGET // 60} min (CDS queue)"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{label} failed: {err}")
