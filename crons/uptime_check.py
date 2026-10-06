#!/usr/bin/env python3
"""Uptime + smoke check for the two carbon-capture Streamlit deployments.

For each service:
  1. GET / with a timeout            -> expect HTTP 200
  2. Body must contain a Streamlit shell marker (case-insensitive 'streamlit')
  3. GET /_stcore/health             -> expect HTTP 200 with body 'ok'

One retry (after a short wait) on connection/timeout failures guards against
Railway cold-start flapping. No retry on non-200 or missing markers.

State is kept in ../hidden_files/uptime_state.json so the caller can tell a
fresh DOWN or RECOVERED transition apart from a steady state.

Output: one human-readable line per service, then one machine-readable
EVENT line per service:
    EVENT service=<name> transition=<down|recovered|none> detail=<symptom> url=<url>
Exit code: 0 when every service is up, 1 when any service is down.
On failure the human-readable line names the failing URL and the symptom
(dns/timeout/refused, non-200, missing marker, unhealthy).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.normpath(os.path.join(BASE, "..", "hidden_files", "uptime_state.json"))

SERVICES = [
    ("v3-staging", "https://carbon-capture-app-v3-staging-production.up.railway.app"),
    ("v2-production", "https://carbon-capture-app-v2-production.up.railway.app"),
]

PAGE_TIMEOUT = 20.0
HEALTH_TIMEOUT = 15.0
RETRY_WAIT = 45.0


def _short_err(e):
    s = str(e).strip().replace("\n", " ")
    if len(s) > 90:
        s = s[:87] + "..."
    return s or e.__class__.__name__


def fetch(url, timeout):
    """Return (status, body, latency_s). Raises on connection/timeout errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "ccs-uptime-check/1.0"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, time.monotonic() - start
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return e.code, body, time.monotonic() - start


def check_service(name, url):
    """Return (status, detail) where status is 'up' or 'down'."""
    last_err = None
    for attempt in (1, 2):
        try:
            code, body, latency = fetch(url, PAGE_TIMEOUT)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = "connection failed (%s)" % _short_err(e)
            if attempt == 1:
                time.sleep(RETRY_WAIT)
                continue
            return "down", "%s url=%s" % (last_err, url)
        except Exception as e:  # defensive: never crash the whole check
            return "down", "unexpected error (%s) url=%s" % (_short_err(e), url)

        if code != 200:
            return "down", "HTTP %s on / (expected 200) url=%s" % (code, url)
        if "streamlit" not in body.lower():
            return "down", "HTTP 200 but Streamlit shell marker missing url=%s" % url

        try:
            hcode, hbody, _ = fetch(url.rstrip("/") + "/_stcore/health", HEALTH_TIMEOUT)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return "down", "health endpoint unreachable (%s) url=%s" % (_short_err(e), url)
        except Exception as e:
            return "down", "health endpoint error (%s) url=%s" % (_short_err(e), url)

        if hcode != 200 or "ok" not in hbody.lower():
            return "down", "health endpoint unhealthy (HTTP %s) url=%s" % (hcode, url)

        return "up", "ok in %.2fs" % latency

    return "down", "%s url=%s" % (last_err or "connection failed", url)


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass  # state is best-effort; the check result still stands


def main():
    prev = load_state()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    new_state = dict(prev)
    any_down = False

    for name, url in SERVICES:
        status, detail = check_service(name, url)
        entry = prev.get(name)
        prev_status = entry.get("status", "unknown") if isinstance(entry, dict) else "unknown"

        if status == "up" and prev_status == "down":
            transition = "recovered"
        elif status == "down" and prev_status != "down":
            transition = "down"
        else:
            transition = "none"

        new_state[name] = {"status": status, "checked_at": now, "detail": detail}
        save_state(new_state)

        mark = "UP  " if status == "up" else "DOWN"
        print("%s: %s - %s" % (name, mark, detail))
        print("EVENT service=%s transition=%s url=%s" % (name, transition, url))

        if status == "down":
            any_down = True

    return 1 if any_down else 0


if __name__ == "__main__":
    sys.exit(main())
