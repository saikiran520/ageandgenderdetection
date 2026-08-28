#!/usr/bin/env python3
"""
Fetch a small set of public-domain portrait photographs for calibration and testing.

These images are ONLY used on the developer machine, to
  1. measure how YuNet's face boxes relate to MiVOLO's own detector
     (scripts/calibrate_face_box.py), and
  2. give scripts/test_mivolo.py something to run against.

They are never shipped in the APK, with the single exception of the one image
copied to app/src/main/assets/test.jpg by scripts/pack_assets.py, which the app's
self-test uses to prove Python and Android agree.

Source : Wikimedia Commons, "Official portraits of members of the 118th United
         States Congress". Works of the US federal government -- public domain.
Usage  : python scripts/fetch_testdata.py --count 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "testdata"

API = "https://commons.wikimedia.org/w/api.php"
CATEGORY = "Category:Official portraits of members of the 118th United States Congress"
USER_AGENT = "MiVOLO-Android-POC/1.0 (offline age-gender proof of concept; local research use)"


def safe_name(title: str) -> str:
    name = title.split(":", 1)[-1]
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return keep[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--width", type=int, default=900, help="Thumbnail width to request.")
    args = parser.parse_args()

    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    collected: list[dict] = []
    continue_token: dict = {}

    while len(collected) < args.count:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": CATEGORY,
            "gcmlimit": "50",
            "gcmtype": "file",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": str(args.width),
            "format": "json",
            **continue_token,
        }
        response = session.get(API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        pages = (payload.get("query") or {}).get("pages") or {}
        if not pages:
            break

        for page in pages.values():
            if len(collected) >= args.count:
                break
            info = page["imageinfo"][0]
            license_name = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            if "public domain" not in license_name.lower():
                continue
            url = info.get("thumburl") or info["url"]
            dest = TESTDATA_DIR / safe_name(page["title"])
            if dest.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                dest = dest.with_suffix(".jpg")
            if not dest.exists():
                # Commons rate-limits aggressively; back off rather than hammer it.
                for attempt in range(5):
                    image = session.get(url, timeout=60)
                    if image.status_code == 429:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    image.raise_for_status()
                    break
                else:
                    print(f"  (skipped {dest.name}: rate limited)")
                    continue
                dest.write_bytes(image.content)
                time.sleep(0.4)
            collected.append(
                {
                    "file": dest.name,
                    "title": page["title"],
                    "license": license_name,
                    "descriptionurl": info.get("descriptionurl", ""),
                }
            )
            print(f"  {dest.name}")

        continue_token = payload.get("continue") or {}
        if not continue_token:
            break

    (TESTDATA_DIR / "SOURCES.json").write_text(json.dumps(collected, indent=2), encoding="utf-8")
    print(f"\n{len(collected)} public-domain portraits in {TESTDATA_DIR}")
    print("Provenance recorded in testdata/SOURCES.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
