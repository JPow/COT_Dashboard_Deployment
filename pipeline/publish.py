#!/usr/bin/env python3
"""
After running COT IBRK Data Grabber locally, verify caches and print git publish steps.

Render (and other hosts) read ``data/cot_data.json`` from the deployed branch — there is
no GitHub cron. Run this script before committing data updates.
"""

import os
import sys

from pipeline.paths import (
    COT_DATA_JSON,
    IB_DAILY_CACHE,
    ORB_INTRADAY_CACHE,
    DATA_DIR,
)


def main():
    missing = []
    for label, path in [
        ("cot_data.json", COT_DATA_JSON),
        ("ib_daily_cache.json", IB_DAILY_CACHE),
        ("ORB_intraday_data.json", ORB_INTRADAY_CACHE),
    ]:
        if not os.path.isfile(path):
            missing.append(f"  - {label} expected at {path}")
    if missing:
        print("Missing cache files (run COT IBRK Data Grabber.ipynb with IB Gateway):", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        sys.exit(1)

    print(f"Data directory: {DATA_DIR}")
    print("Ready to publish. Typical workflow:")
    print("  git add data/cot_data.json")
    print("  git add data/ib_daily_cache.json data/ib_daily_roll_state.json   # if daily rebuild")
    print("  git add data/ORB_intraday_data.json data/ORB_intraday_roll_state.json  # if intraday updated")
    print('  git commit -m "Update COT/IB caches from Grabber"')
    print("  git push origin <deploy-branch>")


if __name__ == "__main__":
    main()
