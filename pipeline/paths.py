"""Canonical paths for Grabber outputs and IB caches."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("COT_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

COT_DATA_JSON = os.path.join(DATA_DIR, "cot_data.json")
IB_DAILY_CACHE = os.path.join(DATA_DIR, "ib_daily_cache.json")
IB_DAILY_ROLL_STATE = os.path.join(DATA_DIR, "ib_daily_roll_state.json")
ORB_INTRADAY_CACHE = os.path.join(DATA_DIR, "ORB_intraday_data.json")
ORB_INTRADAY_STATE = os.path.join(DATA_DIR, "ORB_intraday_roll_state.json")
ORB_CONTRACT_SPECS = os.path.join(DATA_DIR, "ORB_contract_specs.json")
