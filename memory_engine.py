# memory_engine.py
# ===============================
# Memory Orchestration Engine
# ===============================

import os
import json
from datetime import datetime

MEMORY_DIR = "memory"


def ensure_memory_dir():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)


def memory_path(stock):
    return os.path.join(MEMORY_DIR, f"{stock}.json")


def load_memory(stock):
    """
    Load all memory entries for a stock.
    """
    return []


def save_memory(stock, memory):
    """
    Persist memory to disk.
    """
    pass


def write_snapshot(snapshot):
    """
    Append a single snapshot to memory.
    """
    pass


def ensure_memory_up_to_date(stock):
    """
    MASTER ENTRY POINT

    Logic later:
    - If no memory → trigger backfill
    - If memory exists but outdated → extend
    - If memory current → do nothing
    """
    pass
