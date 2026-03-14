# utils/test_counter.py
import os
import json

COUNTER_FILE = "test_counter.json"

def load_counter():
    if not os.path.exists(COUNTER_FILE):
        return 0
    with open(COUNTER_FILE, "r") as f:
        return json.load(f).get("count", 0)

def save_counter(count):
    with open(COUNTER_FILE, "w") as f:
        json.dump({"count": count}, f)
