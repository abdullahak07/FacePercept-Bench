from __future__ import annotations


def parse_binary_prediction(text: str):
    t = text.strip().upper()
    if t.startswith("REAL"):
        return "real"
    if t.startswith("SYNTHETIC") or t.startswith("FAKE") or t.startswith("AI"):
        return "synthetic"
    return None
