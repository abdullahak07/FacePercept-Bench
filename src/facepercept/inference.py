def parse_binary_prediction(text):
    text = text.strip().upper()

    if text.startswith("REAL"):
        return "real"
    if text.startswith(("SYNTHETIC", "FAKE", "AI")):
        return "synthetic"

    return None
