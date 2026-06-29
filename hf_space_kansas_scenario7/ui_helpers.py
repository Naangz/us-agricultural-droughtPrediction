"""Pure helpers for the novice wizard UI."""

LEVEL_ORDER = {"None": 0, "D0": 1, "D1": 2, "D2": 3, "D3": 4, "D4": 5}


def build_input_features(last_week_level, last_week_pct, two_weeks_level, two_weeks_pct):
    """Map wizard answers to the 12 cumulative drought model features."""

    def encode(level, pct):
        severity = LEVEL_ORDER[level]
        coverage = float(pct)
        return [
            100.0 - coverage,
            coverage,
            coverage if severity >= 1 else 0.0,
            coverage if severity >= 2 else 0.0,
            coverage if severity >= 3 else 0.0,
            coverage if severity >= 4 else 0.0,
        ]

    return encode(last_week_level, last_week_pct) + encode(two_weeks_level, two_weeks_pct)


def describe_coverage(percent):
    """Return a plain-language Indonesian summary for a coverage percentage."""

    coverage = float(percent)
    if coverage <= 10:
        return "hanya sebagian kecil wilayah"
    if coverage <= 35:
        return "sebagian wilayah"
    if coverage <= 65:
        return "sekitar setengah wilayah"
    if coverage <= 90:
        return "sebagian besar wilayah"
    return "hampir seluruh wilayah"


def build_review_note(last_week_level, last_week_pct, two_weeks_level, two_weeks_pct):
    """Highlight unusual combinations without blocking the user."""

    if last_week_level == "None" and float(last_week_pct) >= 70:
        return "Periksa kembali: Anda memilih kondisi normal tetapi area terdampak sangat luas."
    if two_weeks_level == "None" and float(two_weeks_pct) >= 70:
        return "Periksa kembali: Dua minggu lalu dipilih normal, tetapi area terdampak sangat luas."
    return ""
