# utils.py

SEVERITY_THRESHOLDS = {
    "Low": (1, 5),
    "Moderate": (6, 15),
    "Critical": (16, 1_000_000)
}

# More detailed trash categories
CLASS_GROUPS = {
    "Plastic": {"trash_plastic", "trash_rubber", "trash_fabric"},
    "Metal": {"trash_metal"},
    "Paper": {"trash_paper"},
    "Wood": {"trash_wood"},
    "Fishing Gear": {"trash_fishing_gear"},
    "Miscellaneous": {"trash_etc"}
}

def severity_from_count(total_count: int) -> str:
    if total_count <= 0:
        return "None"
    for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= total_count <= hi:
            return level
    return "Critical"

def bucket_class(class_name: str) -> str:
    for bucket, names in CLASS_GROUPS.items():
        if class_name in names:
            return bucket
    return None  # ignore non-trash classes like animals/plants/rov
