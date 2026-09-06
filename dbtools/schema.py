"""Bundle packing scheme (GET /api/bundle): short literal keys, pooled string
indices, enum indices. The two implementations that must agree are
dbtools/pack.py (encoder) and website/dbloader.js (decoder); node shape
checks verify them together. See docs/SCHEMA.md.
"""

ENUMS = {
    "tpu_method": ["live_headed", "live_fetch", "curated", "sibling", "missing"],
    "price_flag": ["ok", "thin", "missing"],
    "tpu_status": ["ok", "missing"],
}

WINDOWS = ("7d", "30d", "90d", "365d")
MIN_30D_SAMPLES = 10
