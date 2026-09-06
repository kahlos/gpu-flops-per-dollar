"""One-off maintenance: recompress all zstd blobs at the current level.

Usage: .venv/bin/python -m dbtools.maint [--db gpu.db]
Safe: verifies byte-identical round-trip per blob before replacing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zstandard as zstd  # noqa: E402
from dbtools.store import ZSTD_LEVEL  # noqa: E402

_TABLES = ["listing_blobs:data", "raw_pages:html", "history:data"]


def recompress_db(path: Path, level: int = ZSTD_LEVEL) -> dict:
    from dbtools.store import connect
    con = connect(path)  # also applies pending schema migrations
    c = zstd.ZstdCompressor(level=level)
    d = zstd.ZstdDecompressor()
    stats = {"blobs": 0, "before": 0, "after": 0}
    for spec in _TABLES:
        table, col = spec.split(":")
        pks = {"listing_blobs": ["gpu_id", "cond"], "raw_pages": ["gpu_id"],
               "history": ["ts", "gpu_id"]}[table]
        for row in con.execute(f"SELECT {','.join(pks)}, {col} FROM {table}"):
            *key, blob = row
            raw = d.decompress(blob)
            new = c.compress(raw)
            assert d.decompress(new) == raw, f"round-trip failed for {key}"
            if len(new) != len(blob):
                con.execute(
                    f"UPDATE {table} SET {col}=? WHERE " +
                    " AND ".join(f"{k}=?" for k in pks), [new, *key])
            stats["blobs"] += 1
            stats["before"] += len(blob)
            stats["after"] += len(new) if len(new) != len(blob) else len(blob)
    con.commit()
    con.execute("VACUUM")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.execute("PRAGMA journal_mode=DELETE")
    con.commit()
    con.close()
    stats["saved"] = stats["before"] - stats["after"]
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "gpu.db"))
    args = ap.parse_args()
    st = recompress_db(Path(args.db))
    print(f"{st['blobs']} blobs: {st['before'] // 1024}KB -> {st['after'] // 1024}KB "
          f"(saved {st['saved'] // 1024}KB), VACUUMed")
