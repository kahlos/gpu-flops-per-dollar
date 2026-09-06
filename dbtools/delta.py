"""Minimal JSON diff/apply for history deltas.

Structures are dicts-all-the-way-down except lists, which are replaced
wholesale (monthly buckets etc. change as units). Ops are deterministic
(sorted by path) so identical transitions hash identically.
"""

from __future__ import annotations

import copy


def diff(old, new) -> list[dict]:
    return sorted(_walk(old, new, []), key=lambda o: (len(o["path"]), o["path"], o["op"]))


def _walk(o, n, path: list) -> list[dict]:
    if isinstance(o, dict) and isinstance(n, dict):
        ops: list[dict] = []
        for k in o:
            if k not in n:
                ops.append({"op": "del", "path": [*path, k]})
            else:
                ops.extend(_walk(o[k], n[k], [*path, k]))
        for k in n:
            if k not in o:
                ops.append({"op": "set", "path": [*path, k], "value": n[k]})
        return ops
    if o != n:
        return [{"op": "set", "path": list(path), "value": n}]
    return []


def apply(base, ops: list[dict]):
    doc = copy.deepcopy(base)
    for op in sorted(ops, key=lambda o: (len(o["path"]), o["path"])):
        if not op["path"]:
            doc = op["value"] if op["op"] == "set" else doc
            continue
        node = doc
        for k in op["path"][:-1]:
            nxt = node.get(k) if isinstance(node, dict) else None
            if not isinstance(nxt, dict):
                nxt = {}
                if isinstance(node, dict):
                    node[k] = nxt
            node = nxt
        if not isinstance(node, dict):
            continue
        if op["op"] == "del":
            node.pop(op["path"][-1], None)  # tolerant: missing key is fine
        else:
            node[op["path"][-1]] = op["value"]
    return doc
