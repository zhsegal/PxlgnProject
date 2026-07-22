"""Parse triplet_candidate_list.md -> closed list of marker-index triples + metadata.

Output: results/chunglu/triplet_candidate_list.json
    {"triples": [[a,b,c], ...],            # sorted distinct global marker indices
     "meta":    [{number, module, hub, sign, cell, q, names:[A,B,C]}, ...],
     "contrasts": [{number, triplet, contrast, expectation}, ...],  # section F, not new triples
     "marker_names": [...]}

The wedge statistic is symmetric over the three markers, so triples are stored sorted; the
hub/sign/cell/module fields are biological metadata only. Drops: self-pair rows (a marker
repeats -> not a 3-distinct wedge), any triple with a token absent from marker_names (CD63),
and the cross-cutting CONTRAST rows (section F) which reuse triplets already defined above.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD = BASE / "triplet_candidate_list.md"
MARKERS = BASE / "results" / "chunglu" / "marker_names.json"
OUT = BASE / "results" / "chunglu" / "triplet_candidate_list.json"

# named/alias tokens -> canonical marker (the doc mostly uses CD-numbers already; safety net)
ALIAS = {
    "PSGL-1": "CD162", "LAIR-1": "CD305", "ICAM-2": "CD102", "ICAM-3": "CD50",
    "LFA-3": "CD58", "CEACAM8": "CD66b", "OX40": "CD134", "4-1BB": "CD137",
    "PD-L1": "CD274", "PD-L2": "CD273", "PD-1": "CD279", "TIM-3": "CD366",
    "GITR": "CD357", "ICOS": "CD278", "CD40L": "CD154", "CTLA-4": "CD152",
    "NKG2D": "CD314", "DNAM-1": "CD226", "2B4": "CD244", "BAFF-R": "CD268",
    "BCMA": "CD269", "SLAMF6": "CD150", "SLAMF1": "CD150",
}
EN_DASH = "–"


def resolve(tok, valid):
    tok = tok.strip().strip("*").strip()
    tok = ALIAS.get(tok, tok)
    return tok if tok in valid else None


def main():
    marker_names = json.loads(MARKERS.read_text())
    idx = {m: i for i, m in enumerate(marker_names)}
    valid = set(marker_names)
    lines = MD.read_text().splitlines()

    module = None
    in_contrasts = False
    in_polar = False
    triples, meta, contrasts = [], [], []
    seen = set()
    n_rows = n_selfpair = n_unresolved = 0
    unresolved_tokens = set()

    for ln in lines:
        s = ln.strip()
        if s.startswith("### Module"):
            module = s.lstrip("# ").strip()
            in_polar = False
            continue
        if s.startswith("## F.") or "CROSS-CUTTING CONTRASTS" in s:
            in_contrasts = True
            continue
        if s.startswith("### Optional polarization"):
            in_polar = True
            continue
        if s.startswith("## ") and not s.startswith("## F."):
            in_contrasts = False
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue
        num = cells[0]
        if not num.isdigit():            # header / separator row
            continue
        trip_txt = cells[1]

        if in_contrasts:                 # section F: reuse, record but don't add
            contrasts.append({"number": int(num), "triplet": trip_txt,
                              "contrast": cells[2] if len(cells) > 2 else "",
                              "expectation": cells[3] if len(cells) > 3 else ""})
            continue

        toks = [t for t in re.split(EN_DASH, trip_txt) if t.strip()]
        if len(toks) != 3:
            # fall back: also accept hyphen-minus only if it yields 3 *valid* markers
            alt = [t for t in trip_txt.split("-") if t.strip()]
            if len(alt) == 3 and all(resolve(t, valid) for t in alt):
                toks = alt
            else:
                continue
        n_rows += 1
        res = [resolve(t, valid) for t in toks]
        if any(r is None for r in res):
            n_unresolved += 1
            for t, r in zip(toks, res):
                if r is None:
                    unresolved_tokens.add(t.strip().strip("*").strip())
            continue
        if len(set(res)) != 3:           # self-pair (polarization) -> not a 3-distinct wedge
            n_selfpair += 1
            continue
        tri = tuple(sorted(idx[r] for r in res))
        if tri in seen:
            continue
        seen.add(tri)
        triples.append(list(tri))
        hub = cells[2] if len(cells) > 2 else ""
        cell = cells[3] if len(cells) > 3 else ""
        sign = cells[4] if len(cells) > 4 else ""
        q = cells[5] if len(cells) > 5 else ""
        meta.append({"number": int(num), "module": module, "hub": resolve(hub, valid) or hub,
                     "cell": cell, "sign": sign, "q": q,
                     "names": [marker_names[i] for i in tri]})

    OUT.write_text(json.dumps(
        {"triples": triples, "meta": meta, "contrasts": contrasts,
         "marker_names": marker_names}))

    print(f"parsed triplet-definition rows : {n_rows}")
    print(f"  self-pairs dropped           : {n_selfpair}")
    print(f"  unresolved-token rows dropped: {n_unresolved}  tokens={sorted(unresolved_tokens)}")
    print(f"  contrast rows (section F)    : {len(contrasts)}")
    print(f"UNIQUE triples written         : {len(triples)}  -> {OUT}")
    # spot-check
    for t in triples[:3]:
        print("   ", t, [marker_names[i] for i in t])


if __name__ == "__main__":
    main()
