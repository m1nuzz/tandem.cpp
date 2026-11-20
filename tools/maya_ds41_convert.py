#!/usr/bin/env python3
"""Project Maya's DeepSeek-V4.1-Flash-1017 GGUF (Maya-XS v2: 3 main shards + its two Q6_K Engram shards) -> the layout llama.cpp /
tandem.cpp load: the 3 main shards copied with split.count 3 -> 4 and split.tensors.count 1042 -> 1043 patched in place (same
sizes), and a 4th shard holding per_layer_token_embd.weight [256, n1 + n2] = the bytes of blk.1.engram_embd then
blk.14.engram_embd (the upstream converter's order; check it with --check against a GSQ-RCO Engram file). Maya's files are
only read. 2026-11-16.
  maya_ds41_convert.py --maya-dir DIR [--engram-dir DIR] --out DIR [--link] [--check OUR_ENGRAM.gguf]
--link: shards 2.. become relative symlinks to Maya's files instead of copies (llama.cpp reads split.no from those shards,
which is the same in both layouts, and nothing else it checks): only the first shard and the joined Engram table are
written (Maya-S: about 210 GB instead of 340). Keep Maya's main shards then; its two Engram shards are not needed by
tandem.cpp after the conversion. 2026-11-21.
The shards are told apart by their contents (a shard holding only *.engram_embd.weight is an Engram shard), so Maya's
Hugging Face folder (main shards and Engram shards side by side) works as is. Maya-S uses newer metadata (deepseek41.engram.*,
per-layer hash offsets): its first shard is copied with the deepseek41.ple.* keys the loader reads added (offsets made global
into the joined table) and hash_layer_count 0. 2026-11-18."""
import argparse, glob, os, shutil, struct, sys

GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rs(f):
    return f.read(rd(f, "Q")).decode()


def skip_val(f, t):
    if t == 8:
        rs(f)
    elif t == 9:
        et, n = rd(f, "I"), rd(f, "Q")
        if et == 8:
            for _ in range(n):
                rs(f)
        else:
            f.seek(n * struct.calcsize(GGUF_TYPES[et]), 1)
    else:
        f.seek(struct.calcsize(GGUF_TYPES[t]), 1)


def header(path):
    """kv positions of scalar keys, tensor infos, the data start"""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, nt, nk = rd(f, "I"), rd(f, "Q"), rd(f, "Q")
    kv, align = {}, 32
    for _ in range(nk):
        k = rs(f)
        t = rd(f, "I")
        pos = f.tell()
        if t in GGUF_TYPES:
            v = rd(f, GGUF_TYPES[t])
            kv[k] = (t, pos, v)
            if k == "general.alignment":
                align = v
        elif t == 9 and k.startswith(("deepseek41.engram.", "deepseek41.ple.")):
            et, n = rd(f, "I"), rd(f, "Q")
            kv[k] = (t, pos, (et, [rd(f, GGUF_TYPES[et]) for _ in range(n)]))
        else:
            skip_val(f, t)
    kv_end = f.tell()
    tensors = []
    for _ in range(nt):
        n = rs(f)
        nd = rd(f, "I")
        dims = [rd(f, "Q") for _ in range(nd)]
        t, off = rd(f, "I"), rd(f, "Q")
        tensors.append((n, dims, t, off))
    ti_end = f.tell()
    data = (ti_end + align - 1) // align * align
    return dict(ver=ver, nk=nk, kv=kv, kv_end=kv_end, ti_end=ti_end, tensors=tensors, data=data, align=align)


def w_kv(f, k, t, v):
    w_str(f, k)
    f.write(struct.pack("<I", t))
    if t == 8:
        w_str(f, v)
    elif t == 9:
        et, vals = v
        f.write(struct.pack("<IQ", et, len(vals)) + struct.pack("<%d%s" % (len(vals), GGUF_TYPES[et]), *vals))
    else:
        f.write(struct.pack("<" + GGUF_TYPES[t], v))


def ple_keys(kv):
    """Maya-S (newer Maya metadata) names the Engram hash deepseek41.engram.* with per-layer head offsets; the loader reads
    deepseek41.ple.* (GSQ-RCO / upstream converter) with offsets into the joined table, and deepseek41.hash_layer_count.
    Returns the keys to add, [] when the file already has them (Maya-XS)."""
    if "deepseek41.ple.layers" in kv or "deepseek41.engram.layers" not in kv:
        return []
    E = lambda k: kv["deepseek41.engram." + k][2]
    layers, nemb, offs, primes = E("layers")[1], E("num_embeddings")[1], E("offsets")[1], E("primes")[1]
    per = len(offs) // len(layers)
    base = [sum(nemb[:e]) for e in range(len(layers))]
    goffs = [o + base[i // per] for i, o in enumerate(offs)]
    add = [("deepseek41.ple.layers", 9, (4, layers)),
           ("deepseek41.ple.ngram_size", 4, E("max_ngram_size")),
           ("deepseek41.ple.heads_per_ngram", 4, E("head_count")),
           ("deepseek41.embedding_length_per_layer_input", 4, E("head_dim")),
           ("deepseek41.ple.layer_multipliers", 9, (10, E("multipliers")[1])),
           ("deepseek41.ple.head_offsets", 9, (10, goffs)),
           ("deepseek41.ple.head_vocab_sizes", 9, (10, primes))]
    if "deepseek41.hash_layer_count" not in kv:
        add.append(("deepseek41.hash_layer_count", 4, 0))   # no hash-routed layers (no tid2eid tensors), as in GSQ-RCO
    return add


def add_kv_copy(src, dst, patches, add):
    """copy src to dst with the scalar kv patches applied and the keys in add appended to the kv section (the header grows,
    tensor offsets are relative to the data start and stay)"""
    h = header(src)
    with open(src, "rb") as s:
        head = bytearray(s.read(h["kv_end"]))
        ti = s.read(h["ti_end"] - h["kv_end"])
    struct.pack_into("<Q", head, 16, h["nk"] + len(add))
    for k, v in patches.items():
        t, pos, _ = h["kv"][k]
        struct.pack_into("<" + GGUF_TYPES[t], head, pos, v)
    with open(dst, "wb") as f:
        f.write(head)
        for k, t, v in add:
            w_kv(f, k, t, v)
        f.write(ti)
        f.write(b"\0" * ((-f.tell()) % h["align"]))
        with open(src, "rb") as s:
            s.seek(h["data"])
            shutil.copyfileobj(s, f, 64 << 20)
    print(f"copied {os.path.basename(dst)}: " + ", ".join(f"{k} {h['kv'][k][2]} -> {v}" for k, v in patches.items())
          + "; added " + ", ".join(k for k, *_ in add))


def patch_copy(src, dst, patches):
    """copy src to dst, then overwrite the scalar kv values in patches {key: value} at their positions"""
    h = header(src)
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        for k, v in patches.items():
            t, pos, _ = h["kv"][k]
            f.seek(pos)
            f.write(struct.pack("<" + GGUF_TYPES[t], v))
    print(f"copied {os.path.basename(dst)}: " + ", ".join(f"{k} {h['kv'][k][2]} -> {v}" for k, v in patches.items()))


def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)) + b)


def write_engram(tables, dst, split_no, split_count, n_tensors):
    """tables: [(path, name, dims, type, off, data_start)] of the Engram tensors in order; writes one tensor of them all"""
    rows = sum(t[2][1] for t in tables)
    ne0, qtype = tables[0][2][0], tables[0][3]
    assert all(t[2][0] == ne0 and t[3] == qtype for t in tables)
    sizes = [os.path.getsize(t[0]) - (t[5] + t[4]) for t in tables]
    kvs = [("general.name", 8, "DeepSeek-V4.1-Flash-1017 Maya Engram tables (from Maya's two Engram shards)"),
           ("split.no", 2, split_no), ("split.count", 2, split_count), ("split.tensors.count", 5, n_tensors)]
    import io
    hb = io.BytesIO()
    hb.write(b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)))
    for k, t, v in kvs:
        w_kv(hb, k, t, v)
    w_str(hb, "per_layer_token_embd.weight")
    hb.write(struct.pack("<I", 2) + struct.pack("<QQ", ne0, rows) + struct.pack("<IQ", qtype, 0))
    hb.write(b"\0" * ((-hb.tell()) % 32))
    hdr = hb.getvalue()
    if os.path.exists(dst) and os.path.getsize(dst) == len(hdr) + sum(sizes):   # rerun: header only
        with open(dst, "r+b") as f:
            f.write(hdr)
        print(f"kept {os.path.basename(dst)} (header rewritten)")
        return
    with open(dst, "wb") as f:
        f.write(hdr)
        for (path, *_rest), sz in zip(tables, sizes):
            with open(path, "rb") as s:
                s.seek(_rest[4] + _rest[3])
                left = sz
                while left > 0:
                    buf = s.read(min(left, 64 << 20))
                    f.write(buf)
                    left -= len(buf)
    print(f"wrote {os.path.basename(dst)}: per_layer_token_embd.weight [{ne0}, {rows}] type {qtype}, {sum(sizes)/1e9:.1f} GB")


def check(ours, out_engram, rows):
    """dequantize rows of ours (IQ4_NL) and of the converted table (Q6_K) on both sides of the join: cosine near 1 if the order holds"""
    sys.path.insert(0, os.path.expanduser("~/llama-moe/src/llama.cpp-rebase/gguf-py"))
    import numpy as np
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType as Q, GGML_QUANT_SIZES

    def row(path, r):
        h = header(path)
        _, dims, t, off = h["tensors"][0]
        bs, ts = GGML_QUANT_SIZES[Q(t)]
        rb = dims[0] // bs * ts
        with open(path, "rb") as f:
            f.seek(h["data"] + off + r * rb)
            return dequantize(np.frombuffer(f.read(rb), dtype=np.uint8), Q(t)).astype(np.float64).ravel()
    for r in rows:
        a, b = row(ours, r), row(out_engram, r)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"row {r}: cosine {cos:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maya-dir", required=True)
    ap.add_argument("--engram-dir", default="", help="default: --maya-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="", help="a GSQ-RCO Engram file (per_layer_token_embd) to compare rows with")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--link", action="store_true", help="symlink main shards 2.. to Maya's files instead of copying them")
    a = ap.parse_args()
    cand = sorted(set(glob.glob(f"{a.maya_dir}/*.gguf")) | set(glob.glob(f"{a.engram_dir or a.maya_dir}/*.gguf")))
    mains, engs = [], []
    for p in cand:
        names = [t[0] for t in header(p)["tensors"]]
        (engs if names and all(n.endswith("engram_embd.weight") for n in names) else mains).append(p)
    assert mains and len(engs) == 2, f"expected the main shards and two Engram shards, found {len(mains)} and {len(engs)}"
    print("main shards:", [os.path.basename(p) for p in mains], "engram shards:", [os.path.basename(p) for p in engs])
    os.makedirs(a.out, exist_ok=True)
    n_main = len(mains)
    hs = [header(p) for p in mains]
    n_tensors = sum(len(h["tensors"]) for h in hs) + 1   # Maya-S counts its Engram tensors in split.tensors.count, Maya-XS does not
    stem = os.path.basename(mains[0]).rsplit("-00001-of-", 1)[0]
    out_eng = f"{a.out}/{stem}-{n_main + 1:05d}-of-{n_main + 1:05d}.gguf"
    if not a.check_only:
        for i, p in enumerate(mains):
            dst = f"{a.out}/{stem}-{i + 1:05d}-of-{n_main + 1:05d}.gguf"
            if os.path.islink(dst):
                os.remove(dst)   # a link from an earlier --link run: never write through it into Maya's file
            add = ple_keys(hs[i]["kv"])
            if a.link and i > 0 and not add:
                assert hs[i]["kv"]["split.no"][2] == i, f"{p}: split.no {hs[i]['kv']['split.no'][2]}, expected {i}"
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(os.path.relpath(os.path.abspath(p), os.path.dirname(os.path.abspath(dst))), dst)
                print(f"linked {os.path.basename(dst)} -> {os.path.basename(p)}")
                continue
            patches = {"split.count": n_main + 1, "split.tensors.count": n_tensors}
            if add:
                add_kv_copy(p, dst, patches, add)
            else:
                patch_copy(p, dst, patches)
        tables = []
        for p in engs:
            h = header(p)
            for n, dims, t, off in h["tensors"]:
                if n.endswith("engram_embd.weight"):
                    tables.append((p, n, dims, t, off, h["data"]))
        tables.sort(key=lambda x: int(x[1].split(".")[1]))   # blk.1 then blk.14
        print("engram order:", [t[1] for t in tables])
        if os.path.islink(out_eng):
            os.remove(out_eng)   # never rewrite a header through a link into another set's file
        write_engram(tables, out_eng, n_main, n_main + 1, n_tensors)
    if a.check:
        n1 = None
        for p in engs:
            for n, dims, *_ in header(p)["tensors"]:
                if n == "blk.1.engram_embd.weight":
                    n1 = dims[1]
        check(a.check, out_eng, [0, 1, n1 - 1, n1, n1 + 1, 2 * n1 - 7])


if __name__ == "__main__":
    main()
