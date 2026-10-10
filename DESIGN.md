# DESIGN
## 1. On-disk formats (frozen at contracts-v1)
### .bin            n_rows × row_len tokens, row-major, <u2 or <u4, header-less (memmap at offset 0)
### .idx            128-byte header `<8sHHBBBBIIQQQQ32s32s8s` (table of every field), then row_seg_ptr <u8[n+1],
                    seg_doc_uid <u8, seg_start/seg_len/seg_piece <u4; size = 128 + 8(n_rows+1) + 20 n_segments
### dataset.manifest  canonical JSON (sort_keys, indent 2, UTF-8, trailing newline, ints only); dataset_id = sha256(bytes)
### docs.jsonl      one canonical line per selected doc, sorted by doc_uid; doc_uid as f"{uid:016x}"
### token cache     .tok.idx header `<8sHHB3xQ`, doc_uid <u8[n], offsets <u8[n+1]
### Versioning      major must match; higher minors use reserved bytes (.idx) or extra keys (manifest) only
## 2. Invariants verify checks (and the A4/A5 rules)