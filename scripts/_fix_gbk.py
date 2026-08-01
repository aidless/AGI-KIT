from pathlib import Path
files = ["papers/COVER_LETTER.md", "papers/PUBLISHING.md", "papers/00_INDEX_en.md"]
for f in files:
    raw = Path(f).read_bytes()
    print(f, "raw has", raw.count(b"\xa1\xaa"), "GBK em-dashes")
    # Strip non-UTF8 bytes (0xA1 0xAA etc) by re-decoding
    text = raw.decode("utf-8", errors="replace")
    Path(f).write_text(text, encoding="utf-8")
    print(f, "rewrote", Path(f).stat().st_size, "bytes")
