"""实验：纯 Python 从 OLE2 版 .doc 抽取正文文本（验证可行性）."""
import sys

import olefile


def extract_doc_text(path: str) -> str:
    ole = olefile.OleFileIO(path)
    wd = ole.openstream("WordDocument").read()

    # FibBase: 0x000A 的 bit9 = fWhichTblStm
    flags = int.from_bytes(wd[0x000A:0x000C], "little")
    table_name = "1Table" if (flags & 0x0200) else "0Table"
    if not ole.exists(table_name):
        table_name = "0Table" if ole.exists("0Table") else "1Table"
    tbl = ole.openstream(table_name).read()

    fc_clx = int.from_bytes(wd[0x01A2:0x01A6], "little")
    lcb_clx = int.from_bytes(wd[0x01A6:0x01AA], "little")
    clx = tbl[fc_clx: fc_clx + lcb_clx]

    # Clx = [Prc]* + Pcdt(0x02, lcb, PlcPcd)
    i = 0
    while i < len(clx) and clx[i] == 0x01:
        cb = int.from_bytes(clx[i + 1: i + 3], "little")
        i += 3 + cb
    assert clx[i] == 0x02, f"unexpected clxt {clx[i]:#x}"
    lcb_plcpcd = int.from_bytes(clx[i + 1: i + 5], "little")
    plc = clx[i + 5: i + 5 + lcb_plcpcd]

    n = (lcb_plcpcd - 4) // 12
    cps = [int.from_bytes(plc[k * 4: k * 4 + 4], "little") for k in range(n + 1)]
    out = []
    for k in range(n):
        base = 4 * (n + 1) + k * 8
        fc = int.from_bytes(plc[base + 2: base + 6], "little")
        compressed = bool(fc & 0x40000000)
        fc &= 0x3FFFFFFF
        length = cps[k + 1] - cps[k]
        if compressed:
            raw = wd[fc // 2: fc // 2 + length]
            out.append(raw.decode("cp1252", errors="replace"))
        else:
            raw = wd[fc: fc + length * 2]
            out.append(raw.decode("utf-16-le", errors="replace"))
    ole.close()
    return "".join(out)


if __name__ == "__main__":
    text = extract_doc_text(sys.argv[1])
    bad = sum(1 for ch in text if ch == "\ufffd")
    print(f"len={len(text)} replacement_chars={bad} ratio={bad / max(1, len(text)):.4f}")
    print("---- head ----")
    print(text[:600].replace("\r", "\n"))
    print("---- middle ----")
    mid = len(text) // 2
    print(text[mid:mid + 400].replace("\r", "\n"))
