# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Table 1 as a Word file: tables/table1_comparison.md (written by hand) through pandoc, on
landscape pages in 8 pt, to out/table1_comparison.docx (needs pandoc on the PATH).

    python tables/table1_render.py
"""

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tables" / "table1_comparison.md"
OUT = ROOT / "out" / "table1_comparison.docx"
LANDSCAPE = ('<w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
             '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="0" w:footer="0" w:gutter="0"/>')


def rewrite(path, name, change):
    """Apply change() to one XML part of a .docx file, in place."""
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files[name] = change(files[name].decode()).encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)


def small(styles):
    styles = re.sub(r'<w:sz w:val="\d+"\s*/>', '<w:sz w:val="16"/>', styles)
    return re.sub(r'<w:szCs w:val="\d+"\s*/>', '<w:szCs w:val="16"/>', styles)


def landscape(doc):
    doc, n = re.subn(r"<w:sectPr\s*/>|<w:sectPr>.*?</w:sectPr>", f"<w:sectPr>{LANDSCAPE}</w:sectPr>", doc,
                     flags=re.DOTALL)
    if n != 1:
        raise SystemExit(f"expected one page section in pandoc's output, found {n}")
    return doc


def main():
    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "reference.docx"
        ref.write_bytes(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], check=True,
                                       capture_output=True).stdout)
        rewrite(ref, "word/styles.xml", small)
        subprocess.run(["pandoc", str(SOURCE), "--reference-doc", str(ref), "-o", str(OUT)], check=True)
    rewrite(OUT, "word/document.xml", landscape)
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
