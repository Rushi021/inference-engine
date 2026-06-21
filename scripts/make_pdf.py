import sys                                                     # CLI args
import subprocess                                               # call Chrome headless
import tempfile                                                 # temp HTML file
from pathlib import Path                                        # path handling

import markdown                                                 # markdown -> HTML

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # headless renderer

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
       font-size: 11pt; line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 20pt; border-bottom: 3px solid #2563eb; padding-bottom: 6px; color: #111; }
h2 { font-size: 15pt; margin-top: 22px; color: #1e3a8a; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
h3 { font-size: 12.5pt; margin-top: 16px; color: #1e40af; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 9.5pt;
       background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 6px;
      padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; line-height: 1.4; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }
th, td { border: 1px solid #d1d5db; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #eff6ff; }
blockquote { border-left: 4px solid #93c5fd; margin: 10px 0; padding: 4px 14px;
             background: #f8fafc; color: #334155; }
strong { color: #111; }
a { color: #2563eb; text-decoration: none; }
h2, h3 { page-break-after: avoid; }
"""

HTML_TMPL = "<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def md_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text()                                  # read markdown source
    body = markdown.markdown(                                   # convert to HTML
        text, extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"]
    )
    html = HTML_TMPL.format(css=CSS, body=body)                # wrap with print styling
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)                                          # Chrome reads from a file
        tmp_html = f.name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)         # ensure output dir
    subprocess.run([                                           # render via Chrome headless
        CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", f"file://{tmp_html}",
    ], check=True, capture_output=True)
    Path(tmp_html).unlink()                                    # clean up temp HTML
    print(f"Wrote {pdf_path}")                                 # confirm


def main():
    if len(sys.argv) < 2:                                      # usage guard
        print("usage: python scripts/make_pdf.py docs/phase-0.md [out.pdf]")
        sys.exit(1)
    md_path = Path(sys.argv[1])                                # input markdown
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        md_path.parent / "pdf" / (md_path.stem + ".pdf")       # default docs/pdf/<name>.pdf
    md_to_pdf(md_path, pdf_path)


if __name__ == "__main__":
    main()
