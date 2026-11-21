import modal, pathlib, sys
app = modal.App("midl2027-latex")
img = (modal.Image.debian_slim(python_version="3.11")
       .apt_install("texlive-latex-extra", "texlive-publishers", "texlive-science", "texlive-fonts-recommended", "texlive-bibtex-extra", "latexmk")
       .add_local_dir("paper", "/paper", copy=True))

@app.function(image=img, timeout=900)
def build(main: str = "main.tex"):
    import subprocess, shutil
    stem = main[:-4]; r = None
    for cmd in (["pdflatex", "-interaction=nonstopmode", main], ["bibtex", stem], ["pdflatex", "-interaction=nonstopmode", main], ["pdflatex", "-interaction=nonstopmode", main]):
        r = subprocess.run(cmd, cwd="/paper", capture_output=True, text=True)
    pdf = pathlib.Path("/paper") / main.replace(".tex", ".pdf")
    log = pathlib.Path("/paper") / main.replace(".tex", ".log")
    return (pdf.read_bytes() if pdf.exists() else b""), (log.read_text(errors="ignore") if log.exists() else r.stdout[-3000:])

@app.local_entrypoint()
def main(tex: str = "midl-fullpaper.tex"):
    pdf, log = build.remote(tex)
    pathlib.Path("paper/_build").mkdir(exist_ok=True)
    pathlib.Path("paper/_build/" + tex.replace(".tex", ".pdf")).write_bytes(pdf); pathlib.Path("paper/_build/build.log").write_text(log, encoding="utf-8")
    print("pdf bytes", len(pdf)); print("\n".join(l for l in log.splitlines() if l.startswith("!") or "Output written" in l)[:2000])
