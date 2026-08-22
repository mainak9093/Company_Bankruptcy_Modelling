# Report source

`EE708_report.tex` is the source of `../EE708_report_corrected.pdf`.

## Build

```bash
pdflatex EE708_report.tex
pdflatex EE708_report.tex      # twice, to resolve references
```

Requires the TeX Live packages `courier`, `psnfss`, `booktabs`, `subcaption`,
`fancyhdr`, `float` and `hyperref`. With TinyTeX:

```bash
tlmgr install courier psnfss booktabs subcaption fancyhdr float
```

## Figures

`figures/` is generated, not hand-drawn. Regenerate it from the project root
with:

```bash
python src/make_report_figures.py
```

Every figure is produced from the same `build_fold()` call that produces the
metrics, so the figures and the tables in the report cannot drift apart. If you
change the pipeline, re-run that script before rebuilding the PDF.
