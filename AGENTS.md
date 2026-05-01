# AGENTS.md

## Repo

- This is a paper-first LaTeX repository.
- Most work should be direct manuscript editing.
- Keep edits narrow and follow the existing LaTeX structure.
- When the user says to update Overleaf, treat that as updating the local LaTeX source files in this repo, not as requiring a separate Overleaf remote.

## Git

- Remote: `origin` -> `https://github.com/AI4Collaboration/SuperSycophantic.git`
- Default branch: `main`
- Pull from `origin/main` before new work after a long gap.
- If the user says `push`, commit and push directly to `origin/main` unless they ask for a branch or partial push.
- Before committing, check the intended file list. Do not include accidental build outputs or unrelated binary changes.
- If normal HTTPS push is reset, retry with:
  `git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999 push origin main`
- If HTTPS still fails, use:
  `git push git@github.com:AI4Collaboration/SuperSycophantic.git main:main`

## Do Not Compile Locally

- Do not run `latexmk`, `pdflatex`, `bibtex`, or any local LaTeX build command unless the user explicitly asks.
- Do not create, remove, or clean build artifacts unless the user explicitly asks.
- Verify manuscript edits with text checks, git diff, and file references by default.

## Main Files

- `main.tex`: top-level entry point
- `sections/0-main.tex`: main paper assembly and Figure 2
- `sections/1-Intro.tex`: abstract and introduction
- `sections/2-RelatedWork.tex`: related work
- `sections/3-Method.tex`: benchmark design, evaluation, results draft
- `sections/appendix.tex`: appendix methods, source notes, annotation details
- `tables/BenchScope.tex`: benchmark comparison table
- `main.bib`: bibliography

## Figures

- Figure 1: `images/Figure1.pdf`, referenced from `sections/1-Intro.tex`.
- Keep Figure 1 between the abstract and the Introduction in `sections/1-Intro.tex`.
- Figure 2: use `images/Figure2.pdf`, referenced from `sections/0-main.tex`.
- Figure 2 has no SVG source in the repo now. Do not recreate `images/Figure2.svg` unless the user asks.
- If Figure 2 needs to look flatter, adjust the TeX wrapper in `sections/0-main.tex`, for example with `\scalebox{1}[...]`.
- Do not keep repo PNG previews for Figure 2 unless the user asks.

## Manuscript Rules

- Default to text-only edits.
- Do not edit `neurips_2026.sty` or `neurips_checklist.tex` unless asked.
- Keep main-text source notes concise. Put detailed URLs and source examples in the appendix.
- Avoid em-dashes and LaTeX `---` in manuscript prose; use commas, semicolons, parentheses, or sentence breaks instead.
- Avoid footnote-based source dumps in the main text.
- If a methodological decision changes, keep abstract, intro, method, appendix, and affected tables consistent.
- Watch for stale placeholder text, especially in appendix and tables.

## Benchmark Conventions

- Core trigger taxonomy: seven Cialdini-style triggers plus a simple baseline.
- Primary endpoint: answer change under unsupported pressure.
- GT supports truth preservation and truth-departure analysis.
- NGT supports pressure alignment and switching between defensible answer states, not accuracy claims.
- Eligibility is based on a committed initial answer.
- Evidence-bearing variants are ablations, not new trigger families.

## Bibliography

- Add new BibTeX entries only from arXiv or DBLP.
- Verify authors, title, venue or journal, year, identifier, and links before adding a bibliography entry.
