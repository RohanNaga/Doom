# GPTZero-style reference check (2026-09-27)

Scope: the 27 references that `paper/main.pdf` renders (the appendix is switched off, so `mensink2021factors` and `westny2026latent` do not appear). This file covers four things: what GPTZero checks, what the real scan of the Sep 27 PDF flagged and why, what each rendered reference looks like against the databases after commit f837b2e (refs.bib) and dd82c3d (T1 font encoding), and what could still be flagged.

Earlier audits: `citation-check-2026-09-27.md` (Astra) and `citation-check-opus-2026-09-27.md` (Opus). Both checked refs.bib. Neither checked the rendered PDF, which is what GPTZero parses. One earlier recommendation was wrong, and this file supersedes it: drop Yue Wu from PixArt-α. The ICLR 2024 camera-ready PDF (proceedings.iclr.cc/paper_files/paper/2024/file/fe989bb038b5dcc44181255dd6913e43-Paper-Conference.pdf, page 1) prints 11 authors including Yue Wu. Only the proceedings.iclr.cc metadata page lists 10.

## 1. What GPTZero checks

Sources: the [technical report](https://gptzero.me/news/how-does-gptzeros-hallucination-check-detector-work-a-technical-report/) (May 12, 2026), the [NeurIPS 2025 write-up](https://gptzero.me/news/neurips/) and the [ICLR 2026 write-up](https://gptzero.me/news/iclr-2026/).

Pipeline, per the technical report:
1. It parses the rendered document, with a dedicated PDF parser that also resolves hyperlinked text.
2. It triages each citation as formally verifiable, needing reasoning, or non-checkable. Non-checkable citations are labelled unknown.
3. It retrieves candidate records three ways: a local index, external citation APIs queried with the extracted title, authors and DOI, and targeted web search. It pools the results and ranks them by similarity to the citation.
4. It compares six components (title, authors, publisher/journal, date, URL, DOI). Each gets one of: match, partial match, not match, unknown. Authors can also get "weak match".
5. It assigns a class from tiered rules:
   - Tier 1 is author and title.
   - Tier 2 is URL or DOI, and arXiv IDs count here.
   - Tier 3 is venue and date.

The class rules:
- **exist**: authors and title match. Alternatively, the title matches, authors are missing, and a Tier-2 identifier matches.
- **exist with minor issues**: the citation meets "exist" but some component does not match. Alternatively: one Tier-1 component matches, the other is a partial match, a Tier-2 identifier matches, and no component is "not match".
- **fake**: authors or title do not match. Alternatively, authors match only weakly and a Tier-2 identifier does not match (or all Tier-2 components are unknown and the venue does not match). A URL-only citation with a dead link is also fake.
- **unknown**: the source is explicitly offline, or the citation has only one component.
- **unsure**: anything else. The UI shows this as "may not exist".

The failure taxonomy used in the write-ups:
- **Nonexistent work**: no candidate matches both title and authors.
- **Wrong or invented authors**: fabricated names, authors dropped or added, first names extrapolated from initials.
- **Altered title**: paraphrased, or merged from two papers.
- **Wrong venue or year**: counted as a human-style flaw when title and authors match.
- **Wrong identifier**: the arXiv ID or DOI resolves to a different paper. This is the most common confirmed hallucination in the NeurIPS table.
- **Unreachable link**: the citation is only a URL, and the URL is dead.

The write-ups exclude spelling slips, dead URLs and missing locators from "hallucination". A human then confirms each flag before any consequence. Two lines matter most for us:
- The ICLR write-up says a citation whose title and authors clearly match a real source is not a hallucination, "even if the rest of the citation is wildly inaccurate".
- The NeurIPS write-up admits the false-positive rate is higher because the tool "will flag any citation that can't be verified online".

## 2. What the real scan saw (Sep 27 PDF, before the fixes)

Report: `~/Downloads/GPTZero AI Scan - References.pdf`. It parsed 29 citations from 27 references, because it split Genie and Matrix-Game into two citations each. Results: 1 "don't exist", 7 "may not exist", 21 "exist" (9 of them with issues). "URL/DOI: Missing" appeared on almost every card because the corlabbrvnat style drops `eprint`, so `@inproceedings` entries printed no identifier at all.

| Flag | Cause found in the parsed text | Fix |
|---|---|---|
| Genie: does not exist (Author: No match) | The 4-line, 25-author block was cut off as its own "Untitled" citation, which matched arXiv 2402.15391. The remainder began at the wrapped "T. Rockt¨aschel". Under OT1 font encoding the umlaut is a separate glyph, so that lone author failed to match. | T1 encoding (dd82c3d) gives a precomposed ä (checked: U+00FC in the text layer for Müller). PMLR url and PMLR name forms added. |
| Rombach LDM: may not exist (Title partial) | The title was cut at the hyphen in "syn-/thesis". | Hyphenation disabled inside the bibliography (`@preamble` setting `\bibfont`); DOI added. |
| Matrix-Game 2.0: may not exist (Title partial) | Split at "Matrix-/game"; the style lowercased "Game". | `{Matrix-Game}` braced; no hyphenation; arXiv DOI added. |
| Lotter PredNet: may not exist (Title partial) | The parser dropped the second line of the title. | arXiv DOI added. Title + author match + DOI match now gives at least "exist with minor issues". |
| Villegas: may not exist (Author partial) | No cause visible: the 6 surnames equal arXiv's. Likely the preceding "2017." glued onto "[14]". | arXiv DOI added (same rule as Lotter). |
| DiffFit: may not exist (Author partial, Date missing, Publisher no match) | The parser prefixed "IEEE/CVF:" to the title and matched the arXiv record. | IEEE DOI added. The Crossref record matches the title, all 8 authors, 2023 and ICCV exactly. |
| PixArt-α: may not exist (Author partial) | The tool found the 11-author record; we printed 10. | 11 authors restored, per the camera-ready PDF; arXiv DOI with the same 11. |
| SD3: may not exist (Author partial) | Our 14 authors equal the PMLR page's 14. The printed "M¨uller" was the only defect. | T1 encoding; PMLR url and booktitle. |
| Taylor: minor issue (URL: No match) | The URL wrapped as "jmlr.\norg" and was truncated to ".htm". The tool matched a jmlr.csail.mit.edu PDF instead. | The URL now sits on one line. |
| OccWorld: DOI printed as "72624-8 4" | Under OT1, `\nolinkurl` draws the underscore as a rule, so it is missing from the text layer. | Now given as a `url` (typewriter underscore); T1 also fixes `\doi`. |

## 3. Rendered references against the databases (after f837b2e and dd82c3d)

Method: each rendered title was queried against:
- Crossref (`query.bibliographic`)
- OpenAlex (`search`)
- the arXiv API (`ti:"..."`)

Every rendered DOI was also resolved at doi.org, Crossref or DataCite, and OpenAlex. Every URL was fetched. All 27 identifiers resolve: 200 status, or 202 for IEEE Xplore.

Two sources were unavailable:
- Semantic Scholar returned 429 on every search and on the batch endpoint, so it contributed nothing.
- DBLP and OpenReview (including api.openreview.net) serve bot challenges. They are unusable for scripts, and probably for GPTZero's fetcher too.

Crossref does not index arXiv, ICLR, PMLR or pre-2024 NeurIPS, so a Crossref miss on those entries is expected and says nothing.

Raw responses are in the session scratchpad (`gz/raw/`, `gz/analysis.txt`).

**Classes:**
- **CLEAN**: one record matches the normalized title, every surname in order, the year and the venue, and the rendered identifier resolves to that same record.
- **PARTIAL**: title and authors match exactly, but year or venue differs in every queried record.
- **NOT FOUND**: nothing matches.

| # | Rendered reference (short) | Database hits (title/authors exact) | Field mismatches | Class | Scan before | Change |
|---|---|---|---|---|---|---|
| 1 | Valevski et al., Diffusion models are real-time game engines, ICLR 2025, doi 10.48550/arXiv.2408.14837 | OpenAlex Y/Y, arXiv Y/Y, ICLR proceedings page Y/Y (2025) | Year: arXiv/OpenAlex/DataCite 2024 vs 2025 | PARTIAL | minor (date) | None needed. Exact option: `url` = proceedings.iclr.cc 2025 page, but that URL wraps over two lines |
| 2 | Alonso et al., Diffusion for world modeling, NeurIPS 2024, doi 10.52202/079017-1873 | Crossref Y/Y (NeurIPS 37, 2024), OpenAlex, arXiv | none | CLEAN | exist | — |
| 3 | He et al. (19), Matrix-Game 2.0, arXiv 2508.13009, 2025 | arXiv Y/Y (19), DataCite/OpenAlex by DOI | Current v4 title drops two commas; the v1 title (which the tool matched) is exact | CLEAN | split; may not exist | — |
| 4 | Po et al., MultiGen, arXiv 2603.06679, 2026 | OpenAlex Y/Y, arXiv Y/Y | none | CLEAN | exist | — |
| 5 | Chen et al. (13), XEWorld, arXiv 2608.05799, 2026 | OpenAlex Y/Y, arXiv Y/Y | none | CLEAN | exist | — |
| 6 | Rigter et al., AVID, Reinforcement Learning Journal 6:737–764, 2025, url RLJ | arXiv Y/Y, Crossref/OpenAlex Y/Y on a Qeios copy (10.32388/h7bfdw); RLJ page exact | Year: every indexed record says 2024; only the RLJ page says 2025 | PARTIAL | minor (date) | None. RLJ is the archival record |
| 7 | Gao et al., AdaWorld, Proc. 42nd ICML, PMLR 267:18744–18771, 2025, url PMLR | arXiv Y/Y (2025, "ICML 2025"), PMLR page exact; OpenAlex title search misses it, but a DOI lookup finds it | none | CLEAN | exist | — |
| 8 | Koh et al., Pathdreamer, ICCV 2021, IEEE DOI | Crossref Y/Y exact | none | CLEAN | exist | — |
| 9 | Lample & Chaplot, Playing FPS games, AAAI 2017, AAAI DOI | Crossref Y/Y exact | none | CLEAN | exist | — |
| 10 | Wydmuch et al., ViZDoom competitions, IEEE ToG 11(3):248–259, 2019, DOI | Crossref Y/Y (2019); OpenAlex and arXiv say 2018 | Title case "From" only | CLEAN | minor (tool matched arXiv) | — |
| 11 | Gao et al., Vista, NeurIPS 2024, doi 10.52202/079017-2906 | Crossref Y/Y exact | none | CLEAN | exist | — |
| 12 | Mathieu et al., Deep multi-scale video prediction, ICLR 2016, arXiv DOI | OpenAlex Y/Y (has an ICLR location), arXiv Y/Y | Year: 2015 in every record | PARTIAL | minor (date) | None (the year offset is the canonical "minor issue") |
| 13 | Lotter et al., Deep predictive coding networks, ICLR 2017, arXiv DOI | OpenAlex Y/Y, arXiv Y/Y | Year: 2016 | PARTIAL | may not exist (title split) | None |
| 14 | Villegas et al., High fidelity video prediction, NeurIPS 2019, arXiv DOI | OpenAlex Y/Y 2019, arXiv Y/Y 2019 (comment names NeurIPS), NeurIPS page exact | none | CLEAN | may not exist (author partial) | — |
| 15 | Zheng et al., OccWorld, ECCV 2024 pp. 55–72, url doi.org/…_4 | Crossref Y/Y (ECCV 2024 LNCS), OpenAlex Y/Y | none | CLEAN | minor | — |
| 16 | Karypidis et al., DINO-Foresight, NeurIPS 2025, doi 10.52202/085713-5466 | Crossref Y/Y (NeurIPS 38, 2025) | none | CLEAN | minor (arXiv 2024) | — |
| 17 | Kempka et al., ViZDoom, CIG 2016, IEEE DOI | Crossref Y/Y exact | none | CLEAN | exist | — |
| 18 | Rombach et al., High-resolution image synthesis, CVPR 2022, IEEE DOI | Crossref Y/Y exact | none | CLEAN | may not exist (title split) | — |
| 19 | Chen et al. (11), PixArt-α, ICLR 2024, arXiv DOI | OpenAlex Y/Y (11), arXiv Y/Y (11); ICLR camera-ready PDF 11 authors | Year: arXiv 2023. The ICLR metadata page lists 10 authors | PARTIAL | may not exist (author partial) | None |
| 20 | Esser et al. (14), Scaling rectified flow transformers, Proc. 41st ICML, PMLR 235, url PMLR | PMLR page exact (14 authors); arXiv/OpenAlex list 17 (adds Lacey, Goodwin, Marek) | Authors vs arXiv: 3 extra there | CLEAN (PMLR) | may not exist (author partial) | — |
| 21 | Salimans & Ho, Progressive distillation, ICLR 2022, arXiv DOI | OpenAlex Y/Y 2022, arXiv Y/Y 2022 ("Published … at ICLR 2022") | none | CLEAN | exist | — |
| 22 | Song et al., DDIM, ICLR 2021, arXiv DOI | OpenAlex Y/Y, arXiv Y/Y | Year: 2020 | PARTIAL | minor (date) | None |
| 23 | Bruce et al. (25), Genie, Proc. 41st ICML, PMLR 235:4603–4623, url PMLR | PMLR page exact (name forms copied); arXiv Y/Y (25); OpenAlex Y/Y | none | CLEAN | does not exist (parse) | See risk 1 |
| 24 | Zhang et al., LPIPS, CVPR 2018, IEEE DOI | Crossref Y/Y exact | none | CLEAN | exist | — |
| 25 | Hu et al., LoRA, ICLR 2022, doi 10.48550/arXiv.2106.09685 | arXiv Y/Y. **OpenAlex's record for this DOI is corrupted**: its title reads "LoRA Fine-Tuning of a 3B Code LLM for Algorithmic Efficiency" (merged with a Zenodo record), and the OpenAlex title search does not return LoRA in the top 5 | Year: 2021; corrupted DOI record | PARTIAL | minor (date) | Replace `doi` with `url = {https://arxiv.org/abs/2106.09685}` (see below) |
| 26 | Xie et al., DiffFit, ICCV 2023, IEEE DOI | Crossref Y/Y exact ("Parameter-Efficient") | none | CLEAN | may not exist (author partial) | — |
| 27 | Taylor & Stone, Transfer learning for RL domains, JMLR 10(56):1633–1685, 2009, url JMLR | OpenAlex Y/Y exact (JMLR 2009) | none | CLEAN | minor (URL) | — |

**Counts: 20 CLEAN, 7 PARTIAL, 0 NOT FOUND.** All 7 PARTIALs share one pattern: title and all authors match, and the only difference is the year (arXiv year against conference year) or the venue. Under the published rules that is "exist with minor issues", never "fake".

## 4. refs.bib edits

Applied in f837b2e, the one allowed commit:
- Every cited entry now prints a doi or url.
  - Venue DOIs: DIAMOND, Vista, DINO-Foresight, Pathdreamer, ViZDoom CIG, LDM, LPIPS, DiffFit, Lample AAAI.
  - arXiv DOIs: all ICLR entries, Villegas, Matrix-Game, MultiGen, XEWorld.
  - PMLR urls: Genie, SD3, AdaWorld, using PMLR's own booktitle and series.
  - RLJ `@article` for AVID.
  - OccWorld DOI given as a url.
- Genie authors use the PMLR forms (M. D. Dennis, S. M. E. Bechtle, S. C. Y. Chan, N. De Freitas).
- PixArt has 11 authors.
- `{Matrix-Game}` is braced.
- A bibliography-scoped `@preamble` sets `\hyphenpenalty` and `\exhyphenpenalty` to 10000 through natbib's `\bibfont` hook.

Build: 0 bibtex warnings; 27 of 27 entries carry an identifier. The references now end on page 9 instead of page 8.

Still recommended (not applied, because a second refs.bib commit was not authorized):

```bibtex
% hu2022lora: OpenAlex maps 10.48550/arXiv.2106.09685 to a corrupted title; point at arXiv directly
  doi           = {10.48550/arXiv.2106.09685},   % remove
  url           = {https://arxiv.org/abs/2106.09685}   % add
```

Optional, for exact venue-year matches:
- `valevski2024gamengen`: `url = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/b71ecea210f7159f31e46631fe5c838f-Abstract-Conference.html}` in place of the doi. The ICLR page matches the title, all 4 authors and 2025, but the URL wraps over two lines.
- No other PARTIAL can become exact without citing the arXiv year instead of the venue year. I do not recommend that.

## 5. Risks that remain

1. **Genie's 25-author block.** GPTZero split this entry at a line wrap once already. With T1 accents, the PMLR url and PMLR name forms, a clean parse gives "exist". A bad split can still leave a fragment with a lone author, which is "Author: No match" and therefore fake. The only remaining lever is `and others` after about 10 authors, which renders "et al.". GPTZero's own examples of real citations use "et al.". That is the author's call.
2. **Record choice.** GPTZero often picks the arXiv record even when a venue DOI is printed (it did for Wydmuch). A Tier-2 identifier that does not match the chosen record matters only when authors match weakly. After T1, every author list equals its arXiv list except SD3: arXiv has 17 authors, we print PMLR's 14. If the tool picks arXiv for SD3, expect "author partial", which lands in minor issues or unsure depending on whether the PMLR url is also judged a match.
3. **The corrupted OpenAlex record for LoRA's arXiv DOI.** If the tool resolves our DOI through OpenAlex, it sees a different title. Swap to the arXiv url (edit above).
