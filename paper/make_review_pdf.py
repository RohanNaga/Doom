"""Build paper/review/main_review.pdf: the working paper with each passage highlighted by review status.

    python3 paper/make_review_pdf.py

Green: Rohan read it and his edits are in. Yellow: he gave a direction and the text was written or changed
for him, so it needs one confirming read. No highlight: not read yet. The status of a passage is looked up
by the start of its source line in `STATUS` (default: not read); update that table as the review moves on.
The highlighted copy is for reading only; `paper/main.tex` is never modified.
"""
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review")

DONE, CONFIRM = "done", "confirm"
# start of a source line (after any \item or \caption{) -> status
STATUS = [
    # abstract: read on Sep 29, his three edits applied
    ("World models have become capable simulators", DONE),
    ("A small domain shift, such as a new map", DONE),
    ("With DoomShift, a controlled version", DONE),
    ("We train three diffusion backbones, initialized", DONE),
    ("All three keep their response to controls", DONE),
    ("Very little adaptation repairs it", DONE),
    # introduction
    ("World models learned from recorded play", DONE),              # the opener he chose (variation 1)
    ("However, these models are still evaluated only", DONE),
    ("A learned simulator is useful only", DONE),
    ("Game worlds let us pose this question", DONE),
    ("Yet unchanged dynamics alone", DONE),                          # his sentence, restored on his word
    ("Pretraining one model on every map", DONE),
    ("In short, a world model moved to a new map", DONE),
    ("We introduce DoomShift, a benchmark", DONE),               # short bullets, approved with his two additions
    ("We show that a scene shift breaks appearance", DONE),
    ("We measure how much adaptation repairs", DONE),
    ("\\textbf{Zero-shot vs.", DONE),                                # his title and his chosen ending
    # related work: rewritten on his direction
    ("\\textbf{Game world models.}", DONE),
    ("\\textbf{Generalization of world models.}", DONE),
    ("\\textbf{Adaptation.} Pretrained video and world models", DONE),
    # method and results: single passages written for him
    ("\\textbf{Method overview.}", DONE),                            # the caption he chose (option A)
    ("\\textbf{Data.}", DONE),                                       # read Sep 29, no edits
    ("\\textbf{Models and training.}", DONE),                        # simplified on his wording, approved
    ("We turn all three into next-tic world models", DONE),
    ("We train each backbone for 200k updates", DONE),
    ("\\textbf{Metrics.}", DONE),                                    # read Sep 29
    ("For accuracy, we compute PSNR", DONE),                         # persistence sentence cut on his word
    ("To test control, we swap turn-left", DONE),                     # directional score, his option B
    ("\\textbf{Decoder fine-tuning.}", DONE),                        # his option B with the SD 3.5 numbers
    ("This raises the scene reconstruction", DONE),
    ("\\textbf{Adaptation.} For each unseen map we adapt", DONE),  # shortened on his direction
    ("We train on 8 episodes of the map", DONE),
    ("We report 4k updates throughout", DONE),
    ("As a baseline, we fine-tune all of the U-Net", DONE),
    ("\\textbf{Adaptation cost.}", CONFIRM),                         # condensed on his decisions
    ("After 4k updates on eight episodes", CONFIRM),
    ("Part of the remaining PSNR gap", CONFIRM),
    ("Most of the gain comes early", CONFIRM),
    ("Fine-tuning all of the U-Net", CONFIRM),
    ("One episode already gives most", CONFIRM),
    ("Adaptation costs some in-distribution", CONFIRM),
    ("We introduced DoomShift to measure", CONFIRM),                   # the Conclusion, rewritten for him
    ("On unseen maps of the same game, three diffusion", CONFIRM),
    ("\\textbf{Limitations.}", CONFIRM),
    ("Future work includes other games", CONFIRM),
    ("\\textbf{Zero-shot vs.\\ adapted results on training", DONE),   # the combined table's caption, his title
    ("\\textbf{Zero-shot vs.\\ adapted results per unseen map.}", DONE),   # Figure 3 caption, his title                          # Table 1 caption, his title
    ("\\textbf{Zero-shot evaluation on unseen maps.}", DONE),      # condensed on his outline
    ("Control survives:", DONE),
    ("The scene does not:", DONE),
    ("The loss is nearly the same for all three", DONE),
]
SPLIT_BEFORE = ["To test control, we swap turn-left", "This raises the scene reconstruction"]   # a status change inside one source line
COLORS = {DONE: "reviewdone", CONFIRM: "reviewconfirm"}
PREAMBLE = r"""
\usepackage{xcolor}
\usepackage{soul}
\usepackage{eso-pic}
\definecolor{reviewdone}{rgb}{0.78,0.93,0.78}
\definecolor{reviewconfirm}{rgb}{1.0,0.90,0.55}
\soulregister\citep7 \soulregister\citet7 \soulregister\ref7 \soulregister\appref7
\DeclareRobustCommand{\hldone}[1]{{\sethlcolor{reviewdone}\hl{#1}}}
\DeclareRobustCommand{\hlconfirm}[1]{{\sethlcolor{reviewconfirm}\hl{#1}}}
\AddToShipoutPictureBG{\AtPageUpperLeft{\raisebox{-0.55in}{\hspace{1in}\footnotesize\sffamily
  \colorbox{reviewdone}{\strut read, your edits are in}\quad
  \colorbox{reviewconfirm}{\strut changed for you, confirm}\quad
  \fbox{\strut not read yet}}}}
"""


def status_of(text):
    for start, status in STATUS:
        if text.startswith(start):
            return status
    return None


def wrap(text, status):
    return text if status is None else f"\\hl{status}{{{text}}}"


def mark(tex):
    """The review copy of the paper's source: every body line wrapped by its status."""
    for s in SPLIT_BEFORE:
        tex = tex.replace(" " + s, "\n" + s)
    out, in_body = [], False
    for line in tex.split("\n"):
        if line.startswith("\\begin{abstract}"):
            in_body = True
        if line.startswith("\\label{body-end}"):
            in_body = False
        m = re.match(r"(\\item |\\caption\{)?(.*)$", line)
        head, text = m.group(1) or "", m.group(2)
        if not in_body or not text or (not head and text.startswith(("\\", "%", "{"))
                                       and not text.startswith("\\textbf{")):
            out.append(line)
            continue
        status = status_of(text)
        tail = ""
        if head == "\\caption{":
            text, tail = text[:-1], "}"
        if text.startswith("\\textbf{") and status:          # keep a bold title or run-in heading outside the highlight
            h = re.match(r"(\\textbf\{[^}]*\}) ?(.*)$", text)
            out.append(head + h.group(1) + " " + wrap(h.group(2), status) + tail)
        elif tail:
            out.append(head + wrap(text, status) + tail)
        else:
            # several sentences can share a source line, each with its own status: split on the known starts
            out.append(head + wrap(text, status))
    return "\n".join(out).replace("\\begin{document}", PREAMBLE + "\\begin{document}", 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(HERE):
        src = os.path.join(HERE, name)
        if name in ("review", "versions") or name.startswith("main."):
            continue
        dst = os.path.join(OUT, name)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    with open(os.path.join(HERE, "main.tex")) as f:
        tex = mark(f.read())
    with open(os.path.join(OUT, "main_review.tex"), "w") as f:
        f.write(tex)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main_review.tex"], cwd=OUT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(OUT, "main_review.log"), errors="replace").read()
    print(re.search(r"Output written on .*", log).group(0) if "Output written" in log else "BUILD FAILED")
    print("errors:", len(re.findall(r"^! ", log, flags=re.M)))
    for e in re.findall(r"^! .*\n.*\n.*", log, flags=re.M)[:5]:
        print(e)


if __name__ == "__main__":
    main()
