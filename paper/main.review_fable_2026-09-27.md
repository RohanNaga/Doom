# Prose pass on the four-page draft (Fable 5.1, 12:22 EDT, main a07c415)

Numbered; the owner applies those it agrees with, the numbers wait for Astra's check.

1. Intro, "we found no Doom world model scored on maps it did not train on (MultiGen does not say [4])": write "no published Doom world model is scored on maps it did not train on; MultiGen does not report one [4]".
2. Intro, the long sentence "Because absolute PSNR ranks how static a recording is, every score here is a paired difference against persistence, the last context frame copied forward, a baseline video prediction reported [11, 12, 13] and game world models dropped, read beside the reconstruction upper bound [14, 15]": split. "Because absolute PSNR ranks how static a recording is, every score here is a paired difference against persistence, the last context frame copied forward. Video prediction reports this baseline [11, 12, 13]; game world models dropped it. We read every score beside the reconstruction upper bound [14, 15]."
3. Section 2, "SD 3.5 at a provisional 140k" for the four-tic numbers while Table 1 uses 170k: say "SD 3.5 at 140k, its latest four-tic read" so the two provisional reads are not confused.
4. Section 2, the $A_\text{train}$ interval "4.75 to 5.39" against Table 2's "[4.75, 5.38]": one value; the lead's 10,000-draw figure is 5.38.
5. Section 5, "persistence is weak on fights": "persistence is a weak reference where the camera moves fast".
6. Keywords: "Persistence baseline" is not a keyword a reader searches; "World models, Generalization, Domain adaptation, Doom".
7. Figure 1 caption: the training-map column is being dropped from the file (the lead's four-column redraw, same size); the caption's last clause goes with it.
8. Figure 3 caption, "(b) Every model beats persistence (grey bars) in PSNR on all 13 arenas and loses to it in LPIPS on 13 of 13": good; add "fine-tuned SD 1 decoder for the U-Net and PixArt".
9. Table 2 header "LoRA, arena 7": since the full fine-tune column is [tbd], the arena 7 column reads as arbitrary; caption says "arena 7, the highest-distance arena, as the comparator's target" or drop the column until Monday.
10. Section 4, "a full fine-tune is the comparator ([tbd]: Monday)": the draft to the advisor may keep "[tbd]" but the text should read "a full fine-tune at the GameNGen rate is the comparator (in progress)".
11. Abstract sentence 5 is 39 words; acceptable, but "recover a median 1.8 dB" and "turn the perceptual loss into a win on 11 of 13 arenas" could be one clause: "recover a median 1.8 dB and beat persistence perceptually on 11 of 13".
12. Everywhere: "copy-last" appears in the abstract ("copy-last-frame persistence", "beats copy-last") while the body says "persistence"; one term: "persistence (the last frame copied forward)" once in the abstract, then "persistence".
