# LM_EncDec_Base_LuongGRU

Encoder-decoder translation model ported from legacy TF2 code C3 to PyTorch (cell=gru, attention=general).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 22.8 M
- Test perplexity 60.1, next-word accuracy 0.485
- BLEU-4 greedy (whole test split): 11.9
- BLEU-4 beam width 4 on 500 pairs: 12.6 (greedy on the same pairs 11.8)
- Examples:
  - `this winter has been mild .` -> `cette chambre a ete prise par les fenetres .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils vont se marier en octobre .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `j'ai ete occupe en octobre .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_luong_gru
```
