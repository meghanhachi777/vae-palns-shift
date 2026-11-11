# Paper assets

`make_assets.py` regenerates every number (`numbers.tex`, `numbers.json`), table (`tables/*.tex`) and
figure (`figures/*.pdf`) of the paper directly from the CSV logs in `../results/`. `verify_numbers.py`
re-computes the key numbers by an independent code path and must print `MISMATCHES: 0`.

```bash
cd paper
python3 make_assets.py
python3 verify_numbers.py
```

The manuscript source is not part of this repository while the paper is under review.
