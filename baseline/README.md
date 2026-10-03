# Official reference (immutable local copy)

Source: https://drive.google.com/file/d/1RZjoG0hITfY5XqIPuv5KtKYV01XUiOT3/view

Download `03_illegal structure submission.zip` manually, and extract unchanged into
`baseline/original/`. Expected notebook path:
`baseline/original/03_illegal structure submission/predict.ipynb`.
The ZIP is about 51 MiB compressed, checkpoint 57,461,007 bytes.
`integrity.json` records each original file's size and SHA-256.

Original assets and ZIPs are intentionally not committed. The bundled grant is
for competition participation; general public redistribution outside that purpose
is not assumed. Original notebooks include submission magic commands; never
execute the original entire notebook during preparation.

```sh
python scripts/inspect_baseline.py --forward
```

This command inspects safe checkpoint metadata and runs a single CPU forward;
it never executes notebook code or training.
