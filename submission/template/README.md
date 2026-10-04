# Template only

`predict.ipynb` is the clean generated offline entry point. It requires bundled
`assets/model/model.pt`, `assets/config.yaml`, and `assets/code/terradelta`.
Do not ZIP this folder by itself. Use scripts/export_submission.py and
scripts/make_submission_zip.py; they include assets and original LICENSE/NOTICE.
Never add an API key or submission magic cell.
