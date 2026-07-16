# HistRegGUI v2.0 — Large TIFF/OME-TIFF memory hotfix

Extract this archive into the root of the HistRegGUI repository and allow the listed files to be replaced.

Then create a new commit and start a new GitHub Actions workflow run:

```bash
git add .
git commit -m "Fix large multichannel TIFF memory handling"
git push origin main
```

Do not use **Re-run jobs** on a build created from an older commit.
