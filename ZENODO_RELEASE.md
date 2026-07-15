# Zenodo DOI setup for HistRegGUI v1.0

The repository is prepared with both `CITATION.cff` and `.zenodo.json` metadata.
Zenodo gives priority to `.zenodo.json`; `CITATION.cff` remains useful because
GitHub renders it in the repository citation panel.

## One-time activation

1. Make the GitHub repository public.
2. Sign in to Zenodo with the GitHub account that can access the repository.
3. Open the Zenodo **GitHub** integration page and select **Sync now**.
4. Find `Juaco2r/HistRegWSIGUI` and switch it **On**.
5. Confirm that the repository appears as connected.

This activation cannot be performed from repository code; it grants Zenodo
permission to ingest future GitHub releases.

## Publish v1.0

Run the metadata check locally:

```bash
python scripts/validate_release_metadata.py --tag v1.0
```

Commit the final source and push the version tag:

```bash
git add .
git commit -m "Release HistRegGUI v1.0"
git push origin main
git tag v1.0
git push origin v1.0
```

The GitHub Actions workflow builds all six desktop archives and creates the
GitHub Release. Once the repository has been enabled in Zenodo, that GitHub
Release is ingested automatically and receives a version DOI. Zenodo also keeps
a concept DOI that can represent all future versions of the software.

Do not create another `v1.0` release to correct metadata after Zenodo has
archived it. Correct the metadata and publish a new version tag instead.
