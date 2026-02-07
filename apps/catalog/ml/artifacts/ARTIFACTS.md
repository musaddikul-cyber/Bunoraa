Storage recommendations for large ML artifacts

- **Preferred:** Store large model files in cloud storage (S3, GCS, Azure Blob) or CI/CD artifact stores and reference them in the repo by URL or a download script.
- **If you must keep in repo:** Use Git LFS to track the files and be mindful of storage quotas on GitHub.
- **Local backup:** A local backup branch (`backup-before-filter-YYYYMMDD_HHMMSS`) was created in this repo before the history rewrite. Do not push that branch unless you intend to reintroduce large files.

How to add the local artifact to Git LFS (optional):

1. Ensure Git LFS is installed: `git lfs install`
2. Track the path: `git lfs track "apps/products/ml/artifacts/*.pt"`
3. Add and commit the file (it will be stored in LFS): `git add apps/products/ml/artifacts/product_suggestor.pt && git commit -m "Add product_suggestor.pt to Git LFS"`
4. Push: `git push origin main` (this will upload LFS objects)

If you prefer, I can upload the artifact to S3 and add a download helper; tell me which option you prefer and I'll do it.
