"""Bangla Mega-Corpus Harvester — HuggingFace phase.

Collects all Bangla (`bn`) datasets available on HuggingFace Hub into
per-source folders with metadata + failure log, so the corpus can later be
normalized and deduplicated.

Usage (Colab or local):
  python harvest_hf.py --out /content/raw --top 200      # biggest first
  python harvest_hf.py --out /content/raw --all          # everything

Then Kaggle phase (needs ~/.kaggle/kaggle.json):
  python harvest_hf.py --kaggle --out /content/raw --query bangla
"""
import argparse, csv, json, os, re, time
from huggingface_hub import HfApi, snapshot_download


def build_manifest(api: HfApi) -> list[dict]:
    seen = {}
    for ds in (api.list_datasets(language="bn", limit=5000)
               + list(api.list_datasets(search="bangla", limit=5000))
               + list(api.list_datasets(search="bengali", limit=5000))):
        seen.setdefault(ds.id, ds)
    rows = sorted(seen.values(), key=lambda d: -(d.downloads or 0))
    return [dict(id=d.id, downloads=d.downloads or 0, likes=d.likes or 0,
                 last_modified=str(d.last_modified or "")) for d in rows]


def slug(ds_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", ds_id)


def download_one(api: HfApi, ds_id: str, out: str, max_workers: int = 4):
    dest = os.path.join(out, slug(ds_id))
    done_flag = os.path.join(dest, ".harvest_ok")
    if os.path.exists(done_flag):
        return "skipped"
    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo_id=ds_id, repo_type="dataset", local_dir=dest,
                      max_workers=max_workers)
    with open(done_flag, "w") as f:
        f.write(ds_id)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/raw")
    ap.add_argument("--top", type=int, default=0, help="only N biggest by downloads")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--kaggle", action="store_true")
    ap.add_argument("--query", default="bangla")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    api = HfApi()

    if args.kaggle:
        from kaggle.api.kaggle_api_extended import KaggleApi
        kapi = KaggleApi(); kapi.authenticate()
        for res in kapi.dataset_list(search=args.query):
            print(f"kaggle: {res.ref}")
            kapi.dataset_download_cli(res.ref, path=args.out,
                                      unzip=True, force=False)
        return

    manifest = build_manifest(api)
    with open(f"{args.out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "downloads", "likes",
                                          "last_modified"])
        w.writeheader()
        w.writerows(manifest)
    print(f"manifest: {len(manifest)} datasets")

    queue = manifest if args.all else manifest[:args.top]
    fails, t0 = [], time.time()
    for i, row in enumerate(queue):
        ds_id = row["id"]
        try:
            status = download_one(api, ds_id, args.out)
            print(f"[{i+1}/{len(queue)}] {status:8s} {ds_id} "
                  f"(dl={row['downloads']})")
        except Exception as e:
            fails.append(dict(id=ds_id, error=f"{e.__class__.__name__}: {e}"))
            print(f"[{i+1}/{len(queue)}] FAIL      {ds_id}: "
                  f"{e.__class__.__name__}")
            with open(f"{args.out}/failures.json", "w") as f:
                json.dump(fails, f, indent=2)
    print(f"\ndone in {(time.time()-t0)/60:.0f} min; "
          f"failures: {len(fails)} (see {args.out}/failures.json)")


if __name__ == "__main__":
    main()
