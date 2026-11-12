from pathlib import Path
import urllib.request, zipfile, time, hashlib, json
root=Path(__file__).resolve().parents[1]
out=root/'data/raw/accidents/US_Accidents_March23.csv'
archive=out.with_suffix('.zip.part')
url='https://www.kaggle.com/api/v1/datasets/download/sobhanmoosavi/us-accidents?datasetVersionNumber=13'
if not out.exists():
    start=time.monotonic(); last=start
    with urllib.request.urlopen(url, timeout=120) as r, archive.open('wb') as f:
        total=int(r.headers.get('Content-Length',0)); done=0
        while chunk:=r.read(1024*1024):
            f.write(chunk); done+=len(chunk)
            if time.monotonic()-last>25:
                print(f'Download {done/1e6:.0f}/{total/1e6:.0f} MB',flush=True); last=time.monotonic()
    with zipfile.ZipFile(archive) as z:
        info=z.getinfo(out.name)
        print('Extraction',info.file_size,flush=True)
        with z.open(info) as src, out.with_suffix('.csv.part').open('wb') as dst:
            import shutil; shutil.copyfileobj(src,dst,1024*1024)
    out.with_suffix('.csv.part').rename(out)
    archive.unlink()
print(out,out.stat().st_size,flush=True)
with out.open() as f: print(f.readline(),flush=True)
