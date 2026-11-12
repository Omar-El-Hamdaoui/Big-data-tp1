from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, os, time, json, zipfile, shutil
root=Path(__file__).resolve().parents[1]
out=root/'data/raw/accidents/US_Accidents_March23.csv'
archive=out.with_suffix('.zip.part')
state=archive.with_suffix('.ranges.json')
url='https://www.kaggle.com/api/v1/datasets/download/sobhanmoosavi/us-accidents?datasetVersionNumber=13'
total=684855912
if out.exists(): print('CSV already present'); raise SystemExit
if state.exists(): info=json.loads(state.read_text())
else:
 start=archive.stat().st_size if archive.exists() else 0
 ranges=[(a,min(a+8*1024*1024-1,total-1)) for a in range(start,total,8*1024*1024)]
 info={'initial':start,'ranges':ranges,'done':[]}
 state.write_text(json.dumps(info))
fd=os.open(archive,os.O_CREAT|os.O_RDWR)
os.ftruncate(fd,total)
def download(index,limits):
 a,b=limits
 for attempt in range(5):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
   with urllib.request.urlopen(req,timeout=90) as response:
    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {a}-{b}/{total}':
     raise ValueError('Incorrect Range response')
    pos=a
    while chunk:=response.read(256*1024):
     os.pwrite(fd,chunk,pos);pos+=len(chunk)
    if pos!=b+1: raise ValueError('Short chunk')
   return index
  except Exception:
   if attempt==4: raise
   time.sleep(1+attempt)
with ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(download,i,limits) for i,limits in enumerate(info['ranges']) if i not in info['done']]
 for future in as_completed(futures):
  info['done'].append(future.result());state.write_text(json.dumps(info))
  done=info['initial']+sum(info['ranges'][i][1]-info['ranges'][i][0]+1 for i in info['done'])
  print(f'Download {done/1e6:.0f}/{total/1e6:.0f} MB',flush=True)
os.close(fd)
print('Extraction',flush=True)
with zipfile.ZipFile(archive) as z, z.open(out.name) as src, out.with_suffix('.csv.part').open('wb') as dst:
 shutil.copyfileobj(src,dst,1024*1024)
out.with_suffix('.csv.part').rename(out)
archive.unlink(); state.unlink()
print('SUCCESS',out.stat().st_size,flush=True)
