from pathlib import Path
import gzip,hashlib,json,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from tp3.common import acquire
p=root/'data/raw/accidents/US_Accidents_March23.csv'
if p.exists():
    dest=p.with_suffix('.csv.gz');temp=dest.with_suffix('.gz.part')
    digest=hashlib.sha256();size=0
    with p.open('rb') as src,gzip.open(temp,'wb',compresslevel=1) as dst:
        for block in iter(lambda:src.read(8*1024*1024),b''):
            digest.update(block);dst.write(block);size+=len(block)
    check=hashlib.sha256();verified_size=0
    with gzip.open(temp,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            check.update(block);verified_size+=len(block)
    assert size==verified_size==3058183727 and check.digest()==digest.digest()
    temp.rename(dest)
    p.unlink()  # Copie téléchargée par ce TP ; contenu préservé et vérifié dans gzip.
    print('Compressed and SHA-256 verified:',dest.stat().st_size,flush=True)
print(acquire('accidents'),flush=True)
