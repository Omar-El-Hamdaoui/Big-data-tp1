"""Exécuter et sauvegarder les sorties, même si une cellule échoue."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
p=root/sys.argv[1]
nb=nbformat.read(p,as_version=4)
def started(cell,cell_index,**kwargs):
    first=cell.source.splitlines()[0] if cell.source else ''
    print(f'Cell {cell_index}: {first[:100]}',flush=True)
client=NotebookClient(nb,timeout=1800,kernel_name='tp3',resources={'metadata':{'path':str(root)}},on_cell_start=started)
try:
    client.execute()
finally:
    nbformat.write(nb,p)
print('SUCCESS',p.name,flush=True)
