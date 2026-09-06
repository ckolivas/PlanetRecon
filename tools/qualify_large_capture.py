"""Full streaming run of a generated SER larger than its Linux CPU memory cap.

Flat zero frames test bounded input/process mechanics, not reconstruction quality.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from planetrecon.io.ser import write_ser
from planetrecon.result import load_snapshot


def main(out,executable=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    frames,side,cap=4096,512,512*1024**2
    source=write_ser(out/'generated-large.ser',np.zeros((1,side,side),dtype='u2'))
    with source.open('r+b') as stream:
        stream.seek(38);stream.write(struct.pack('<I',frames))
        stream.truncate(178+frames*side*side*2)
    command=[str(Path(executable).resolve())] if executable else [sys.executable,'-m','planetrecon']
    command+=['--threads','2','stack','--device','cpu','--cpu-memory-mib','512',
              '--path',str(source),'--out',str(out/'stack')]
    started=time.monotonic()
    with (out/'run.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    elapsed=time.monotonic()-started
    result=load_snapshot(out/'stack/stack.npz')
    budget=result.provenance['cpu_memory_budget']
    assert not result.incomplete and result.n_used==frames and result.n_rejected==0
    assert budget['enforced'] and budget['effective_bytes']<=cap
    assert budget['process_peak_rss_bytes']<=cap and source.stat().st_size>cap
    assert result.validity.all() and not result.image.any()
    report={'status':'passed','scope':'generated flat SER; input/memory/lifetime mechanics only',
            'capture_bytes':source.stat().st_size,'frames':frames,'shape':[side,side],
            'wall_s':elapsed,'cpu_memory_budget':budget,'frames_used':result.n_used,
            'standalone':executable is not None}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--executable',type=Path)
    args=parser.parse_args()
    main(args.out,args.executable)
