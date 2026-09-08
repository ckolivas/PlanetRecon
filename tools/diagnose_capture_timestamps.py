"""Read-only bounded duplicate-timestamp pixel checks on hashed private captures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from planetrecon.io.ser import SERSource
from tools.study_io import file_hash,write_json,identities


def diagnose(path, expected_hash, *, max_pairs=64):
    if int(max_pairs)!=max_pairs or max_pairs<1: raise ValueError('positive integer pair cap required')
    path=Path(path);before=path.stat()
    if file_hash(path)!=expected_hash: raise ValueError('capture does not match hashed intake')
    source=SERSource(path)
    try:
        stamps=source.timestamps();pairs=[]
        duplicates=np.empty(0,dtype=int) if stamps is None else np.flatnonzero(stamps[1:]==stamps[:-1])
        indices=np.linspace(0,len(duplicates)-1,min(len(duplicates),int(max_pairs)),dtype=int) if len(duplicates) else []
        for selected in indices:
            i=int(duplicates[selected])
            a=source.read_raw(i);b=source.read_raw(i+1)
            pairs.append({'first_frame':i,'second_frame':i+1,'pixels_identical':bool(np.array_equal(a,b))})
    finally:
        source.close()
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('capture changed during diagnosis')
    return {'capture_sha256':expected_hash,'duplicate_intervals':len(duplicates),'sampled_pairs':pairs,
            'sampled_identical_pairs':sum(p['pixels_identical'] for p in pairs),
            'timestamps_present':stamps is not None,'source_stat_unchanged':True,
            'timing_repaired':False,'frames_removed':False,
            'scope':'Uniform bounded sample of adjacent equal-timestamp pairs; no conclusion about unsampled pairs, independent observations or cadence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-spec',type=Path,required=True);p.add_argument('--intake',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise ValueError('use a new diagnostic output')
    identity=identities([Path(__file__)]);intake_hash=file_hash(a.intake)
    paths={r['id']:r['path'] for r in json.loads(a.private_spec.read_text())}
    intake=json.loads(a.intake.read_text());records=[]
    for row in intake['records']:
        if row['timing']['status']=='invalid':
            records.append({'id':row['id'],**diagnose(paths[row['id']],row['capture_sha256'])})
    if identity!=identities([Path(__file__)]) or intake_hash!=file_hash(a.intake):
        raise ValueError('source or intake changed during diagnosis')
    report={'records':records,'max_pairs_per_capture':64,'intake_sha256':intake_hash,
            'generator_identity':identity,'scientific_qualification':False,'source_input_unchanged':True}
    a.out.parent.mkdir(parents=True,exist_ok=True);write_json(a.out,report)
