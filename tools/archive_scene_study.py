"""Archive study JSON without private image/checkpoint payloads; never rewrite outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def archive(source,destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for path in sorted(source.glob('*.json')):
        if path.name.startswith('attempt-') or path.name=='progress.json':
            continue
        shutil.copyfile(path,destination/path.name)
    report=json.loads((destination/'report.json').read_text())
    if report.get('source_input_unchanged') is not True:
        raise ValueError('study does not retain unchanged source/input identities')
    checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob('*.json'))}
    (destination/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();archive(a.source,a.destination)
