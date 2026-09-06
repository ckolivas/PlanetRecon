"""Linux candidate build checks, bundled notices, SBOM and artifact integrity.

This produces an unsigned local candidate, not clean-system release acceptance.
"""
import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def toolchain():
    return {'system':platform.system(),'machine':platform.machine(),'python':platform.python_version(),
            'packages':dict(sorted((dist.metadata['Name'],dist.version) for dist in metadata.distributions() if dist.metadata['Name'].lower()!='planetrecon'))}


def check_toolchain():
    expected=json.loads((ROOT/'packaging/toolchain-linux.json').read_text())
    actual=toolchain()
    if actual != expected:
        raise ValueError(f'build toolchain differs from the recorded lock: {actual}')
    return actual


def bundled_paths(bundle):
    bundle=Path(bundle).resolve()
    for path in sorted(bundle.rglob('*')):
        if path.is_symlink() and not path.resolve().is_relative_to(bundle):
            raise ValueError(f'bundle link escapes its directory: {path}')
        if path.is_symlink():
            if not path.exists() or (path.is_dir() and path.parent.resolve().is_relative_to(path.resolve())):
                raise ValueError(f'bundle link is broken or cyclic: {path}')
            yield path
        elif path.is_file():yield path


def file_record(path,bundle):
    link=str(path.readlink()) if path.is_symlink() else None
    directory_link=path.is_symlink() and path.is_dir()
    return {'path':path.relative_to(bundle).as_posix(),
            'size':len(link.encode()) if directory_link else path.stat().st_size,
            'sha256':hashlib.sha256(link.encode()).hexdigest() if directory_link else digest(path),
            'symlink':link,'kind':'directory_symlink' if directory_link else 'file'}


def notices(bundle,toc):
    data=ast.literal_eval(Path(toc).read_text())
    sources=set()
    def walk(value):
        if isinstance(value,(list,tuple)):
            if len(value)==3 and all(isinstance(v,str) for v in value) and value[2] in ('BINARY','EXTENSION','PYMODULE','PYSOURCE','DATA'):
                if Path(value[1]).is_file():sources.add(value[1])
            else:
                for item in value:walk(item)
    walk(data)
    owners=set()
    ordered=sorted(sources)
    for start in range(0,len(ordered) if shutil.which('dpkg-query') else 0,200):
        result=subprocess.run(['dpkg-query','-S',*ordered[start:start+200]],capture_output=True,text=True)
        for line in result.stdout.splitlines():
            if ': ' in line:
                owner=line.rsplit(': ',1)[0]
                if not owner.startswith(('diversion','local diversion')):
                    owners.update(p.split(':')[0] for p in owner.split(', '))
    target=Path(bundle)/('Contents/Resources/licenses' if Path(bundle).suffix=='.app' else 'licenses')
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    python_license=Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.is_file():
        dest=target/'Python-LICENSE.txt';dest.write_bytes(python_license.read_bytes())
        records.append({'provider':'Python','path':dest.relative_to(bundle).as_posix()})
    for project_license in (ROOT/'LICENSE',ROOT/'LICENSE.txt',ROOT/'LICENSE.md'):
        if project_license.is_file():
            dest=target/f'PlanetRecon-{project_license.name}';dest.write_bytes(project_license.read_bytes())
            records.append({'provider':'PlanetRecon','path':dest.relative_to(bundle).as_posix()})
    for owner in sorted(owners):
        path=Path('/usr/share/doc')/owner/'copyright'
        if path.is_file():
            dest=target/f'debian-{owner}.txt';dest.write_bytes(path.read_bytes())
            records.append({'provider':owner,'path':dest.relative_to(bundle).as_posix()})
    for name in toolchain()["packages"]:
        dist=metadata.distribution(name)
        for item in dist.files or []:
            if (Path(item).name.lower().startswith(('license','copying')) or
                    any(part.lower() in ('licenses','licences','license') for part in Path(item).parts)):
                path=Path(dist.locate_file(item))
                if path.is_file():
                    dest=target/f'{name}-{digest(path)[:12]}.txt';dest.write_bytes(path.read_bytes())
                    records.append({'provider':name,'path':dest.relative_to(bundle).as_posix()})
    (target/'index.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


def inventory(bundle,out,toc=None,notice_records=None):
    bundle=Path(bundle).resolve();out=Path(out).resolve()
    if out==bundle or out.is_relative_to(bundle):
        raise ValueError('inventory must be outside the bundle')
    out.mkdir(parents=True,exist_ok=False)
    license_records=notice_records if notice_records is not None else (notices(bundle,toc) if toc else [])
    files=[]
    for path in bundled_paths(bundle):
        files.append(file_record(path,bundle))
    source_files=sorted([*ROOT.glob('planetrecon/**/*.py'),*ROOT.glob('packaging/*.py'),*ROOT.glob('packaging/*.spec'),*ROOT.glob('requirements/*.lock'),*ROOT.glob('tools/*.py'),*ROOT.glob('.github/workflows/*.yml'),*ROOT.glob('packaging/*.json'),ROOT/'pyproject.toml'])
    source_identity={p.relative_to(ROOT).as_posix():digest(p) for p in source_files}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'base_revision':revision,'source_files':source_identity,'schema':'planetrecon-local-release-candidate-1','status':'incomplete','release_ready':False,
        'toolchain':toolchain(),'files':files,'notices':license_records,
        'acceptance':{'linux_development_host':'smoke tests recorded separately','linux_clean_offline':'not run',
            'windows_clean_offline':'not run','macos_clean_offline':'not run','supported_gpu':'smoke tests recorded separately',
            'signing_notarization':'not performed','project_license':'not specified'},
        'scope':'file SBOM and bundled notices; licensing/signing/clean-system acceptance not certified'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sbom={'bomFormat':'CycloneDX','specVersion':'1.5','version':1,'components':[
        {'type':'file','bom-ref':f['path'],'name':f['path'],'hashes':[{'alg':'SHA-256','content':f['sha256']}]}
        for f in files]}
    (out/'sbom.cdx.json').write_text(json.dumps(sbom,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in files))
    return manifest


def verify(bundle,manifest):
    bundle=Path(bundle).resolve()
    expected=json.loads(Path(manifest).read_text())['files']
    actual={p.relative_to(bundle).as_posix():p for p in bundled_paths(bundle)}
    if set(actual)!={f['path'] for f in expected}:
        raise ValueError('bundle file set changed')
    for record in expected:
        path=actual[record['path']]
        current=file_record(path,bundle)
        if any(current[key]!=value for key,value in record.items()):
            raise ValueError(f'bundle content changed: {record["path"]}')
    return len(actual)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check-toolchain')
    inv=sub.add_parser('inventory');inv.add_argument('--bundle',type=Path,required=True);inv.add_argument('--out',type=Path,required=True);inv.add_argument('--toc',type=Path,required=True)
    ver=sub.add_parser('verify');ver.add_argument('--bundle',type=Path,required=True);ver.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='check-toolchain':print(json.dumps(check_toolchain()))
    elif args.command=='inventory':print(f"inventoried {len(inventory(args.bundle,args.out,args.toc)['files'])} files")
    else:print(f'verified {verify(args.bundle,args.manifest)} files')
