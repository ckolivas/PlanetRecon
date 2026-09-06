"""Verify every declared target before publishing a version-tag GitHub release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build_native import ROOT, version_from_tag
from release import digest


def verified_assets(directory,tag,revision=None):
    version_from_tag(tag)
    directory=Path(directory)
    expected={item['id'] for item in json.loads((ROOT/'packaging/targets.json').read_text())}
    reports=[json.loads(path.read_text()) for path in directory.glob('*.release.json')]
    targets=[report['target'] for report in reports]
    if set(targets)!=expected or len(targets)!=len(expected):
        raise ValueError('missing, duplicate or unknown native build targets; release will not publish')
    if len({report['revision'] for report in reports})!=1:
        raise ValueError('native assets were built from different revisions')
    if revision is None:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if reports[0]['revision']!=revision:
        raise ValueError('native assets do not match the tagged checkout revision')
    allowed=set()
    for report in reports:
        if report['tag']!=tag or report['version']!=version_from_tag(tag):
            raise ValueError('asset version does not match the triggering tag')
        stem=f"planetrecon-{report['version']}-{report['target']}"
        descriptor=directory/(stem+'.release.json')
        checksum=directory/(stem+'.sha256')
        if not descriptor.is_file() or not checksum.is_file():
            raise ValueError('missing build descriptor or checksum file')
        named={item['name'] for item in report['assets']}
        if not {stem+'.manifest.json',stem+'.sbom.cdx.json'}.issubset(named):
            raise ValueError('missing bundle inventory or SBOM')
        if not set(report['archive_parts']).issubset(named):
            raise ValueError('archive parts are missing from the asset manifest')
        for item in report['assets']:
            name=item['name']
            if Path(name).name!=name or name in allowed:
                raise ValueError('unsafe or duplicate release asset name')
            path=directory/name
            if not path.is_file() or path.stat().st_size>=2*1024**3 or path.stat().st_size!=item['size'] or digest(path)!=item['sha256']:
                raise ValueError(f'release asset integrity mismatch: {name}')
            allowed.add(name)
        if not report['archive_parts'] or len(set(report['archive_parts']))!=len(report['archive_parts']):
            raise ValueError('empty or duplicate archive parts')
        archive_hash=hashlib.sha256();archive_size=0
        for name in report['archive_parts']:
            with (directory/name).open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    archive_hash.update(block);archive_size+=len(block)
        if archive_hash.hexdigest()!=report['archive']['sha256'] or archive_size!=report['archive']['size']:
            raise ValueError('reassembled archive integrity mismatch')
        checks={}
        for line in checksum.read_text().splitlines():
            sha,name=line.split('  ',1)
            if name in checks:raise ValueError('duplicate checksum entry')
            checks[name]=sha
        if set(checks)!=named|{descriptor.name}:
            raise ValueError('checksum file set mismatch')
        for name,sha in checks.items():
            if digest(directory/name)!=sha:raise ValueError(f'checksum mismatch: {name}')
        allowed.update((descriptor.name,checksum.name))
    if {p.name for p in directory.iterdir()}!=allowed:
        raise ValueError('unexpected files in release staging directory')
    return sorted(directory/name for name in allowed)


def publish(directory,tag,repository):
    files=verified_assets(directory,tag)
    notes=Path(directory).parent/'RELEASE_NOTES.md'
    notes.write_text(f'PlanetRecon {tag}\n\n'
        'Native CPU bundles: Linux x64, Windows x64, macOS Intel and Apple Silicon. '
        'The Linux CUDA bundle additionally includes Torch and CUDA user-space libraries.\n\n'
        'Download the archive for your OS and keep the complete extracted bundle together. '
        'Python/Qt/numerical/image components are included. CUDA needs a compatible NVIDIA driver.\n\n'
        'For split Linux archives, download all numbered parts and concatenate them in order: '
        '`cat NAME.tar.gz.part* > NAME.tar.gz`. Original archive and individual part SHA-256 '
        'hashes are recorded in the corresponding `.release.json` and `.sha256` files.\n\n'
        'Linux CPU workflow checks run before publication. Windows/macOS runtime testing '
        'is excluded from this process; these are build artifacts, not a clean-system certification. '
        'The macOS app is ad-hoc signed; no publisher signing/notarization is configured. '
        'The advanced atmospheric solver remains experimental and Q3 unqualified.\n')
    command=['gh','release','create',tag,'--repo',repository,'--verify-tag','--draft',
             '--title',f'PlanetRecon {tag}','--notes-file',str(notes)]
    if '-' in tag:command.append('--prerelease')
    subprocess.run(command,check=True)
    # Keep a failed upload as a draft. Publish only after every verified asset is uploaded.
    subprocess.run(['gh','release','upload',tag,'--repo',repository,*map(str,files)],check=True)
    subprocess.run(['gh','release','edit',tag,'--repo',repository,'--draft=false'],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--repository',required=True)
    args=parser.parse_args()
    publish(args.assets,args.tag,args.repository)
