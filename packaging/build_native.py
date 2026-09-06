"""Build a versioned native bundle and checksum-indexed GitHub release assets."""
import argparse
from contextlib import contextmanager
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile

from packaging.requirements import Requirement

sys.path.insert(0,str(Path(__file__).resolve().parent))
import release

ROOT=Path(__file__).resolve().parents[1]
ASSET_LIMIT=2*1024**3


def version_from_tag(tag):
    if not re.fullmatch(r'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-(?:alpha|beta|rc)\.[1-9][0-9]*)?',tag):
        raise ValueError('tag must be vMAJOR.MINOR.PATCH, optionally -alpha.N, -beta.N or -rc.N')
    return tag[1:]


def target_id(flavor):
    system={'linux':'linux','win32':'windows','darwin':'macos'}.get(sys.platform)
    arch={'AMD64':'x86_64','x86_64':'x86_64','arm64':'arm64','aarch64':'arm64'}.get(platform.machine())
    target=f'{system}-{arch}-{flavor}'
    if target not in {item['id'] for item in json.loads((ROOT/'packaging/targets.json').read_text())}:
        raise ValueError(f'no declared native build target: {target}')
    return target


def check_requirements(flavor):
    if platform.python_version()!='3.13.5':
        raise ValueError('native build recipes require Python 3.13.5')
    paths=[ROOT/'requirements/build-native.lock']
    if flavor=='gpu':paths.extend([ROOT/'requirements/build-gpu-python.lock',ROOT/'requirements/gpu-cu132.lock'])
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith(('#','--')):continue
            req=Requirement(line)
            if req.marker is None or req.marker.evaluate():
                if metadata.version(req.name) not in req.specifier:
                    raise ValueError(f'installed build dependency does not match {line}')


@contextmanager
def build_version(version):
    path=ROOT/'planetrecon/_build_version.py'
    # Separate runner checkouts permit parallel native builds; never overwrite a
    # manual version file or a concurrent build's stamp in the same checkout.
    for cache in (path.parent/'__pycache__').glob('_build_version.*.pyc'):cache.unlink()
    with path.open('x') as stream:stream.write(f'VERSION = {version!r}\n')
    try:yield
    finally:
        path.unlink()
        for cache in (path.parent/'__pycache__').glob('_build_version.*.pyc'):cache.unlink()


def split_asset(path,limit=ASSET_LIMIT,part_size=1024**3):
    path=Path(path)
    if path.stat().st_size < limit:return [path]
    parts=[]
    with path.open('rb') as source:
        while source.tell()<path.stat().st_size:
            part=path.with_name(path.name+f'.part{len(parts)+1:03d}')
            with part.open('xb') as dest:
                remaining=part_size
                while remaining:
                    block=source.read(min(8*1024**2,remaining))
                    if not block:break
                    dest.write(block);remaining-=len(block)
            parts.append(part)
    path.unlink()
    return parts


def build(tag,flavor,out,skip_smoke=False):
    version=version_from_tag(tag);target=target_id(flavor);check_requirements(flavor)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=out/'assets';assets.mkdir()
    env={**os.environ,'PLANETRECON_BUNDLE_GPU':'1' if flavor=='gpu' else '0',
         'PLANETRECON_BUILD_VERSION':version,'PLANETRECON_THREADS':'2','QT_QPA_PLATFORM':'offscreen'}
    with build_version(version):
        subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean',
                        '--distpath',str(out/'dist'),'--workpath',str(out/'build'),
                        str(ROOT/'packaging/planetrecon.spec')],cwd=ROOT,env=env,check=True)
        name='planetrecon-gpu' if flavor=='gpu' else 'planetrecon'
        bundle=out/'dist'/('PlanetRecon.app' if sys.platform=='darwin' else name)
        executable=bundle/('Contents/MacOS/planetrecon' if sys.platform=='darwin' else name+('.exe' if sys.platform=='win32' else ''))
        smoke='not run (Windows/macOS runtime testing excluded by request)'
        if sys.platform=='linux' and not skip_smoke:
            subprocess.run([str(executable),'gui-smoke','--out',str(out/'gui-smoke')],env=env,check=True,timeout=60)
            for script in ('smoke_export.py','smoke_capture.py'):
                subprocess.run([sys.executable,str(ROOT/'packaging'/script),str(executable)],env=env,check=True,timeout=180)
            actual=subprocess.check_output([str(executable),'--version'],env=env,text=True).strip()
            if actual!=f'PlanetRecon {version}':raise ValueError('frozen version stamp mismatch')
            smoke='Linux CPU GUI, checkpoint resume, encoders, native AVI and geometry passed; hardware CUDA tested separately'
        elif sys.platform=='linux':smoke='not run (explicit skip)'
        quickstart=(bundle/'Contents/Resources' if sys.platform=='darwin' else bundle)/'QUICKSTART.txt'
        quickstart.write_text('PlanetRecon '+version+'\n\nRun the app with no arguments to open Qt. Copy the whole bundle.\n'
            'Open a SER or uncompressed AVI; choose CPU (or GPU in the Linux CUDA bundle).\n'
            'Bayer input previews use nearest-neighbour RGB; reconstruction consumes raw CFA.\n'
            'Select calibration/geometry only from known capture information, then Run.\n'
            'Optional accumulator checkpoint and Resume controls preserve completed batches.\n'
            'Float TIFF preserves scale. Integer PNG/TIFF needs explicit black/white levels.\n'
            'Python, Qt and numerical/image components are bundled. The native OS graphics\n'
            'stack and a compatible NVIDIA driver for CUDA remain host requirements.\n'
            'Advanced atmospheric inference remains experimental; Q3 is not qualified.\n'
            'Windows/macOS runtime validation was excluded from this release process.\n'
            'Builds have no publisher signature or notarization; macOS uses ad-hoc signing.\n')
        toc=out/'build/planetrecon/Analysis-00.toc'
        notices=release.notices(bundle,toc)
        if sys.platform=='darwin':
            subprocess.run(['codesign','--force','--deep','--sign','-',str(bundle)],check=True)
        inventory_dir=out/'inventory'
        manifest=release.inventory(bundle,inventory_dir,notice_records=notices)
        release.verify(bundle,inventory_dir/'manifest.json')
        stem=f'planetrecon-{version}-{target}'
        copied=[]
        for src,suffix in [('manifest.json','.manifest.json'),('sbom.cdx.json','.sbom.cdx.json')]:
            dest=assets/(stem+suffix);shutil.copyfile(inventory_dir/src,dest);copied.append(dest)
        if sys.platform=='win32':
            archive=Path(shutil.make_archive(str(assets/stem),'zip',root_dir=bundle.parent,base_dir=bundle.name))
        else:
            archive=assets/(stem+'.tar.gz')
            with tarfile.open(archive,'w:gz',compresslevel=6) as stream:stream.add(bundle,arcname=bundle.name)
        archive_identity={'name':archive.name,'size':archive.stat().st_size,'sha256':release.digest(archive)}
        parts=split_asset(archive)
        all_assets=copied+parts
        records=[{'name':p.name,'size':p.stat().st_size,'sha256':release.digest(p)} for p in all_assets]
        result={'tag':tag,'version':version,'target':target,'revision':manifest['base_revision'],
                'native_host':platform.platform(),'libc':platform.libc_ver(),
                'runtime_validation':smoke,'publisher_signed':False,'macos_ad_hoc_signed':sys.platform=='darwin',
                'archive':archive_identity,'archive_parts':[p.name for p in parts],'assets':records}
        descriptor=assets/(stem+'.release.json');descriptor.write_text(json.dumps(result,indent=2)+'\n')
        checksums=records+[{'name':descriptor.name,'sha256':release.digest(descriptor)}]
        (assets/(stem+'.sha256')).write_text(''.join(f"{item['sha256']}  {item['name']}\n" for item in checksums))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--flavor',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--skip-smoke',action='store_true')
    args=parser.parse_args()
    print(json.dumps(build(args.tag,args.flavor,args.out,args.skip_smoke)))
