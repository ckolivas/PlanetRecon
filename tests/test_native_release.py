import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_build',ROOT/'packaging/build_native.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
import publish


@pytest.mark.parametrize('tag',['v1.2.3','v0.0.1','v1.2.3-rc.1'])
def test_version_tags(tag):
    assert build.version_from_tag(tag)==tag[1:]


@pytest.mark.parametrize('tag',['1.2.3','v01.2.3','v1.2','v1.2.3; echo broken','../v1.2.3'])
def test_invalid_tags_are_rejected(tag):
    with pytest.raises(ValueError):build.version_from_tag(tag)


def test_version_stamp_is_removed_after_failed_build(tmp_path,monkeypatch):
    monkeypatch.setattr(build,'ROOT',tmp_path)
    (tmp_path/'planetrecon').mkdir()
    with pytest.raises(RuntimeError):
        with build.build_version('1.2.3'):
            assert '1.2.3' in (tmp_path/'planetrecon/_build_version.py').read_text()
            raise RuntimeError('build failed')
    assert not (tmp_path/'planetrecon/_build_version.py').exists()


def test_oversized_archive_parts_reassemble_exactly(tmp_path):
    path=tmp_path/'archive.tar.gz';content=bytes(range(251))*7;path.write_bytes(content)
    parts=build.split_asset(path,limit=1000,part_size=512)
    assert not path.exists() and len(parts)==4
    assert all(part.stat().st_size<=512 for part in parts)
    assert b''.join(part.read_bytes() for part in parts)==content


def release_set(directory):
    for target in json.loads((ROOT/'packaging/targets.json').read_text()):
        stem='planetrecon-1.2.3-'+target['id']
        archive=directory/(stem+'.tar.gz');archive.write_bytes(target['id'].encode())
        record={'name':archive.name,'size':archive.stat().st_size,'sha256':build.release.digest(archive)}
        records=[record]
        for suffix in ('.manifest.json','.sbom.cdx.json'):
            path=directory/(stem+suffix);path.write_text('{}')
            records.append({'name':path.name,'size':path.stat().st_size,'sha256':build.release.digest(path)})
        report={'tag':'v1.2.3','version':'1.2.3','target':target['id'],'revision':'abc',
                'assets':records,'archive_parts':[archive.name],'archive':record}
        descriptor=directory/(stem+'.release.json');descriptor.write_text(json.dumps(report))
        checksum=directory/(stem+'.sha256')
        checksum.write_text(''.join(f"{item['sha256']}  {item['name']}\n" for item in records)+f"{build.release.digest(descriptor)}  {descriptor.name}\n")


def test_publish_requires_all_targets_and_exact_asset_integrity(tmp_path):
    release_set(tmp_path)
    assert len(publish.verified_assets(tmp_path,'v1.2.3',revision='abc'))==25
    with pytest.raises(ValueError,match='checkout revision'):
        publish.verified_assets(tmp_path,'v1.2.3',revision='different')
    archive=next(tmp_path.glob('*.tar.gz'));archive.write_bytes(b'damaged')
    with pytest.raises(ValueError,match='integrity mismatch'):
        publish.verified_assets(tmp_path,'v1.2.3',revision='abc')
    next(tmp_path.glob('*.release.json')).unlink()
    with pytest.raises(ValueError,match='native build targets'):
        publish.verified_assets(tmp_path,'v1.2.3',revision='abc')


def test_publish_rejects_missing_inventory(tmp_path):
    release_set(tmp_path)
    descriptor=next(tmp_path.glob('*.release.json'))
    report=json.loads(descriptor.read_text())
    report['assets']=[item for item in report['assets'] if not item['name'].endswith('.sbom.cdx.json')]
    descriptor.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='inventory or SBOM'):
        publish.verified_assets(tmp_path,'v1.2.3',revision='abc')
