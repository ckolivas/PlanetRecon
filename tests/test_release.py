import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('release',Path(__file__).resolve().parents[1]/'packaging/release.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_inventory_detects_modified_and_extra_files(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'app').write_bytes(b'app')
    out=tmp_path/'inventory'
    report=r.inventory(bundle,out)
    assert report['release_ready'] is False
    assert r.verify(bundle,out/'manifest.json')==1
    (bundle/'app').write_bytes(b'changed')
    with pytest.raises(ValueError,match='content changed'):r.verify(bundle,out/'manifest.json')
    (bundle/'app').write_bytes(b'app');(bundle/'extra').write_bytes(b'extra')
    with pytest.raises(ValueError,match='file set changed'):r.verify(bundle,out/'manifest.json')


def test_bundle_symlinks_must_stay_inside(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'escape').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='escapes'):list(r.bundled_paths(bundle))


def test_frozen_no_argument_launch_opens_gui(monkeypatch):
    import sys
    from planetrecon.cli import main
    import planetrecon.gui.app as gui
    called=[]
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    monkeypatch.setattr(sys,'argv',['planetrecon'])
    monkeypatch.setattr(gui,'main',lambda args:called.append(args) or 0)
    assert main()==0 and called==[[]]


def test_inventory_rejects_untracked_nonfile_links(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'app').write_bytes(b'app')
    out=tmp_path/'inventory';r.inventory(bundle,out)
    link=bundle/'extra';link.symlink_to('missing')
    with pytest.raises(ValueError,match='broken or cyclic'):r.verify(bundle,out/'manifest.json')
    link.unlink();link.symlink_to('.',target_is_directory=True)
    with pytest.raises(ValueError,match='broken or cyclic'):r.verify(bundle,out/'manifest.json')


def test_framework_directory_link_is_recorded_and_verified(tmp_path):
    bundle=tmp_path/'bundle';version=bundle/'Versions/A';version.mkdir(parents=True)
    (version/'binary').write_bytes(b'framework')
    (bundle/'Versions/Current').symlink_to('A',target_is_directory=True)
    out=tmp_path/'inventory';report=r.inventory(bundle,out)
    assert any(f['kind']=='directory_symlink' for f in report['files'])
    assert r.verify(bundle,out/'manifest.json')==2
