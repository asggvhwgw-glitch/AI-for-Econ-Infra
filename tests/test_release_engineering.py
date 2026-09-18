"""Tests for local release helpers (network/OS CI success is not mocked here)."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_clean_install_wheel_selection_requires_one_file(tmp_path):
    m=script('verify_clean_install')
    with pytest.raises(ValueError):m.select_wheel(None,tmp_path)
    a=tmp_path/'econhdfe-1-py3-none-any.whl';a.write_bytes(b'a')
    assert m.select_wheel(None,tmp_path)==a.resolve()
    (tmp_path/'econhdfe-2-py3-none-any.whl').write_bytes(b'b')
    with pytest.raises(ValueError):m.select_wheel(None,tmp_path)

@pytest.mark.parametrize('mode',['valid','unlisted','duplicate','unsafe','wrong'])
def test_release_checksum_manifest_coverage(tmp_path,mode):
    m=script('verify_release');a=tmp_path/'a.txt';a.write_text('ok')
    line=f'{m.digest(a)}  a.txt\n'
    if mode=='unlisted':(tmp_path/'other.txt').write_text('not hashed')
    if mode=='duplicate':line*=2
    if mode=='unsafe':line=line.replace('a.txt','../a.txt')
    if mode=='wrong':line='0'*64+'  a.txt\n'
    (tmp_path/'SHA256SUMS.txt').write_text(line)
    if mode=='valid':assert m.verify_checksums(tmp_path)==1
    else:
        with pytest.raises(ValueError):m.verify_checksums(tmp_path)
