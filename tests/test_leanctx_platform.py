import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('leanctx_installer', Path(__file__).resolve().parents[1] / 'scripts/install_leanctx.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.mark.parametrize('system,machine,libc,expected', [
    ('Darwin', 'arm64', '', 'aarch64-apple-darwin'),
    ('Darwin', 'x86_64', '', 'x86_64-apple-darwin'),
    ('Linux', 'x86_64', 'glibc', 'x86_64-unknown-linux-gnu'),
    ('Linux', 'aarch64', 'glibc', 'aarch64-unknown-linux-gnu'),
    ('Linux', 'x86_64', 'musl', 'x86_64-unknown-linux-musl'),
])
def test_platform_selects_matching_official_binary(system, machine, libc, expected):
    assert installer.release_asset(system, machine, libc) == f'lean-ctx-{expected}.tar.gz'


def test_unsupported_architecture_does_not_install_wrong_binary():
    with pytest.raises(ValueError):
        installer.release_asset('Linux', 'riscv64')
