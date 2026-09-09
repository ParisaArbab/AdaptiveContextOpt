"""Install the pinned official LeanCTX binary locally, without editor hooks."""
import hashlib
import json
import platform
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = '3.10.1'


def main():
    arch = {'arm64': 'aarch64', 'aarch64': 'aarch64',
            'x86_64': 'x86_64', 'AMD64': 'x86_64'}.get(platform.machine())
    system = {'Darwin': 'apple-darwin', 'Linux': 'unknown-linux-gnu'}.get(platform.system())
    if not arch or not system:
        raise SystemExit('Install the official binary for this platform and set LEAN_CTX_BINARY.')
    name = f'lean-ctx-{arch}-{system}.tar.gz'
    base = f'https://github.com/yvgude/lean-ctx/releases/download/v{VERSION}/'
    checksums = urllib.request.urlopen(base + 'SHA256SUMS', timeout=60).read().decode()
    expected = next(line.split()[0] for line in checksums.splitlines()
                    if line.split()[-1].lstrip('*') == name)
    target = Path(__file__).resolve().parents[1] / '.tools' / 'leanctx' / VERSION
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / name
        urllib.request.urlretrieve(base + name, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise SystemExit('Official release checksum mismatch')
        with tarfile.open(archive) as bundle:
            members = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == 'lean-ctx']
            if len(members) != 1:
                raise SystemExit('Unexpected release archive layout')
            binary = bundle.extractfile(members[0]).read()
        target.mkdir(parents=True, exist_ok=True)
        pending = target / 'lean-ctx.new'
        pending.write_bytes(binary)
        pending.chmod(0o755)
        pending.replace(target / 'lean-ctx')
        (target / 'installation.json').write_text(json.dumps({
            'release': f'v{VERSION}', 'asset': base + name, 'sha256': expected}, indent=2))
    print(target / 'lean-ctx')


if __name__ == '__main__':
    main()
