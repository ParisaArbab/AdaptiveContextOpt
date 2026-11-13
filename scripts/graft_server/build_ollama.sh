#!/usr/bin/env bash
# User-invoked, out-of-tree build of the pinned Linux serving and audit executable.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# == 2 && "$(uname -s)" == Linux ]] || { echo 'Usage: build_ollama.sh CLEAN_PINNED_OLLAMA_CHECKOUT NEW_OUTPUT_DIR' >&2; exit 2; }
src="$(cd "$1" && pwd)"
out="$(realpath -m "$2")"
pin=9b0c7cc7b90562b370ce6a30efdc667326799223
[[ "$(git -C "$src" rev-parse HEAD)" == "$pin" && -z "$(git -C "$src" status --porcelain)" ]] || { echo 'Need a clean pinned Ollama checkout' >&2; exit 2; }
command -v nvcc >/dev/null
mkdir "$out"
git -C "$src" archive "$pin" | tar -xf - -C "$out"
cp "$root/scripts/graft_server/native_tokenizer.go" "$out/graft_tokenizer.go"
python3 - "$out/main.go" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1])
s=p.read_text()
assert s.count("func main() {")==1
s=s.replace('"context"','"context"\n "os"')
s=s.replace("func main() {",'''func main() {
 if len(os.Args) > 1 && os.Args[1] == "graft-tokenize" {
   cobra.CheckErr(graftTokenize())
   return
 }''')
p.write_text(s)
PY
cd "$out"
gofmt -w main.go graft_tokenizer.go
export GOTOOLCHAIN=local
go mod download
cmake --preset CPU
cmake --build --preset CPU --parallel "${GRAFT_BUILD_JOBS:-4}"
cmake --install build --component CPU --prefix "$out/dist"
cmake --preset 'CUDA 12' -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build --preset 'CUDA 12' --parallel "${GRAFT_BUILD_JOBS:-4}"
cmake --install build --component CUDA --prefix "$out/dist"
mkdir -p dist/bin
CGO_ENABLED=1 go build -mod=readonly -trimpath -ldflags='-X github.com/ollama/ollama/version.Version=0.17.7 -X github.com/ollama/ollama/server.mode=release' -o dist/bin/ollama .
test -x dist/bin/ollama
./dist/bin/ollama --help >/dev/null
# Bind the installed library inventory, source pin and audit source to the executable.
python3 - "$root" <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
dist=Path("dist")
def h(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
receipt={"schema":"graft-ollama-linux-build-v1","source_commit":"9b0c7cc7b90562b370ce6a30efdc667326799223",
         "audit_source_sha256":h(root/"scripts/graft_server/native_tokenizer.go"),
         "build_script_sha256":h(root/"scripts/graft_server/build_ollama.sh"),
         "tools":{name:subprocess.check_output(command,text=True,stderr=subprocess.STDOUT).strip()
                  for name,command in {"go":["go","version"],"cmake":["cmake","--version"],"cuda":["nvcc","--version"],
                                       "cxx":[*os.environ.get("CXX","c++").split(),"--version"]}.items()},
         "binary_sha256":h(dist/"bin/ollama"),
         "libraries":{p.relative_to(dist).as_posix():h(p) for p in sorted((dist/"lib/ollama").rglob("*")) if p.is_file()}}
assert any("cuda" in p for p in receipt["libraries"])
(dist/"graft-build.json").write_text(json.dumps(receipt,indent=2)+"\n")
PY
