#!/usr/bin/env bash
# Lints the sources the current build actually configured, so a file excluded
# for want of cuTENSOR is not linted without a compile command. Headers are
# reached through the sources that include them; -header-filter is passed here
# because .clang-tidy's HeaderFilterRegex is read but not applied.
set -euo pipefail

readonly build="${1:-build}"
readonly db="${build}/compile_commands.json"

if [[ ! -f ${db} ]]; then
  echo "configure first: cmake -S . -B ${build} -GNinja" >&2
  exit 1
fi

python3 - "${db}" <<'PY' | xargs -r clang-tidy -p "${build}" --quiet \
  --header-filter='.*/(cuein|test)/.*'
import json, sys
entries = json.load(open(sys.argv[1]))
for path in sorted({e["file"] for e in entries}):
    if "/test/" in path or "/src/" in path:
        print(path)
PY
