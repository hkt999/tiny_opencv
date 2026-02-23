#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_coverage}"
MIN_LINE="${MIN_LINE_COVERAGE:-70}"
MIN_BRANCH="${MIN_BRANCH_COVERAGE:-50}"

echo "[coverage] configure: $BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON >/dev/null
echo "[coverage] build"
cmake --build "$BUILD_DIR" -j >/dev/null
echo "[coverage] run tests"
"$BUILD_DIR/tiny_opencv_test" >/dev/null

if ! command -v gcov >/dev/null 2>&1; then
    echo "[coverage] error: gcov not found"
    exit 2
fi

TMP_OUTPUT="$(mktemp)"
trap 'rm -f "$TMP_OUTPUT"' EXIT

while IFS= read -r src; do
    obj="$BUILD_DIR/CMakeFiles/tiny_opencv.dir/${src}.o"
    if [[ -f "$obj" ]]; then
        gcov -n -b -o "$obj" "$ROOT_DIR/$src" >>"$TMP_OUTPUT" 2>/dev/null || true
    fi
done < <(cd "$ROOT_DIR" && find src -name '*.cpp' -not -path 'src/3rd_party/*' | sort)

read -r LINE_PCT BRANCH_PCT LINE_TOTAL BRANCH_TOTAL < <(
awk '
BEGIN { line_exec=0; line_total=0; branch_exec=0; branch_total=0; }
/^File / {
  in_scope = 0;
  if ($0 ~ /\/src\// && $0 !~ /\/src\/3rd_party\//) {
    in_scope = 1;
  }
}
/^Lines executed:/ && in_scope {
  pct = $2;
  sub("executed:", "", pct);
  sub("%", "", pct);
  total = $4 + 0;
  line_exec += (pct + 0) * total / 100.0;
  line_total += total;
}
/^Branches executed:/ && in_scope {
  pct = $2;
  sub("executed:", "", pct);
  sub("%", "", pct);
  total = $4 + 0;
  branch_exec += (pct + 0) * total / 100.0;
  branch_total += total;
}
END {
  line_pct = (line_total > 0) ? (line_exec * 100.0 / line_total) : 0;
  branch_pct = (branch_total > 0) ? (branch_exec * 100.0 / branch_total) : 0;
  printf("%.2f %.2f %d %d\n", line_pct, branch_pct, line_total, branch_total);
}
' "$TMP_OUTPUT"
)

echo "[coverage] lines: ${LINE_PCT}% (${LINE_TOTAL})"
echo "[coverage] branches: ${BRANCH_PCT}% (${BRANCH_TOTAL})"
echo "[coverage] gate: lines>=${MIN_LINE}% branches>=${MIN_BRANCH}%"

pass=1
awk "BEGIN {exit !($LINE_PCT >= $MIN_LINE)}" || pass=0
awk "BEGIN {exit !($BRANCH_PCT >= $MIN_BRANCH)}" || pass=0

if [[ "$pass" -ne 1 ]]; then
    echo "[coverage] FAIL"
    exit 1
fi

echo "[coverage] PASS"
