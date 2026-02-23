#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_coverage}"
MIN_LINE="${MIN_LINE_COVERAGE:-70}"
MIN_BRANCH="${MIN_BRANCH_COVERAGE:-50}"
KEY_FILE_GATES="${KEY_FILE_GATES:-src/mat.cpp:82:88;src/filter/filter_2d.cpp:92:90;src/algorithm/kalman_filter.cpp:95:95;src/randn.cpp:95:95;src/utils/split.cpp:88:95;src/utils/merge.cpp:90:94;src/cvtcolor/rgb2hsv.cpp:68:65}"

echo "[coverage] configure: $BUILD_DIR"
rm -rf "$BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON >/dev/null
echo "[coverage] build"
cmake --build "$BUILD_DIR" -j >/dev/null
find "$BUILD_DIR" -name '*.gcda' -delete
echo "[coverage] run tests"
if ! "$BUILD_DIR/tiny_opencv_test" >/dev/null; then
    echo "[coverage] error: tiny_opencv_test failed"
    exit 1
fi

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

get_file_cov() {
    local rel="$1"
    local obj="$BUILD_DIR/CMakeFiles/tiny_opencv.dir/${rel}.o"
    if [[ ! -f "$obj" ]]; then
        echo "0 0"
        return
    fi

    local out line_cov branch_cov
    out="$(gcov -n -b -o "$obj" "$ROOT_DIR/$rel" 2>/dev/null || true)"
    line_cov="$(awk -v f="$ROOT_DIR/$rel" '
    $0=="File '\''"f"'\''" {hit=1; next}
    hit && /^File / {hit=0}
    hit && /^Lines executed:/ {
        v=$0; sub("Lines executed:","",v); split(v,a,"%"); print a[1]; exit
    }
    ' <<< "$out")"
    branch_cov="$(awk -v f="$ROOT_DIR/$rel" '
    $0=="File '\''"f"'\''" {hit=1; next}
    hit && /^File / {hit=0}
    hit && /^Branches executed:/ {
        v=$0; sub("Branches executed:","",v); split(v,a,"%"); print a[1]; exit
    }
    ' <<< "$out")"

    if [[ -z "${line_cov}" ]]; then line_cov="0"; fi
    if [[ -z "${branch_cov}" ]]; then branch_cov="0"; fi
    echo "$line_cov $branch_cov"
}

IFS=';' read -r -a gate_specs <<< "$KEY_FILE_GATES"
for spec in "${gate_specs[@]}"; do
    [[ -z "$spec" ]] && continue
    IFS=':' read -r rel_file min_l min_b <<< "$spec"
    if [[ -z "${rel_file:-}" || -z "${min_l:-}" || -z "${min_b:-}" ]]; then
        echo "[coverage] skip malformed file gate: $spec"
        pass=0
        continue
    fi

    read -r file_line file_branch < <(get_file_cov "$rel_file")
    echo "[coverage] file: ${rel_file} lines=${file_line}% branches=${file_branch}% gate>=${min_l}%/${min_b}%"
    awk "BEGIN {exit !($file_line >= $min_l)}" || pass=0
    awk "BEGIN {exit !($file_branch >= $min_b)}" || pass=0
done

if [[ "$pass" -ne 1 ]]; then
    echo "[coverage] FAIL"
    exit 1
fi

echo "[coverage] PASS"
