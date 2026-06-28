#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_DIR/run_all_scenarios_$TIMESTAMP"

mkdir -p "$RUN_LOG_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' tidak ditemukan di PATH."
  echo "Jalankan script ini dari shell yang sudah mengenali conda."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: Tidak bisa menemukan conda.sh dari conda base."
  exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env_ta

export MPLBACKEND=Agg
export DISABLE_MATPLOTLIB_SHOW=1
export PYTHONUNBUFFERED=1

SCENARIOS=(
  "kansas/BiLSTM_Scenario1_Kansas_Baseline.py"
  "kansas/BiLSTM_Scenario2_Correlation.py"
  "kansas/BiLSTM_Scenario2A_Correlation.py"
  "kansas/BiLSTM_Scenario2B_Correlation.py"
  "kansas/BiLSTM_Scenario2C_Correlation.py"
  "kansas/BiLSTM_Scenario3_F1Selection.py"
  "kansas/BiLSTM_Scenario4_WeatherOnly.py"
  "kansas/BiLSTM_Scenario5_WeatherLagOnly.py"
  "kansas/BiLSTM_Scenario6_NoDroughtHistory.py"
  "kansas/BiLSTM_Scenario7_DroughtHistoryOnly.py"
  "nebraska/BiLSTM_Scenario1_Nebraska_Baseline.py"
  "nebraska/BiLSTM_Scenario2_Correlation.py"
  "nebraska/BiLSTM_Scenario2A_Correlation.py"
  "nebraska/BiLSTM_Scenario2B_Correlation.py"
  "nebraska/BiLSTM_Scenario2C_Correlation.py"
  "nebraska/BiLSTM_Scenario3_F1Selection.py"
  "nebraska/BiLSTM_Scenario4_WeatherOnly.py"
  "nebraska/BiLSTM_Scenario5_WeatherLagOnly.py"
  "nebraska/BiLSTM_Scenario6_NoDroughtHistory.py"
  "nebraska/BiLSTM_Scenario7_DroughtHistoryOnly.py"
)

echo "Conda env aktif: $(conda info --envs | awk '/\*/ {print $1}')"
echo "Root dir: $ROOT_DIR"
echo "Log dir: $RUN_LOG_DIR"
echo "Total scenario: ${#SCENARIOS[@]}"
echo "MPLBACKEND=$MPLBACKEND"
echo "DISABLE_MATPLOTLIB_SHOW=$DISABLE_MATPLOTLIB_SHOW"
echo

run_scenario() {
  local script_path="$1"
  local script_name
  local log_path

  script_name="$(basename "$script_path" .py)"
  log_path="$RUN_LOG_DIR/${script_name}.log"

  echo "============================================================"
  echo "Running: $script_path"
  echo "Log: $log_path"
  echo "============================================================"

  (
    cd "$ROOT_DIR"
    python "$script_path"
  ) 2>&1 | tee "$log_path"
}

for script_path in "${SCENARIOS[@]}"; do
  run_scenario "$script_path"
done

echo
echo "Semua scenario selesai dijalankan."
echo "Log tersimpan di: $RUN_LOG_DIR"
