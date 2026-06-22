#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${1:-bilstm-gpu}}"
PYTHON_BIN="${PYTHON_BIN:-}"
MPLBACKEND_VALUE="${MPLBACKEND_VALUE:-Agg}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="$SCRIPT_DIR/logs_train_all/$TIMESTAMP"

SCENARIOS=(
  "BiLSTM_Scenario1_Nebraska_Baseline.py"
  "BiLSTM_Scenario2_Correlation.py"
  "BiLSTM_Scenario2A_Correlation.py"
  "BiLSTM_Scenario2B_Correlation.py"
  "BiLSTM_Scenario2C_Correlation.py"
  "BiLSTM_Scenario3_F1Selection.py"
  "BiLSTM_Scenario4_WeatherOnly.py"
  "BiLSTM_Scenario5_WeatherLagOnly.py"
  "BiLSTM_Scenario6_NoDroughtHistory.py"
  "BiLSTM_Scenario7_DroughtHistoryOnly.py"
)

mkdir -p "$LOG_DIR"

if [[ -n "$PYTHON_BIN" ]]; then
  RUNNER_LABEL="$PYTHON_BIN"
else
  RUNNER_LABEL="conda env: $CONDA_ENV_NAME"
fi

echo "Nebraska batch training"
echo "Folder kerja : $SCRIPT_DIR"
echo "Runner       : $RUNNER_LABEL"
echo "MPLBACKEND   : $MPLBACKEND_VALUE"
echo "Log folder   : $LOG_DIR"
echo

if [[ -n "$PYTHON_BIN" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter tidak ditemukan: $PYTHON_BIN" >&2
    echo "Contoh override: PYTHON_BIN=python ./train_all_scenarios.sh"
    exit 1
  fi
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "Perintah 'conda' tidak ditemukan di PATH." >&2
    echo "Alternatif: PYTHON_BIN=/path/to/python ./train_all_scenarios.sh"
    exit 1
  fi

  if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    echo "Conda env tidak ditemukan: $CONDA_ENV_NAME" >&2
    echo "Alternatif: CONDA_ENV_NAME=nama_env ./train_all_scenarios.sh"
    echo "Atau bypass conda: PYTHON_BIN=/path/to/python ./train_all_scenarios.sh"
    exit 1
  fi
fi

success_count=0
failure_count=0
failed_scenarios=()

for scenario in "${SCENARIOS[@]}"; do
  scenario_path="$SCRIPT_DIR/$scenario"
  scenario_name="${scenario%.py}"
  log_path="$LOG_DIR/${scenario_name}.log"

  echo "============================================================"
  echo "Menjalankan $scenario"
  echo "Log: $log_path"
  echo "============================================================"

  if [[ ! -f "$scenario_path" ]]; then
    echo "File tidak ditemukan: $scenario_path" | tee "$log_path"
    failure_count=$((failure_count + 1))
    failed_scenarios+=("$scenario")
    echo
    continue
  fi

  if [[ -n "$PYTHON_BIN" ]]; then
    (
      cd "$SCRIPT_DIR" &&
      MPLBACKEND="$MPLBACKEND_VALUE" "$PYTHON_BIN" "$scenario_path"
    ) 2>&1 | tee "$log_path"
  else
    (
      cd "$SCRIPT_DIR" &&
      MPLBACKEND="$MPLBACKEND_VALUE" conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$scenario_path"
    ) 2>&1 | tee "$log_path"
  fi

  exit_code=${PIPESTATUS[0]}
  if [[ $exit_code -eq 0 ]]; then
    success_count=$((success_count + 1))
    echo "Status: SUKSES ($scenario)"
  else
    failure_count=$((failure_count + 1))
    failed_scenarios+=("$scenario")
    echo "Status: GAGAL ($scenario), exit code $exit_code"
  fi
  echo
done

echo "==================== RINGKASAN ===================="
echo "Total scenario : ${#SCENARIOS[@]}"
echo "Berhasil       : $success_count"
echo "Gagal          : $failure_count"
echo "Log folder     : $LOG_DIR"

if [[ $failure_count -gt 0 ]]; then
  echo "Scenario gagal :"
  for scenario in "${failed_scenarios[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi

echo "Semua scenario Nebraska selesai."
