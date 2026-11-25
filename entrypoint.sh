#!/bin/bash
# =============================================================================
# Document extraction pipeline: the starter
# =============================================================================
#
# The first thing that runs in a pipeline job, and the way to run a task by hand.
# It prepares the machine (shell safety, CUDA and vLLM variables, the conda
# environment, a log file, a look at the cards) and then starts the task:
#
#     python3 -m stages
#
# Which task, which model and which folders are worked out there, in Python
# (common/stage_settings.py), from the environment first and the settings file second.
#
# What is decided at launch is decided here, in bash, where it can be read in one place:
# which settings file is read, that nothing is written beside the code, and which commit
# of the code this is.
#
# The variables a job may set:
#   KFP_TASK                 split, classify, extract, clean or evaluate
#   model                    a registered model
#   image_dir                the folder of pages to read
#   output                   the folder the stages write to
#   ground_truth             the answer key, for evaluate
#   dataset                  a dataset of the settings file, whose pages are all of one type
#   run_config               a settings file of the job's own, read whole in place of config/run_config.yml
#   CLEAR_PREV_OUTPUT        true = delete the stage's earlier output first; default false
#   DOC_EXTRACT_LOG_DIR      the folder for the log
#   DOC_EXTRACT_CONDA_ENV    the conda environment's folder
#   DOC_EXTRACT_GPU_MEMORY_UTILIZATION   the share of each card's memory the engine may take
#
# By hand:
#   KFP_TASK=classify model=gemma4-31b-w4a16-vllm image_dir=<folder> output=<folder> bash entrypoint.sh
#
# =============================================================================

# #############################################################################
#  SETUP: Shell Safety, CUDA, Logging (pre-conda)
# #############################################################################

# ---- Shell Safety Settings ---- #
# errexit:  Exit immediately if any command fails (non-zero exit code)
# nounset:  Treat unset variables as an error (catches typos in var names)
# pipefail: A pipeline fails if ANY command in the pipe fails, not just the last
set -o errexit
set -o nounset
set -o pipefail

# ---- Variables under their old names are refused, not ignored ---- #
# Until 21 November 2026 this script's own variables began with another prefix. A job that still sets one
# would otherwise run with the setting silently dropped: its log, for one, written where nobody asked.
OLD_PREFIX="L""MM_"
OLD_NAMES="$(env | grep -oE "^${OLD_PREFIX}[A-Z_]+" | tr '\n' ' ' || true)"
if [[ -n "$OLD_NAMES" ]]; then
  echo "FATAL: variable(s) set under a prefix this pipeline no longer reads: $OLD_NAMES"
  echo "  What:     the prefix ${OLD_PREFIX} was replaced by DOC_EXTRACT_; the old names are not read."
  echo "  Where:    the environment of this job (the pipeline manifest, or the command that started it)."
  echo "  Expected: DOC_EXTRACT_LOG_DIR, DOC_EXTRACT_CONDA_ENV, DOC_EXTRACT_GPU_MEMORY_UTILIZATION."
  echo "  Fix:      rename each variable listed above, keeping its value, and start the job again."
  exit 1
fi

# ---- Nothing is written beside the code ---- #
# Under a pipeline this folder is in the job's pod: its storage is not kept, and a job that writes
# there is stopped. git writes it once, when the job is launched from its commit, and is not used
# after that. So the folders the job names for its output and its log are refused, further down,
# if they are inside this folder.
CODE_DIR="$(pwd -P)"
beside_the_code() {
  case "$(realpath -m -- "$1")/" in "$CODE_DIR"/*) return 0 ;; *) return 1 ;; esac
}

# ---- The commit this job was launched from ---- #
# Read from the checkout's own files, since git is not run once the job has started. Each task
# that uses the cards notes it in <output>/run_manifest.json. Empty if this folder is no checkout.
HEAD_NOW="$(cat .git/HEAD 2>/dev/null || true)"
if [[ "$HEAD_NOW" == "ref: "* ]]; then
  HEAD_NOW="$(cat ".git/${HEAD_NOW#ref: }" 2>/dev/null || true)"
fi
export DOC_EXTRACT_COMMIT="${HEAD_NOW:0:7}"

# ---- The settings file ---- #
# config/run_config.yml beside this script, unless the job names its own in run_config. A job
# changes a setting by naming a file where it can write, never by editing the one in here. The
# file named is the whole of the settings: nothing is merged from the shipped one.
CONFIG_FILE="./config/run_config.yml"
if [[ -n "${run_config:-}" && "${run_config}" != "None" ]]; then
  CONFIG_FILE="$run_config"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "FATAL: the settings file does not exist: $CONFIG_FILE"
  echo "  What:     the job names a settings file in run_config, and there is no such file."
  echo "  Where:    the variable run_config in the environment of this job."
  echo "  Expected: a whole settings file the job can read, e.g. run_config=/data/run_20261126/run_config.yml"
  echo "  Fix:      copy config/run_config.yml to the job's storage, change what the run needs, and name the copy; or unset run_config."
  exit 1
fi

# ---- The share of a card's memory, if the job sets one ---- #
SHARE="${DOC_EXTRACT_GPU_MEMORY_UTILIZATION:-}"
if [[ -n "$SHARE" && "$SHARE" != "None" ]] && ! [[ "$SHARE" =~ ^(0?\.[0-9]*[1-9][0-9]*|1(\.0*)?)$ ]]; then
  echo "FATAL: DOC_EXTRACT_GPU_MEMORY_UTILIZATION is not a share of a card's memory: $SHARE"
  echo "  What:     the engine is given this share of each card's memory, and it is not a number above 0 and at most 1."
  echo "  Where:    the variable DOC_EXTRACT_GPU_MEMORY_UTILIZATION in the environment of this job."
  echo "  Expected: a number such as 0.90. Unset, the share is the model's gpu_memory_utilization in the settings file."
  echo "  Fix:      set it to such a number, or leave it unset."
  exit 1
fi

# ---- CUDA Environment ---- #
# Deterministic GPU indexing: ensures cuda:0 always maps to the same physical
# GPU regardless of driver enumeration order. Critical for multi-GPU so that
# log messages ("GPU 0 failed") match nvidia-smi output.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# vLLM attention backend: left for vLLM to choose. The variable is for an environment with
# pre-compiled FlashInfer kernels.
# export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# Suppress vLLM usage telemetry (avoids TLS cert errors in air-gapped envs).
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

# vLLM's compile cache: both lines are off, and vLLM is left to do as it does. It keeps what
# torch.compile makes under ~/.cache/vllm/torch_compile_cache, which is in the pod (6 files in a run
# of both models on the sandbox, 26 November 2026). If a pod refuses that write, uncomment ONE:
#   the first moves vLLM's cache folder to storage the job can write to (name a real folder);
#   the second keeps no compile cache at all, so every engine start compiles again.
# export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/vllm_cache}"
# export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

# NCCL shared memory: KFP pods may default /dev/shm to 64 MB, which is
# too small for NCCL's SHM transport under tensor parallelism: after ~11
# images the SHM region fills and NCCL silently deadlocks. The fix is to
# increase /dev/shm in the KFP pod spec (emptyDir medium=Memory, 8Gi+).
# Setting NCCL_SHM_DISABLE=1 does NOT work: on cards without
# NVLink, disabling SHM leaves NCCL with no viable intra-node transport
# and it fails immediately with "unhandled system error".

# ---- Log Configuration ---- #
# All output (stdout + stderr) is captured to a timestamped log file on EFS,
# while still being printed to the console (so KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
#
# Priority: DOC_EXTRACT_LOG_DIR env var > bootstrap.logging.log_dir of the settings file > fail
# No silent fallback: in KFP, pod-local writes are ephemeral/forbidden.

# ---- The conda environment's own python ---- #
# The log folder has to be known BEFORE output is redirected into the log, and working it out needs
# PyYAML, which lives only inside the conda environment. So the environment's own interpreter is called
# by path here, and the environment is activated properly further down.
#
# The environment itself cannot be named in run_config.yml, since reading that file needs it.
# Both models run in vllm_env4 (vLLM 0.29.0). See conda_envs/vllm_env4.yaml.
CONDA_ENV="${DOC_EXTRACT_CONDA_ENV:-/home/jovyan/.conda/envs/vllm_env4}"
CONDA_PY="${CONDA_ENV}/bin/python"
if [[ ! -x "$CONDA_PY" ]]; then
  echo "FATAL: bootstrap interpreter not found: $CONDA_PY"
  echo "  What:  the python used to parse $CONFIG_FILE (it needs PyYAML) is missing."
  echo "  Where: CONDA_ENV='$CONDA_ENV': from DOC_EXTRACT_CONDA_ENV env var, or the default in entrypoint.sh."
  echo "  Fix:   point DOC_EXTRACT_CONDA_ENV at a real conda env dir so \$DOC_EXTRACT_CONDA_ENV/bin/python exists, e.g."
  echo "           export DOC_EXTRACT_CONDA_ENV=/home/jovyan/.conda/envs/vllm_env4"
  exit 1
fi
# The log folder: DOC_EXTRACT_LOG_DIR, else bootstrap.logging.log_dir in the settings file, else stop.
# On failure the command prints the reason, which is then what LOG_DIR holds.
LOG_DIR="$("$CONDA_PY" -m common.stage_settings "$CONFIG_FILE")" || { echo "$LOG_DIR"; exit 1; }

# The two folders the job writes to are refused if they are beside the code (see the top).
for WRITTEN in "$LOG_DIR" "${output:-}"; do
  if [[ -n "$WRITTEN" && "$WRITTEN" != "None" ]] && beside_the_code "$WRITTEN"; then
    echo "FATAL: the job would write beside the code: $WRITTEN"
    echo "  What:     a folder the job names for its output or its log is inside the code's own folder, $CODE_DIR"
    echo "  Where:    the variables output and DOC_EXTRACT_LOG_DIR in the environment of this job."
    echo "  Expected: folders on the storage mounted for the job's data. The code's folder is in the pod: it is not kept."
    echo "  Fix:      name folders outside $CODE_DIR"
    exit 1
  fi
done
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"
# The settings this start was run with are kept beside its log, under the same name.
cp "$CONFIG_FILE" "${LOG_FILE%.log}_run_config.yml"

# `exec` redirects ALL subsequent output through `tee`, which writes to
# both the console (for KFP) and the log file (for persistent debugging).
# The `2>&1` merges stderr into stdout so errors are captured too.
# NOTE: Process substitution with tee means the subshell can outlive this
# script: the final log line(s) may flush slightly after exit. This is a
# known bash nuance and is harmless in practice.
exec > >(tee -a "$LOG_FILE") 2>&1

# Timestamped logging function: prefixes every message with a timestamp
# so you can correlate events with KFP logs and identify slow steps.
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# ---- Cleanup Trap ---- #
# `trap ... EXIT` runs this code whenever the script exits, whether it
# succeeds, fails, or gets killed (e.g. OOM). This guarantees you always
# see the exit code and how long the run took, even on crashes.
# $SECONDS is a built-in bash variable that counts elapsed seconds.
SECONDS=0
trap 'rc=$?; echo ""; log "Exited with code $rc after ${SECONDS}s"; log "Log file: $LOG_FILE"' EXIT

# #############################################################################
#  SETUP: Conda Activation & Environment
# #############################################################################

# ---- Banner ---- #
log "================================================================="
log "    Document extraction pipeline"
log "================================================================="
log ""

# ---- Conda Activation ---- #
# Assumes EFS is mounted via KFP volume spec at /efs/shared/. If the
# volume mount is missing, conda activate will fail below. Check the
# KFP pipeline YAML volume definitions if this step errors.
#
# KFP containers start with a bare shell. We need to initialise conda
# (the `eval` line) and then activate our environment which has all
# the Python dependencies (torch, transformers, etc.) pre-installed.
log "Activating conda environment..."
eval "$(conda shell.bash hook)"
# CONDA_ENV was bootstrapped (and its python validated) during the YAML
# resolution above, before the tee redirect.
log "Conda env: $CONDA_ENV"
# Temporarily allow unbound variables: conda activation scripts (e.g. MKL)
# reference variables that may not be set yet.
set +o nounset
conda activate "$CONDA_ENV" || { set -o nounset; log "FATAL: conda activate failed"; exit 1; }
set -o nounset
# Ensure conda's libstdc++ is found before the (older) system copy.
# The vllm_env ships libstdcxx-ng>=12 which provides GLIBCXX_3.4.30
# needed by libzmq; without this, the linker finds /usr/lib64/libstdc++
# first and fails with "version GLIBCXX_3.4.30 not found".
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Use vLLM's PyTorch-native top-k/top-p sampler instead of FlashInfer's.
# REQUIRED on this box, and it costs nothing: every extraction path here decodes
# GREEDILY (VllmBackend builds SamplingParams(temperature=0)), so the top-k/top-p
# kernel is never used for real work: vLLM just warms it at startup regardless.
#
# Without this, engine start FAILS during sampler warmup. flashinfer 0.6.18 has no
# pre-compiled kernels on PyPI (see conda_envs/vllm_env4.yaml), so it JIT-compiles,
# and its JIT shells out to the SYSTEM toolkit at /usr/local/cuda-12.4, which is too
# old for a flag flashinfer emits:
#     nvcc fatal : Unknown option '--compress-mode=size'   (needs nvcc >= 12.8)
# Measured on 21 September 2026: setting this to 0 was the difference between an
# engine that would not start and one that did.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Suppress verbose INFO logging from vLLM engine, transformers, and tokenizers.
# Override by setting these env vars before running entrypoint.sh.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Log environment details for debugging failed runs:
# knowing the Python version, conda env, and GPU type is critical
# when something works locally but fails in the pipeline.
log "---------------------------------------"
log "Python:  $(which python3)"
log "Version: $(python3 --version 2>&1)"
log "Conda:   $(conda info --envs | grep '*' || echo 'unknown')"
log "Log dir: $LOG_DIR"
log "---------------------------------------"
log ""

# ---- GPU Health Check ---- #
# Verify GPUs are accessible and healthy before a model is loaded onto them.
# Catches ECC errors, fallen-off-bus GPUs, and driver mismatches early:
# much cheaper than discovering mid-inference after a 60s model load.
log "GPU environment:"
log "  CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES:-<not set: all GPUs visible>}"
log "  CUDA_DEVICE_ORDER:     ${CUDA_DEVICE_ORDER}"
log "  NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-<not set>}"
log ""

if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
  log "Detected $GPU_COUNT GPU(s):"
  log ""
  # Per-GPU detail: index, name, VRAM, temperature, ECC errors
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name mem_total mem_free temp ecc; do
    log "  GPU $idx: $name |$(echo "$mem_total" | xargs) total |$(echo "$mem_free" | xargs) free | ${temp}C | ECC errors: $(echo "$ecc" | xargs)"
  done
  log ""

  # Check for GPUs in error state: ERR or "Unknown Error" in nvidia-smi
  # means the GPU has fallen off the bus or has a hardware fault.
  if nvidia-smi --query-gpu=index,pstate --format=csv,noheader 2>/dev/null | grep -qi "err"; then
    log "WARNING: One or more GPUs report error state. Run may fail."
    log "$(nvidia-smi --query-gpu=index,pstate,ecc.errors.uncorrected.volatile.total --format=csv 2>/dev/null)"
    log ""
  fi
else
  log "WARNING: nvidia-smi not found: cannot verify GPU health"
  log ""
fi

# #############################################################################
#  THE TASK
# #############################################################################
# KFP_TASK names it. Its model and folders are worked out in Python, where a task that is not known, or a
# setting that is missing, stops the run with a message saying what to set and where.
log "KFP_TASK: ${KFP_TASK:-<not set>}"
log "Settings: $CONFIG_FILE"
log "Commit:   ${DOC_EXTRACT_COMMIT:-<not a checkout>}"
log ""
python3 -m stages "$CONFIG_FILE"
