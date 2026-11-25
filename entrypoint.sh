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
# What is decided at launch is decided here, in bash, where it can be read in one place: that
# the task has the variables it needs, that each value can be used, that nothing is written
# beside the code, and which commit of the code this is. A variable that is unset, empty or the
# word None is not set: a pipeline manifest passes a blank parameter as the text "None".
#
# The variables a job may set. They are the whole of a run's settings:
#   KFP_TASK                 split, classify, extract, clean or evaluate
#   model                    a registered model. Needed by split, classify and extract
#   image_dir                the folder of pages to read. Needed by split, classify and extract
#   output                   the folder the stages write to. Needed by every task
#   ground_truth             the answer key, a .jsonl file. Needed by evaluate
#   document_types           read only the pages whose file name holds one of these, written with
#                            commas, e.g. receipt,invoice. Unset: every page
#   max_images               read at most this many pages. Unset: every page
#   photo_split              true: the split task asks the model where each receipt is, and cuts a
#                            photograph of several. Unset: false, every photograph is passed through
#   document_type            every page is of this document type, and classify asks no model.
#                            Unset: the model is asked
#   CLEAR_PREV_OUTPUT        true: delete the stage's earlier output first. Unset: false
#   DOC_EXTRACT_LOG_DIR      the folder for the log. Needed by every task
#   DOC_EXTRACT_TRACE_FILE   keep what is sent to the model, in this file. Unset: not kept
#   DOC_EXTRACT_CONDA_ENV    the conda environment's folder
#   DOC_EXTRACT_GPU_MEMORY_UTILIZATION   the share of each card's memory the engine may take.
#                            Unset: the model's own, in its registration
#
# By hand:
#   KFP_TASK=classify model=gemma4-31b-w4a16-vllm image_dir=<folder> output=<folder> \
#     DOC_EXTRACT_LOG_DIR=<folder> bash entrypoint.sh
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

# ---- How a job that cannot be started is stopped ---- #
# stop <what is wrong, in a line> <what> <where> <expected> <fix>
stop() {
  echo "FATAL: $1"
  echo "  What:     $2"
  echo "  Where:    $3"
  echo "  Expected: $4"
  echo "  Fix:      $5"
  exit 1
}
IN_THE_JOB="in the environment of this job (the pipeline manifest, or the command that started it)."

# ---- Variables under their old names are refused, not ignored ---- #
# Until 21 November 2026 this script's own variables began with another prefix. A job that still sets one
# would otherwise run with the setting silently dropped: its log, for one, written where nobody asked.
OLD_PREFIX="L""MM_"
OLD_NAMES="$(env | grep -oE "^${OLD_PREFIX}[A-Z_]+" | tr '\n' ' ' || true)"
if [[ -n "$OLD_NAMES" ]]; then
  stop "variable(s) set under a prefix this pipeline no longer reads: $OLD_NAMES" \
    "the prefix ${OLD_PREFIX} was replaced by DOC_EXTRACT_; the old names are not read." \
    "the variables $IN_THE_JOB" \
    "DOC_EXTRACT_LOG_DIR, DOC_EXTRACT_CONDA_ENV, DOC_EXTRACT_GPU_MEMORY_UTILIZATION." \
    "rename each variable listed above, keeping its value, and start the job again."
fi

# ---- A variable that is blank is not set ---- #
JOB_VARIABLES="KFP_TASK model image_dir output ground_truth document_types max_images photo_split
  document_type CLEAR_PREV_OUTPUT DOC_EXTRACT_LOG_DIR DOC_EXTRACT_TRACE_FILE
  DOC_EXTRACT_GPU_MEMORY_UTILIZATION"
for NAME in $JOB_VARIABLES dataset run_config; do
  if [[ -z "${!NAME:-}" || "${!NAME}" == "None" ]]; then unset "$NAME"; fi
done

# ---- Two variables that are no longer read are refused, not ignored ---- #
# Until 26 November 2026 a run's choices were keys of a settings file, which run_config named, and
# dataset named a block of it. A job that still sets one would otherwise run with it dropped.
if [[ -n "${dataset:-}" ]]; then
  stop "dataset is no longer read: $dataset" \
    "a set of pages that are all of one type is now two variables, and no block of a settings file." \
    "the variable dataset $IN_THE_JOB" \
    "document_type=<the type of every page> image_dir=<the folder of pages>, e.g. document_type=SROIE_RECEIPT" \
    "set document_type and image_dir in place of dataset."
fi
if [[ -n "${run_config:-}" ]]; then
  stop "run_config is no longer read: $run_config" \
    "a run has no settings file: its settings are the variables listed at the top of entrypoint.sh." \
    "the variable run_config $IN_THE_JOB" \
    "each choice of the run as its own variable, e.g. max_images=12 document_types=receipt photo_split=true" \
    "set the variables the run needs in place of the file, and unset run_config."
fi

# ---- The task, and what it needs ---- #
case "${KFP_TASK:-}" in
  split | classify | extract) NEEDED="model image_dir output" ;;
  clean) NEEDED="output" ;;
  evaluate) NEEDED="output ground_truth" ;;
  *)
    stop "KFP_TASK is not a task of this pipeline: ${KFP_TASK:-<not set>}" \
      "KFP_TASK names the task this job runs." \
      "the variable KFP_TASK $IN_THE_JOB" \
      "split, classify, extract, clean or evaluate." \
      "set it to one of those, e.g. KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh"
    ;;
esac
for NAME in $NEEDED DOC_EXTRACT_LOG_DIR; do
  if [[ -z "${!NAME:-}" ]]; then
    stop "the task $KFP_TASK needs $NAME, and it is not set" \
      "$KFP_TASK cannot run without $NAME. Unset, empty and the word None all count as not set." \
      "the variable $NAME $IN_THE_JOB" \
      "what the top of entrypoint.sh says of $NAME, e.g. output=/data/run_20261126" \
      "set $NAME and start the job again."
  fi
done
if [[ "$NEEDED" == *image_dir* && ! -d "$image_dir" ]]; then
  stop "the folder of pages does not exist: $image_dir" \
    "$KFP_TASK reads its pages from the folder image_dir names, and there is no such folder." \
    "the variable image_dir $IN_THE_JOB" \
    "a folder that holds the pages to read, e.g. image_dir=/data/pages" \
    "correct the folder, or create it and put the pages in it."
fi
if [[ "$KFP_TASK" == "evaluate" && "$ground_truth" != *.jsonl ]]; then
  stop "the answer key is not a JSON lines file: $ground_truth" \
    "JSON lines is the one format read: an answer a line, with the page's file name and its type's fields." \
    "the variable ground_truth $IN_THE_JOB" \
    "a file ending .jsonl, e.g. ground_truth=/data/pages/ground_truth.jsonl" \
    "name the set's ground_truth.jsonl, or write the key out as JSON lines."
fi

# ---- A run's choices: each value is one that can be used ---- #
if [[ -n "${max_images:-}" ]] && ! [[ "$max_images" =~ ^[1-9][0-9]*$ ]]; then
  stop "max_images is not a number of pages: $max_images" \
    "at most this many pages are read, and it is not a whole number of 1 or more." \
    "the variable max_images $IN_THE_JOB" \
    "a whole number such as max_images=12. Unset, every page is read." \
    "set it to such a number, or leave it unset."
fi
if [[ -n "${document_types:-}" ]] && ! [[ "$document_types" =~ [^,[:space:]] ]]; then
  stop "document_types names no type: '$document_types'" \
    "only pages whose file name holds one of these are read, and none is named." \
    "the variable document_types $IN_THE_JOB" \
    "names written with commas, in either case, e.g. document_types=receipt,invoice. Unset, every page is read." \
    "name at least one, or leave it unset."
fi
# The two that are true or false. A manifest may pass a YAML true as the text "True".
for NAME in photo_split CLEAR_PREV_OUTPUT; do
  VALUE="$(echo "${!NAME:-false}" | tr 'A-Z' 'a-z')"
  if [[ "$VALUE" != "true" && "$VALUE" != "false" ]]; then
    stop "$NAME is neither true nor false: ${!NAME}" \
      "photo_split=true has the split task ask the model and cut; CLEAR_PREV_OUTPUT=true deletes the stage's earlier output first." \
      "the variable $NAME $IN_THE_JOB" \
      "true or false, e.g. $NAME=true. Unset means false." \
      "set it to true or false, or leave it unset."
  fi
  export "$NAME=$VALUE"
done
SHARE="${DOC_EXTRACT_GPU_MEMORY_UTILIZATION:-}"
if [[ -n "$SHARE" ]] && ! [[ "$SHARE" =~ ^(0?\.[0-9]*[1-9][0-9]*|1(\.0*)?)$ ]]; then
  stop "DOC_EXTRACT_GPU_MEMORY_UTILIZATION is not a share of a card's memory: $SHARE" \
    "the engine is given this share of each card's memory, and it is not a number above 0 and at most 1." \
    "the variable DOC_EXTRACT_GPU_MEMORY_UTILIZATION $IN_THE_JOB" \
    "a number such as 0.90. Unset, the share is the model's own, in its registration." \
    "set it to such a number, or leave it unset."
fi

# ---- Nothing is written beside the code ---- #
# Under a pipeline this folder is in the job's pod: its storage is not kept, and a job that writes
# there is stopped. git writes it once, when the job is launched from its commit, and is not used
# after that. So the places the job names to write to are refused if they are inside this folder.
CODE_DIR="$(pwd -P)"
for WRITTEN in "$DOC_EXTRACT_LOG_DIR" "${output:-}" "${DOC_EXTRACT_TRACE_FILE:-}"; do
  if [[ -n "$WRITTEN" && "$(realpath -m -- "$WRITTEN")/" == "$CODE_DIR"/* ]]; then
    stop "the job would write beside the code: $WRITTEN" \
      "a place the job names for its output, its log or its record of what was sent to the model is inside the code's own folder, $CODE_DIR" \
      "the variables output, DOC_EXTRACT_LOG_DIR and DOC_EXTRACT_TRACE_FILE $IN_THE_JOB" \
      "places on the storage mounted for the job's data. The code's folder is in the pod: it is not kept." \
      "name places outside $CODE_DIR"
  fi
done

# ---- The commit this job was launched from ---- #
# Read from the checkout's own files, since git is not run once the job has started. Each task
# that uses the cards notes it in <output>/run_manifest.json. Empty if this folder is no checkout.
HEAD_NOW="$(cat .git/HEAD 2>/dev/null || true)"
if [[ "$HEAD_NOW" == "ref: "* ]]; then
  HEAD_NOW="$(cat ".git/${HEAD_NOW#ref: }" 2>/dev/null || true)"
fi
export DOC_EXTRACT_COMMIT="${HEAD_NOW:0:7}"

# ---- The settings file ---- #
# What is left of it: where each model's weights are. A run's choices are the variables above.
CONFIG_FILE="./config/run_config.yml"

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
# All output (stdout + stderr) is captured to a timestamped log file in DOC_EXTRACT_LOG_DIR,
# while still being printed to the console (so the KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
LOG_DIR="$DOC_EXTRACT_LOG_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"

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
log "What this job set:"
for NAME in $JOB_VARIABLES; do
  log "$(printf '  %-36s %s' "$NAME" "${!NAME:-<not set>}")"
done
log ""

# ---- Conda Activation ---- #
# Assumes EFS is mounted via KFP volume spec at /efs/shared/. If the
# volume mount is missing, conda activate will fail below. Check the
# KFP pipeline YAML volume definitions if this step errors.
#
# KFP containers start with a bare shell. We need to initialise conda
# (the `eval` line) and then activate our environment which has all
# the Python dependencies (torch, transformers, etc.) pre-installed.
# Both models run in vllm_env4 (vLLM 0.29.0). See conda_envs/vllm_env4.yaml.
CONDA_ENV="${DOC_EXTRACT_CONDA_ENV:-/home/jovyan/.conda/envs/vllm_env4}"
if [[ ! -d "$CONDA_ENV" ]]; then
  stop "the conda environment is not there: $CONDA_ENV" \
    "the task runs in a conda environment, and this folder does not exist." \
    "the variable DOC_EXTRACT_CONDA_ENV $IN_THE_JOB Unset, it is the folder written in entrypoint.sh." \
    "the folder of a conda environment, e.g. DOC_EXTRACT_CONDA_ENV=/home/jovyan/.conda/envs/vllm_env4" \
    "point DOC_EXTRACT_CONDA_ENV at the environment's folder."
fi
log "Activating conda environment..."
eval "$(conda shell.bash hook)"
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
# KFP_TASK names it, and it is run with the variables checked above.
log "KFP_TASK: $KFP_TASK"
log "Commit:   ${DOC_EXTRACT_COMMIT:-<not a checkout>}"
log ""
python3 -m stages "$CONFIG_FILE"
