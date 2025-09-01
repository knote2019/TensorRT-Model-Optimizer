#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

while [ $# -gt 0 ]; do
  case "$1" in
    --num_proc*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_PROC="${1#*=}"
      ;;
    --model*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL="${1#*=}"
      ;;
    --restore_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODELOPT_RESTORE_PATH="${1#*=}"
      ;;
    --data_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATA_PATH="${1#*=}"
      ;;
    --max_length*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL_MAX_LENGTH="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument ${1#*=}\n"
      exit 1
      ;;
  esac
  shift
done

set -x

NUM_PROC=${NUM_PROC:-8}
MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
MODELOPT_RESTORE_PATH=${MODELOPT_RESTORE_PATH:-"saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1/pts_modelopt_state.pth"}
DATA_PATH=${DATA_PATH:-"data/cnn_eval.json"}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}

CMD="accelerate launch --multi_gpu --num_processes $NUM_PROC eval.py \
    --model_dir $MODEL \
    --modelopt_restore_path $MODELOPT_RESTORE_PATH \
    --data_path $DATA_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --batch_size 1 \
    --beam_size 4 \
"

sh -c "$CMD"
