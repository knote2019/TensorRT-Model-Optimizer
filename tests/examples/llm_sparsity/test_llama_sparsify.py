# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import json
import pytest
from _test_utils.examples.run_command import (
    run_example_command,
    run_llm_sparsity_command,
    run_llm_sparsity_ft_command,
    run_llm_sparsity_eval_command,
    run_llm_sparsity_export_command,
    run_llm_sparsity_test_command,
)
from _test_utils.torch_misc import minimum_gpu


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp("data")
    run_example_command(["python", "data_prep.py", "--save_path", data_path], "llm_sparsity")

    # Copy eval data to train path for faster test
    run_example_command(
        ["mv", data_path / "cnn_eval.json", data_path / "cnn_eval_default.json"], "llm_sparsity"
    )

    with open(data_path / "cnn_eval_default.json") as f1:
        data = json.load(f1)
        with open(data_path / "cnn_eval.json", 'w') as f2:
            json.dump(data[:50], f2, indent=4, ensure_ascii=False)

    return data_path


@pytest.mark.parametrize(
    ("sparsity_fmt", "dtype"),
    [
        ("sparsegpt", "bf16"),
        ("sparse_magnitude", "bf16"),
    ],
)
def test_llama_sparsity(tiny_llama_path, tmp_path, sparsity_fmt, dtype):
    run_llm_sparsity_command(
        model=tiny_llama_path,
        output_dir=tmp_path,
        sparsity_fmt=sparsity_fmt,
        dtype=dtype,
        calib_size=8,
        model_max_length=128,
    )


def _test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype, num_gpus):
    pts_output = tmp_path / "pts_output"
    finetune_output = tmp_path / "finetune_output"
    finetune_ckpt_output = tmp_path / "finetune_ckpt_output"
    finetune_engine_output = tmp_path / "finetune_engine_output"

    # First do sparsification
    run_llm_sparsity_command(
        model=tiny_llama_path,
        output_dir=pts_output,
        sparsity_fmt=sparsity_fmt,
        dtype=dtype,
        calib_size=8,
        model_max_length=128,
    )

    # Then do finetuning using the sparsified model
    restore_path = pts_output / "pts_modelopt_state.pth"
    run_llm_sparsity_ft_command(
        model=tiny_llama_path,
        restore_path=restore_path,
        output_dir=finetune_output,
        data_path=data_path,
        num_epochs=0.001,
        max_length=128,
    )

    # Then do eval using the sparsified model.
    run_llm_sparsity_eval_command(
        model=tiny_llama_path,
        restore_path=restore_path,
        data_path=f"{data_path}/cnn_eval.json",
        num_gpus=num_gpus,
    )

    # Then do export using the sparsified model.
    run_llm_sparsity_export_command(
        model=tiny_llama_path,
        restore_path=restore_path,
        output_dir=finetune_ckpt_output,
        tensor_parallel=num_gpus,
    )

    # Then do test using the sparsified model.
    run_llm_sparsity_test_command(
        model=tiny_llama_path,
        ckpt_dir=finetune_ckpt_output,
        output_dir=finetune_engine_output,
        tensor_parallel=num_gpus,
    )


@pytest.mark.parametrize(
    ("sparsity_fmt", "dtype"),
    [
        ("sparsegpt", "bf16"),
    ],
)
def test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype, num_gpus):
    _test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype, num_gpus)


@pytest.mark.parametrize(
    ("sparsity_fmt", "dtype"),
    [
        ("sparsegpt", "fp16"),
    ],
)
@minimum_gpu(2)
def test_llama_sparsity_finetune_multi_gpu(
    tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype
):
    _test_llama_sparsity_finetune(tiny_llama_path, tmp_path, data_path, sparsity_fmt, dtype)
