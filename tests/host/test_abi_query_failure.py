# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

from build_stub import run_build


@pytest.mark.parametrize("abi,expected", [("True", "ON"), ("False", "OFF")])
def test_successful_abi_query_keeps_value(tmp_path, abi, expected):
    result, arguments, marker, _, _ = run_build(tmp_path, abi=abi)
    assert result.returncode == 0
    assert marker.exists()
    assert "-DUSE_CXX11_ABI=" + expected in json.loads(arguments.read_text())


def test_failed_abi_query_warns_and_falls_back(tmp_path):
    """A probe that fails must be loud but must not abort the build (issue #65)."""
    result, arguments, marker, _, _ = run_build(tmp_path, abi="", python_exit=1)
    assert result.returncode == 0
    assert marker.exists()
    assert "WARNING: failed to query the PyTorch C++ ABI" in result.stderr
    assert "-DUSE_CXX11_ABI=OFF" in json.loads(arguments.read_text())


def test_unexpected_abi_output_warns_and_falls_back(tmp_path):
    result, arguments, marker, _, _ = run_build(tmp_path, abi="maybe")
    assert result.returncode == 0
    assert marker.exists()
    assert "WARNING: unexpected PyTorch C++ ABI query result" in result.stderr
    assert "-DUSE_CXX11_ABI=OFF" in json.loads(arguments.read_text())


def test_explicit_override_wins_over_failed_probe(tmp_path):
    result, arguments, marker, _, _ = run_build(tmp_path, abi="", python_exit=1,
                                                extra_args=("--cxx11-abi", "on"))
    assert result.returncode == 0
    assert marker.exists()
    assert "set by --cxx11-abi" in result.stdout
    assert "-DUSE_CXX11_ABI=ON" in json.loads(arguments.read_text())


def test_invalid_override_is_rejected(tmp_path):
    result, _, marker, _, _ = run_build(tmp_path, extra_args=("--cxx11-abi", "maybe"))
    assert result.returncode != 0
    assert not marker.exists()
    assert "Invalid --cxx11-abi value" in result.stderr


def test_auto_after_override_uses_probe(tmp_path):
    result, arguments, marker, _, _ = run_build(
        tmp_path, abi="False",
        extra_args=("--cxx11-abi", "on", "--cxx11-abi", "auto"),
    )
    assert result.returncode == 0
    assert marker.exists()
    assert "-DUSE_CXX11_ABI=OFF" in json.loads(arguments.read_text())
