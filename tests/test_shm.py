# Appose: multi-language interprocess cooperation with shared memory.
# Copyright (C) 2023 - 2026 Appose developers.
# SPDX-License-Identifier: BSD-2-Clause

import pytest

import appose
from appose.service import TaskStatus
from appose.shm import _bytes_per_element, _normalize_dtype

ndarray_inspect = """
task.outputs["rsize"] = data.shm.rsize
task.outputs["size"] = data.shm.size
task.outputs["dtype"] = data.dtype
task.outputs["shape"] = data.shape
task.outputs["sum"] = sum(v for v in data.shm.buf)
"""


def test_ndarray():
    env = appose.system()
    with (
        env.python() as service,
        appose.SharedMemory(create=True, rsize=2 * 2 * 20 * 25) as shm,
    ):
        # Construct the data.
        shm.buf[0] = 123
        shm.buf[456] = 78
        shm.buf[1999] = 210
        data = appose.NDArray("uint16", [2, 20, 25], shm)

        # Run the task.
        task = service.task(ndarray_inspect, {"data": data})
        task.wait_for()

        # Validate the execution result.
        assert TaskStatus.COMPLETE == task.status
        # The requested size is 2*20*25*2=2000, but actual allocated
        # shm size varies by platform; e.g. on macOS it is 16384.
        assert 2 * 20 * 25 * 2 == task.outputs["rsize"]
        assert task.outputs["size"] >= task.outputs["rsize"]
        assert "uint16" == task.outputs["dtype"]
        assert [2, 20, 25] == task.outputs["shape"]
        assert 123 + 78 + 210 == task.outputs["sum"]


def test_dtype_standard_names():
    for dtype, size in [
        ("int8", 1),
        ("int16", 2),
        ("int32", 4),
        ("int64", 8),
        ("uint8", 1),
        ("uint16", 2),
        ("uint32", 4),
        ("uint64", 8),
        ("float16", 2),
        ("float32", 4),
        ("float64", 8),
        ("complex64", 8),
        ("complex128", 16),
        ("bool", 1),
    ]:
        assert dtype == _normalize_dtype(dtype)
        assert size == _bytes_per_element(dtype)


def test_dtype_short_forms():
    for short, name in [
        ("i1", "int8"),
        ("i2", "int16"),
        ("i4", "int32"),
        ("i8", "int64"),
        ("u1", "uint8"),
        ("u2", "uint16"),
        ("u4", "uint32"),
        ("u8", "uint64"),
        ("f2", "float16"),
        ("f4", "float32"),
        ("f8", "float64"),
        ("c8", "complex64"),
        ("c16", "complex128"),
        ("b1", "bool"),
        ("?", "bool"),
    ]:
        assert name == _normalize_dtype(short)
        assert name == _normalize_dtype("=" + short)
        assert name == _normalize_dtype("|" + short)


def test_dtype_explicit_byte_order():
    for dtype in ["<u2", ">u2", "<f4", ">c16", "<uint16", ">float32"]:
        with pytest.raises(ValueError, match="native byte order"):
            _normalize_dtype(dtype)


def test_dtype_unsupported():
    for dtype in [
        "",
        "=",
        "==u2",
        "|=u2",
        "=uint16",
        "|uint8",
        "uint",
        "u3",
        "f16",
        "c32",
        "?1",
        "l",
        "g",
        "longdouble",
        "float128",
        "intp",
        "U10",
        "datetime64[ns]",
        "object",
        "FLOAT32",
    ]:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _normalize_dtype(dtype)


def test_ndarray_normalizes_dtype():
    with appose.NDArray("=u2", [3, 5]) as data:
        assert "uint16" == data.dtype
        assert 3 * 5 * 2 == data.shm.rsize
