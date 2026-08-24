# Appose: multi-language interprocess cooperation with shared memory.
# Copyright (C) 2023 - 2026 Appose developers.
# SPDX-License-Identifier: BSD-2-Clause

"""
TODO
"""

from __future__ import annotations

import re
from math import ceil, prod
from multiprocessing import resource_tracker, shared_memory
from typing import TYPE_CHECKING

from .util import message

if TYPE_CHECKING:
    from typing import Self


class SharedMemory(shared_memory.SharedMemory):
    """
    An enhanced version of Python's multiprocessing.shared_memory.SharedMemory
    class which can be used with a `with` statement. When the program flow
    exits the `with` block, this class's `dispose()` method will be invoked,
    which might call `close()` or `unlink()` depending on the value of its
    `unlink_on_dispose` flag.
    """

    def __init__(self, name: str | None = None, create: bool = False, rsize: int = 0):
        """
        Create a new shared memory block, or attach to an existing one.

        Args:
            name: The unique name for the requested shared memory, specified as a
                string. If create is True (i.e. a new shared memory block) and
                no name is given, a novel name will be generated.
            create: Whether a new shared memory block is created (True)
                or an existing one is attached to (False).
            rsize: Requested size in bytes. The true allocated size will be at least
                this much, but may be rounded up to the next block size multiple,
                depending on the running platform.
        """
        super().__init__(name=name, create=create, size=rsize)
        self.rsize: int = rsize
        self._unlink_on_dispose: bool = create
        if message._worker_mode:
            # HACK: Remove this shared memory block from the resource_tracker,
            # which would otherwise want to clean up shared memory blocks
            # after all known references are done using them.
            #
            # There is one resource_tracker per Python process, and they will
            # each try to delete shared memory blocks known to them when they
            # are shutting down, even when other processes still need them.
            #
            # As such, the rule Appose follows is: let the service process
            # always handle cleanup of shared memory blocks, regardless of
            # which process initially allocated it.
            try:
                resource_tracker.unregister(self._name, "shared_memory")
            except ModuleNotFoundError:
                # Unfortunately, on (some?) Windows systems, we see the error:
                #
                # Traceback (most recent call last):
                #   File "...\site-packages\appose\types.py", line 97, in decode
                #     return json.loads(the_json, object_hook=_appose_object_hook)
                #   File "...\lib\json\__init__.py", line 359, in loads
                #     return cls(**kw).decode(s)
                #   File "...\lib\json\decoder.py", line 337, in decode
                #     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
                #   File "...\lib\json\decoder.py", line 353, in raw_decode
                #     obj, end = self.scan_once(s, idx)
                #   File "...\site-packages\appose\types.py", line 177, in _appose_object_hook
                #     return SharedMemory(name=(obj["name"]), size=(obj["size"]))
                #   File "...\site-packages\appose\types.py", line 63, in __init__
                #     resource_tracker.unregister(self._name, "shared_memory")
                #   File "...\lib\multiprocessing\resource_tracker.py", line 159, in unregister
                #     self._send('UNREGISTER', name, rtype)
                #   File "...\lib\multiprocessing\resource_tracker.py", line 162, in _send
                #     self.ensure_running()
                #   File "...\lib\multiprocessing\resource_tracker.py", line 129, in ensure_running
                #     pid = util.spawnv_passfds(exe, args, fds_to_pass)
                #   File "...\lib\multiprocessing\util.py", line 448, in spawnv_passfds
                #     import _posixsubprocess
                # ModuleNotFoundError: No module named '_posixsubprocess'
                #
                # A bug in Python? Regardless: we guard against it here.
                # See also: https://github.com/imglib/imglib2-appose/issues/1
                pass

    def unlink_on_dispose(self, value: bool) -> None:
        """
        Set whether the `unlink()` method should be invoked to destroy
        the shared memory block when the `dispose()` method is called.

        Note: dispose() is the method called when exiting a `with` block.

        By default, shared memory objects constructed with `create=True`
        will behave this way, whereas shared memory objects constructed
        with `create=False` will not. But this method allows to override
        the behavior.
        """
        self._unlink_on_dispose = value

    def dispose(self) -> None:
        if self._unlink_on_dispose:
            self.unlink()
        else:
            self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.dispose()


class NDArray:
    """
    Data structure for a multi-dimensional array.
    The array contains elements of a data type, arranged in
    a particular shape, and flattened into SharedMemory.
    """

    def __init__(self, dtype: str, shape: list[int], shm: SharedMemory | None = None):
        """
        Create an NDArray.

        Args:
            dtype: The type of the data elements; e.g. int8, uint8, float32, float64.
            shape: The dimensional extents; e.g. a stack of 7 image planes
                with resolution 512x512 would have shape [7, 512, 512].
            shm: The SharedMemory containing the array data, or None to create it.
        """
        self.dtype: str = dtype
        self.shape: list[int] = shape
        self.shm: SharedMemory = (
            SharedMemory(
                create=True, rsize=ceil(prod(shape) * _bytes_per_element(dtype))
            )
            if shm is None
            else shm
        )

    def __str__(self):
        return (
            f"NDArray("
            f"dtype='{self.dtype}', "
            f"shape={self.shape}, "
            f"shm='{self.shm.name}' ({self.shm.rsize}))"
        )

    def ndarray(self):
        """
        Create a NumPy ndarray object for working with the array data.
        No array data is copied; the NumPy array wraps the same SharedMemory.
        Requires the numpy package to be installed.
        """
        try:
            import numpy

            return numpy.ndarray(
                prod(self.shape), dtype=self.dtype, buffer=self.shm.buf
            ).reshape(self.shape)
        except ModuleNotFoundError:
            raise ImportError("NumPy is not available.")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.shm.dispose()


message.register(
    SharedMemory,
    "shm",
    lambda shm: {"name": shm.name, "rsize": shm.rsize},
    lambda m: SharedMemory(name=m["name"], rsize=m["rsize"]),
)
message.register(
    NDArray,
    "ndarray",
    lambda nda: {"dtype": nda.dtype, "shape": nda.shape, "shm": nda.shm},
    lambda m: NDArray(m["dtype"], m["shape"], m["shm"]),
)


def _bytes_per_element(dtype: str) -> int | float:
    """
        Returns the number of bytes for the given type name
        The type name can be standard, e.g. uint16, float32... in that case, parsing the number in the string gives the number of bits, to divide by 8.
        The type name can also be a short version with bytes numbers, e.g. >u2, <f4... Parsing gives directly the nb of bytes
    """
    try:
        # Standard names (e.g., 'uint16', 'float32')
        if dtype.startswith(("uint", "int", "float", "bool", "complex")):
            bits = int(re.sub("[^0-9]", "", dtype))
            bytes_size = bits / 8
        # Short names (e.g., '>u2', '<f4', '|u1')
        else:
            bytes_size = int(re.sub("[^0-9]", "", dtype))
    except ValueError:
        raise ValueError(f"Invalid dtype: {dtype}")
    return bytes_size 
