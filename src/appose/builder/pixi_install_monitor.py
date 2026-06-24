# Appose: multi-language interprocess cooperation with shared memory.
# Copyright (C) 2023 - 2026 Appose developers.
# SPDX-License-Identifier: BSD-2-Clause

"""
Monitor pixi install progress by parsing stderr output and polling the
conda-meta directory for installed packages.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable

from . import ProgressConsumer
from ..util import platform


# -- Stderr patterns (requires -vv) --

# Matches: DEBUG solve{...}: ...solve_pixi: fetched N records
_FETCHED_RECORDS = re.compile(r"solve\{.*\}.*fetched (\d+) records")

# Matches: INFO ...::update: resolved conda environment for solve group 'X' 'platform'
_SOLVED = re.compile(r"resolved conda environment for solve group")

# Matches: DEBUG pixi_core::environment::conda_prefix: updating prefix for 'X'
_CONDA_PREFIX_UPDATING = re.compile(r"conda_prefix: updating prefix for")

# Matches: DEBUG pixi_core::environment::conda_metadata: Prefix file updated
_CONDA_METADATA_DONE = re.compile(r"conda_metadata: Prefix file updated")

# Matches: DEBUG pixi_install_pypi: N of M required packages are considered installed
_PYPI_REQUIRED = re.compile(
    r"pixi_install_pypi: (\d+) of (\d+) required packages are considered installed"
)

# Matches: INFO pixi_install_pypi: Prepared N packages
_PYPI_PREPARED = re.compile(r"pixi_install_pypi: Prepared (\d+) packages")

# Matches: INFO pixi_install_pypi: Installed N packages
_PYPI_INSTALLED = re.compile(r"pixi_install_pypi: Installed (\d+) packages")

# Matches: INFO ...::update: Installed environment 'X'
_ENV_INSTALLED = re.compile(r"Installed environment '")


def _pixi_platform_string() -> str:
    """
    Map the Appose platform string to pixi's lock file platform convention.
    """
    mapping = {
        "MACOS|ARM64": "osx-arm64",
        "MACOS|X64": "osx-64",
        "LINUX|ARM64": "linux-aarch64",
        "LINUX|X64": "linux-64",
        "WINDOWS|ARM64": "win-arm64",
        "WINDOWS|X64": "win-64",
    }
    return mapping.get(platform.PLATFORM, platform.PLATFORM.lower().replace("|", "-"))


class PixiInstallMonitor:
    """
    Monitors pixi install progress by parsing stderr output (at ``-vv``
    verbosity) and polling the ``conda-meta/`` directory for installed
    packages. Fires progress events through existing progress subscribers.

    This class sits between pixi's stderr stream and the builder's error
    subscribers, intercepting each line to detect phase transitions and
    emit structured progress updates.
    """

    def __init__(
        self,
        env_dir: Path,
        env_name: str,
        progress_subscribers: list[ProgressConsumer],
        original_error_consumer: Callable[[str], None] | None,
    ):
        """
        Create a new monitor for a pixi install operation.

        Args:
            env_dir: The pixi project directory (containing pixi.toml).
            env_name: The pixi environment name (e.g. "default").
            progress_subscribers: The progress subscribers to fire events to.
            original_error_consumer: The original error consumer to pass lines
                through to.
        """
        self._env_dir = Path(env_dir)
        self._env_name = env_name
        self._pixi_platform = _pixi_platform_string()
        self._progress_subscribers = progress_subscribers
        self._original_error_consumer = original_error_consumer

        # Solve tracking
        self._total_platforms = 0
        self._solved = 0

        # Conda install tracking
        self._total_conda = -1
        # Stop signal for the conda polling thread (set = stop polling).
        self._conda_polling_stop = threading.Event()
        self._conda_polling_thread: threading.Thread | None = None

        # PyPI tracking
        self._total_pypi = 0
        self._already_installed_pypi = 0

    def intercept(self, line: str) -> None:
        """
        Intercept a stderr line, parsing it for phase-transition signals
        and firing progress events as appropriate. The line is always
        forwarded to the original error consumer.

        Args:
            line: A line from pixi's stderr.
        """
        if line:
            self._parse_line(line)
        # Always forward to the original consumer.
        if self._original_error_consumer is not None:
            self._original_error_consumer(line)

    def shutdown(self) -> None:
        """
        Stop any background polling threads. Call this after
        ``pixi install`` completes.
        """
        self._conda_polling_stop.set()
        t = self._conda_polling_thread
        if t is not None:
            t.join(2.0)
            self._conda_polling_thread = None

    # -- Helper methods --

    def _parse_line(self, line: str) -> None:
        # Solve phase: platform fetched records (accumulates denominator).
        if _FETCHED_RECORDS.search(line):
            self._total_platforms += 1
            return

        # Solve phase: a platform resolved.
        if _SOLVED.search(line):
            self._solved += 1
            if self._total_platforms > 0:
                self._fire_progress("Solving", self._solved, self._total_platforms)
            return

        # Conda install begins: parse lock file, start polling.
        if _CONDA_PREFIX_UPDATING.search(line):
            self._total_conda = self._parse_lock_file_conda_count()
            if self._total_conda > 0:
                self._start_conda_polling()
            return

        # Conda install finished.
        if _CONDA_METADATA_DONE.search(line):
            self._conda_polling_stop.set()
            if self._total_conda > 0:
                self._fire_progress(
                    "Installing conda packages", self._total_conda, self._total_conda
                )
            return

        # PyPI: required packages count.
        m = _PYPI_REQUIRED.search(line)
        if m:
            self._already_installed_pypi = int(m.group(1))
            self._total_pypi = int(m.group(2))
            return

        # PyPI: packages prepared.
        m = _PYPI_PREPARED.search(line)
        if m:
            prepared = int(m.group(1))
            to_install = self._total_pypi - self._already_installed_pypi
            if to_install > 0:
                self._fire_progress("Downloading PyPI packages", prepared, to_install)
            return

        # PyPI: packages installed.
        m = _PYPI_INSTALLED.search(line)
        if m:
            installed = int(m.group(1))
            to_install = self._total_pypi - self._already_installed_pypi
            if to_install > 0:
                self._fire_progress("Installing PyPI packages", installed, to_install)
            return

        # Environment fully installed.
        if _ENV_INSTALLED.search(line):
            self._fire_progress("Done", 1, 1)

    def _parse_lock_file_conda_count(self) -> int:
        """
        Parse ``pixi.lock`` to count conda packages for the target environment
        and platform. Uses a simple line-by-line state machine.

        Returns:
            The number of conda packages, or 0 if the lock file cannot be parsed.
        """
        lock_file = self._env_dir / "pixi.lock"
        if not lock_file.is_file():
            return 0

        # State machine: track which section we're in.
        # Level 0: top-level
        # Level 1: inside "environments:"
        # Level 2: inside target env (e.g. "  default:")
        # Level 3: inside "    packages:"
        # Level 4: inside target platform (e.g. "      osx-arm64:")
        level = 0
        count = 0

        try:
            with open(lock_file, "r", encoding="utf-8") as reader:
                for ln in reader:
                    # Strip trailing newline only for indent counting purposes.
                    ln = ln.rstrip("\n")
                    # Skip blank lines and comments.
                    trimmed = ln.strip()
                    if not trimmed or trimmed.startswith("#"):
                        continue

                    indent = _leading_spaces(ln)

                    if level == 0:
                        # Look for "environments:" at top level.
                        if indent == 0 and trimmed == "environments:":
                            level = 1
                    elif level == 1:
                        # Inside "environments:" — look for our env name.
                        if indent == 0:
                            # Left the environments section entirely.
                            return count
                        if indent == 2 and trimmed == self._env_name + ":":
                            level = 2
                    elif level == 2:
                        # Inside target environment — look for "packages:".
                        if indent <= 2:
                            # Left the target env section.
                            return count
                        if indent == 4 and trimmed == "packages:":
                            level = 3
                    elif level == 3:
                        # Inside "packages:" — look for target platform.
                        if indent <= 4:
                            # Left the packages section.
                            return count
                        if indent == 6 and trimmed == self._pixi_platform + ":":
                            level = 4
                    elif level == 4:
                        # Inside target platform — count "- conda:" lines.
                        # List items sit at the same indent (6) as the platform
                        # key, so only a shallower indent, or a sibling key at
                        # indent 6 (e.g. another platform), ends the section.
                        if indent < 6 or (indent == 6 and not trimmed.startswith("-")):
                            # Left the platform section.
                            return count
                        if trimmed.startswith("- conda:"):
                            count += 1
        except OSError:
            # Lock file unreadable; return what we have.
            pass
        return count

    def _start_conda_polling(self) -> None:
        """
        Start a background thread that polls the conda-meta directory for
        ``.json`` files, emitting progress events every 500ms.
        """
        self._conda_polling_stop.clear()
        conda_meta_path = (
            self._env_dir / ".pixi" / "envs" / self._env_name / "conda-meta"
        )

        def poll() -> None:
            while not self._conda_polling_stop.is_set():
                installed = _count_json_files(conda_meta_path)
                self._fire_progress(
                    "Installing conda packages", installed, self._total_conda
                )
                # Wait up to 500ms, returning early if polling is stopped.
                self._conda_polling_stop.wait(0.5)

        self._conda_polling_thread = threading.Thread(
            target=poll, name="PixiInstallMonitor-conda-poll", daemon=True
        )
        self._conda_polling_thread.start()

    def _fire_progress(self, title: str, current: int, maximum: int) -> None:
        for subscriber in self._progress_subscribers:
            subscriber(title, current, maximum)


def _leading_spaces(s: str) -> int:
    """Count leading space characters in a string."""
    count = 0
    for ch in s:
        if ch == " ":
            count += 1
        else:
            break
    return count


def _count_json_files(directory: Path) -> int:
    """Count ``*.json`` files in a directory, returning 0 if it doesn't exist."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.name.endswith(".json"))
