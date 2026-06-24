# Appose: multi-language interprocess cooperation with shared memory.
# Copyright (C) 2023 - 2026 Appose developers.
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for PixiInstallMonitor stderr parsing and lock file counting."""

from appose.builder.pixi_install_monitor import (
    PixiInstallMonitor,
    _pixi_platform_string,
)


def _make_monitor(env_dir, env_name="default"):
    events = []
    errors = []
    monitor = PixiInstallMonitor(
        env_dir,
        env_name,
        [lambda title, cur, mx: events.append((title, cur, mx))],
        lambda line: errors.append(line),
    )
    return monitor, events, errors


def test_parse_lock_file_conda_count(tmp_path):
    """Counts conda packages for the target platform, ignoring other envs/platforms."""
    plat = _pixi_platform_string()
    lock = f"""# generated
version: 6
environments:
  default:
    channels:
    - url: https://conda.anaconda.org/conda-forge/
    packages:
      {plat}:
      - conda: https://conda.anaconda.org/conda-forge/python-3.12.conda
      - conda: https://conda.anaconda.org/conda-forge/numpy-1.0.conda
      - pypi: https://files.pythonhosted.org/foo.whl
      some-other-platform:
      - conda: https://example/other.conda
  other:
    packages:
      {plat}:
      - conda: https://example/should-not-count.conda
packages:
- conda: top-level-ignored
"""
    (tmp_path / "pixi.lock").write_text(lock, encoding="utf-8")
    monitor, _, _ = _make_monitor(tmp_path)
    # Two conda packages for our platform in the "default" env; the pypi entry,
    # the other platform, and the "other" env must not be counted.
    assert monitor._parse_lock_file_conda_count() == 2


def test_parse_lock_file_missing(tmp_path):
    """Returns 0 when no lock file is present."""
    monitor, _, _ = _make_monitor(tmp_path)
    assert monitor._parse_lock_file_conda_count() == 0


def test_solve_progress_events(tmp_path):
    """Solve lines accumulate the denominator and fire numbered progress."""
    monitor, events, errors = _make_monitor(tmp_path)
    monitor.intercept("DEBUG solve{x}: solve_pixi: fetched 100 records")
    monitor.intercept("DEBUG solve{x}: solve_pixi: fetched 200 records")
    monitor.intercept(
        "INFO update: resolved conda environment for solve group 'default' 'osx-arm64'"
    )
    monitor.intercept(
        "INFO update: resolved conda environment for solve group 'default' 'linux-64'"
    )
    assert ("Solving", 1, 2) in events
    assert ("Solving", 2, 2) in events
    # Every line is forwarded to the original error consumer.
    assert len(errors) == 4


def test_pypi_progress_events(tmp_path):
    """PyPI prepare/install lines fire progress relative to packages to install."""
    monitor, events, _ = _make_monitor(tmp_path)
    monitor.intercept(
        "DEBUG pixi_install_pypi: 1 of 5 required packages are considered installed"
    )
    monitor.intercept("INFO pixi_install_pypi: Prepared 4 packages")
    monitor.intercept("INFO pixi_install_pypi: Installed 4 packages")
    monitor.intercept("INFO update: Installed environment 'default'")
    # 5 required - 1 already installed = 4 to install.
    assert ("Downloading PyPI packages", 4, 4) in events
    assert ("Installing PyPI packages", 4, 4) in events
    assert ("Done", 1, 1) in events


def test_intercept_forwards_none_safely(tmp_path):
    """A None/empty line is forwarded without attempting to parse it."""
    monitor, events, errors = _make_monitor(tmp_path)
    monitor.intercept("")
    assert errors == [""]
    assert events == []
