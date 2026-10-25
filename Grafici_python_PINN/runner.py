"""Starts pinn.exe in the background and reads its output (epochs, losses, current phase).

The C++ program does all the computation; this module only launches it and follows its standard output,
so that the page can show a progress bar and the loss chart while the training runs.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Lines of the pinn.exe output recognised here (see TrainNeuralNetwork.cpp and ExperimentRunner.cpp)
RE_EPOCH = re.compile(r"Epoch \[(\d+)/(\d+)\] -> Loss FNN\^sol: (\S+) \| Loss FNN\^fb: (\S+) \| lr: (\S+)")
RE_SEED = re.compile(r"Seed: (-?\d+)")
RE_TOTAL = re.compile(r"^Addestramento .*: (\d+) epoche|^Training .*: (\d+) epochs")


@dataclass
class Run:
    """One run of pinn.exe: process, results folder and state read from the output."""
    folder: Path
    process: subprocess.Popen
    n_seeds: int
    log_every: int = 10                             # pinn.exe prints the losses every log_every epochs
    start: float = field(default_factory=time.time)
    lines: list = field(default_factory=list)
    history: list = field(default_factory=list)    # (seed, epoch, loss_sol, loss_fb, lr)
    current_seed: int | None = None
    seeds_done: int = 0
    epoch: int = 0
    epochs: int = 0
    phase: str = "starting"
    stopped: bool = False

    @property
    def running(self) -> bool:
        """True while pinn.exe is still working."""
        return self.process.poll() is None

    @property
    def exit_code(self) -> int | None:
        """Exit code of pinn.exe (None while it runs, 0 if it finished correctly)."""
        return self.process.poll()

    def progress(self) -> float:
        """Completed fraction (0..1), counting finished seeds and the epochs of the current one."""
        if self.epochs == 0:
            return 0.0
        return min(1.0, (self.seeds_done + self.epoch / self.epochs) / max(self.n_seeds, 1))

    def stop(self) -> None:
        """Stops the run (button "Stop" of the interface)."""
        if self.running:
            self.stopped = True
            self.process.kill()


def _read_output(run: Run) -> None:
    """Thread: reads the stdout of the process and updates the state (no Streamlit calls here)."""
    log = open(run.folder / "log.txt", "w", encoding="utf-8")
    for line in run.process.stdout:
        line = line.rstrip("\r\n")
        log.write(line + "\n")
        log.flush()
        run.lines.append(line)
        m = RE_EPOCH.search(line)
        if m:
            run.epoch, run.epochs = int(m.group(1)), int(m.group(2))
            run.history.append((run.current_seed, run.epoch, float(m.group(3)), float(m.group(4)), float(m.group(5))))
            run.phase = "training"
            continue
        if line.startswith("Starting PINN training"):
            m = RE_SEED.search(line)
            if run.current_seed is not None:
                run.seeds_done += 1
            run.current_seed = int(m.group(1)) if m else None
            run.epoch, run.phase = 0, "training"
        elif RE_TOTAL.search(line):
            m = RE_TOTAL.search(line)
            run.epochs = int(m.group(1) or m.group(2))
        elif line.startswith("Evaluating test set"):
            run.phase = "test-set evaluation"
        elif line.startswith("FINE") or line.startswith("DONE"):
            run.seeds_done, run.phase = run.n_seeds, "finished"
    log.close()


def new_folder(base: Path, name: str) -> Path:
    """Creates runs/<date>_<time>_<name>, where config.json, log.txt and all the CSV files of the run are saved."""
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    folder = base / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{clean}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def start(exe: Path, config: dict, runs_folder: Path) -> Run:
    """Saves config.json in a new run folder and launches pinn.exe --config ... --out ..."""
    exe = Path(exe).resolve()
    runs_folder = Path(runs_folder).resolve()   # pinn.exe starts from its own folder: absolute paths are needed
    if not exe.is_file():
        raise FileNotFoundError(f"pinn.exe not found in: {exe}\nBuild the C++ project (cmake --build build --config Release).")
    d = config["option"]["numberOfAssets"]
    folder = new_folder(runs_folder, f"d{d}_{config['variants']['name']}")
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW   # no black console window on Windows
    process = subprocess.Popen(
        [str(exe), "--config", str(config_path), "--out", str(folder)],
        cwd=str(exe.parent),            # the LibTorch DLLs are next to pinn.exe
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, **options)
    run = Run(folder=folder, process=process, n_seeds=len(config["seeds"]),
              log_every=int(config["training"].get("logEvery", 10)))
    threading.Thread(target=_read_output, args=(run,), daemon=True).start()
    return run
