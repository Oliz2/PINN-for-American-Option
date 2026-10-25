"""Compiles the Java pricer, runs it in the background and reads its progress.

The pricing is done entirely by the Java program (java/src); this module only compiles it with javac,
launches it and follows its standard output (TASK and PRICE lines, see MonteCarloApp.java).

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import settings_mc as cfg

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "java" / "src"
CLASSES = HERE / "java" / "out"
MAIN_CLASS = "it.univr.montecarlo.MonteCarloApp"

# Progress lines printed by MonteCarloApp: "TASK i/n | ..." and "PRICE | method | S0 = ... | price (European ...)"
RE_TASK = re.compile(r"TASK (\d+)/(\d+) \| ([^|]+)\|")
RE_PRICE = re.compile(r"PRICE \| ([^|]+) \| S0 = ([^|]+) \| (\S+) \(European (\S+)\)")


def find_jdk() -> str:
    """bin folder of the JDK: first the PATH, then the usual Windows installation folders."""
    javac = shutil.which("javac")
    if javac:
        return str(Path(javac).parent)
    candidates = []
    for base in [r"C:\Program Files\Java", r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Microsoft",
                 r"C:\Program Files\Zulu", r"C:\Program Files\Amazon Corretto"]:
        candidates += glob.glob(os.path.join(base, "jdk*", "bin"))
    candidates = [c for c in candidates
                  if os.path.isfile(os.path.join(c, "javac.exe")) or os.path.isfile(os.path.join(c, "javac"))]
    return sorted(candidates)[-1] if candidates else ""


def executable(jdk_bin: str, name: str) -> str:
    """Full path of java / javac in the JDK folder, or just the name (then it is looked up in the PATH)."""
    if not jdk_bin:
        return name
    for ext in ("", ".exe"):
        p = Path(jdk_bin) / (name + ext)
        if p.is_file():
            return str(p)
    return name


def compile_java(jdk_bin: str) -> str:
    """Compiles the Java sources if the classes are missing or a source is newer. Returns a message."""
    sources = sorted(SOURCES.rglob("*.java"))
    main = CLASSES / "it" / "univr" / "montecarlo" / "MonteCarloApp.class"
    if main.exists() and all(s.stat().st_mtime <= main.stat().st_mtime for s in sources):
        return "Java classes up to date"
    CLASSES.mkdir(parents=True, exist_ok=True)
    cmd = [executable(jdk_bin, "javac"), "-encoding", "UTF-8", "-d", str(CLASSES)] + [str(s) for s in sources]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    r = subprocess.run(cmd, capture_output=True, text=True, **options)
    if r.returncode != 0:
        raise RuntimeError("Java compilation failed:\n" + (r.stderr or r.stdout))
    return "Java classes compiled"


@dataclass
class Run:
    """One run of the Java pricer: process, results folder and state read from its output."""
    folder: Path
    process: subprocess.Popen
    start: float = field(default_factory=time.time)
    lines: list = field(default_factory=list)
    prices: list = field(default_factory=list)      # (method, S0, price, European)
    done: int = 0
    total: int = 0
    current: str = "starting"
    stopped: bool = False

    @property
    def running(self) -> bool:
        """True while the Java program is still working."""
        return self.process.poll() is None

    @property
    def exit_code(self):
        """Exit code of the Java program (None while it runs)."""
        return self.process.poll()

    def progress(self) -> float:
        """Completed fraction (0..1) from the TASK counter."""
        if not self.running:
            return 1.0
        return 0.0 if self.total == 0 else min(1.0, max(self.done - 1, 0) / self.total)

    def stop(self) -> None:
        """Stops the run (button "Stop")."""
        if self.running:
            self.stopped = True
            self.process.kill()


def _read(run: Run) -> None:
    """Thread: reads the stdout of the process, saves it in log.txt and updates the state."""
    log = open(run.folder / "log.txt", "w", encoding="utf-8")
    for line in run.process.stdout:
        line = line.rstrip("\r\n")
        log.write(line + "\n")
        log.flush()
        run.lines.append(line)
        m = RE_TASK.search(line)
        if m:
            run.done, run.total, run.current = int(m.group(1)), int(m.group(2)), line.split("|", 1)[1].strip()
            continue
        m = RE_PRICE.search(line)
        if m:
            run.prices.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3)), float(m.group(4))))
    log.close()


def start(jdk_bin: str, config: dict, runs_folder: Path, memory_mb: int) -> Run:
    """Compiles if needed, creates runs/<date>_<time>_d<d>_<methods>, writes the configuration and starts Java.

    memory_mb is the maximum Java heap (-Xmx): Longstaff-Schwartz keeps all the training paths in memory.
    """
    compile_java(jdk_bin)
    runs_folder = Path(runs_folder).resolve()
    name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_d{config['numberOfAssets']}_{'-'.join(config['methods'])}"
    folder = runs_folder / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (folder / "config.properties").write_text(cfg.properties(config), encoding="utf-8")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.Popen(
        [executable(jdk_bin, "java"), f"-Xmx{int(memory_mb)}m", "-Dstdout.encoding=UTF-8", "-cp", str(CLASSES),
         MAIN_CLASS, str(folder / "config.properties"), str(folder)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1, **options)
    run = Run(folder=folder, process=process)
    threading.Thread(target=_read, args=(run,), daemon=True).start()
    return run
