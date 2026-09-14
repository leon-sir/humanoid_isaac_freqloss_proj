"""Run one training command after an existing Linux process exits.

Uses pidfd rather than polling a PID that could be reused. Exit of the watched
process (including failure) triggers the command; its exit code is unavailable
because it is not our child. No retry or recurring schedule is installed.
"""

import argparse
import datetime
import os
from pathlib import Path
import select
import subprocess
import time


def report(message):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    # Fail closed if the requested process has already gone away.
    fd = os.pidfd_open(args.pid) if hasattr(os, 'pidfd_open') else None
    try:
        command = Path(f'/proc/{args.pid}/cmdline').read_bytes().replace(b'\0', b' ')
        if b'scripts/rsl_rl/train.py' not in command:
            raise ValueError('Watched process is not the expected training script')
        report(f'Waiting for PID {args.pid}: {command.decode()}')
        if fd is not None:
            select.select([fd], [], [])
        else:
            def identity():
                try:
                    fields = Path(f'/proc/{args.pid}/stat').read_text().rsplit(')', 1)[1].split()
                    return None if fields[0] == 'Z' else fields[19]
                except FileNotFoundError:
                    return None
            start = identity()
            if start is None:
                raise RuntimeError('Process vanished before watcher initialization')
            while identity() == start:
                time.sleep(10)
    finally:
        if fd is not None:
            os.close(fd)
    report('Previous process exited; starting V1 once (fresh training, no resume).')
    result = subprocess.run(
        ['bash', '-c',
         'source /home/ps/anaconda3/etc/profile.d/conda.sh && '
         'conda activate env_isaaclab_v23 && '
         'exec python -u scripts/rsl_rl/train.py '
         '--task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1 --headless'],
        cwd=project,
    )
    report(f'New training exited with code {result.returncode}; no retry.')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
