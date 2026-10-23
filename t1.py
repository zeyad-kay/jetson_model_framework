"""Send SIGUSR1 to t2.py. Usage: python t1.py [pid]  (defaults to the pid in t2.pid)."""

import os
import signal
import sys

PID_FILE = "t2.pid"

if len(sys.argv) > 1:
    pid = int(sys.argv[1])
else:
    with open(PID_FILE) as f:
        pid = int(f.read())

os.kill(pid, signal.SIGUSR1)
print(f"t1: sent SIGUSR1 to pid {pid}")
