"""Wait for SIGUSR1 sent by t1.py. Run this first, then run t1.py."""

import os
import signal
import sys

PID_FILE = "t2.pid"


def on_sigusr1(signum, frame):
    print("t2: received SIGUSR1 from t1", flush=True)


signal.signal(signal.SIGUSR1, on_sigusr1)

with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))
print(f"t2: pid {os.getpid()}, waiting for SIGUSR1 (Ctrl+C to quit)", flush=True)

try:
    while True:
        signal.pause()
except KeyboardInterrupt:
    pass
finally:
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    sys.exit(0)
