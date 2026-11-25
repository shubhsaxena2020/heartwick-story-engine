# Heartwick round 4: concurrency probe for CPU kernels (runs 4 minutes, prints start/end time).
import time, datetime
print("start", datetime.datetime.utcnow().isoformat(), flush=True)
time.sleep(240)
print("end", datetime.datetime.utcnow().isoformat(), flush=True)
