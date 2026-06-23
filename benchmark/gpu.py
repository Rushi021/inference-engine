import shutil                                                   # detect nvidia-smi
import subprocess                                               # query the GPU
import threading                                                 # sample in the background


def has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None              # present only on NVIDIA GPU hosts


class GpuSampler:
    def __init__(self, interval=0.25):
        self.interval = interval                               # seconds between samples
        self.samples = []                                      # list of (util%, mem_MiB)
        self._stop = threading.Event()                         # stop signal
        self._thread = None                                    # background sampler

    def start(self):
        if not has_nvidia_smi():                               # no-op on CPU/MPS (e.g. this Mac)
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"]             # numeric-only output
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(query, text=True).strip().splitlines()[0]
                util, mem = out.split(",")                     # first GPU only
                self.samples.append((float(util), float(mem)))
            except Exception:
                pass                                          # never let sampling crash the run
            self._stop.wait(self.interval)

    def stop(self):
        if not has_nvidia_smi():                               # nothing sampled
            return None
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return None
        utils = [s[0] for s in self.samples]                  # utilization series
        mems = [s[1] for s in self.samples]                   # memory series
        return {
            "util_mean_pct": round(sum(utils) / len(utils), 1),  # average GPU utilization
            "util_max_pct": round(max(utils), 1),             # peak utilization
            "mem_max_mib": round(max(mems), 1),               # peak memory
            "samples": len(self.samples),                     # sampling coverage
        }
