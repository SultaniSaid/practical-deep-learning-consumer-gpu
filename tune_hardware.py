import os
import time
import json
import torch
import torch_directml
import psutil
import threading
import subprocess
import traceback
from pathlib import Path
from fastai.vision.all import *
from dml_fastai_utils import setup_dml, optimize_dls, get_local_path

# Try to import Rich for "Wonderful Matrix"
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Configuration
LOG_FILE = "tuning_log.json"
GPU_DATA_SUBSET = 1000
CPU_DATA_SUBSET = 200 
NUM_EPOCHS = 3 # Increased as requested to see quality metrics

def label_func(x): return x[0].isupper()

class SystemMonitor:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.records = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                gpu_util = 0.0
                try:
                    cmd = 'powershell -Command "(Get-Counter \\"\\GPU Engine(*)\\Utilization Percentage\\").CounterSamples | Where-Object {$_.InstanceName -match \\"3d\\"} | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"'
                    res = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
                    if res: gpu_util = float(res)
                except: pass
                self.records.append({"cpu": cpu, "ram": ram, "gpu": gpu_util})
            except: pass
            time.sleep(self.interval)

    def start(self):
        self.records = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=2)
        return self.get_stats()

    def get_stats(self):
        if not self.records: return {"avg_cpu": 0, "avg_ram": 0, "avg_gpu": 0}
        avg_cpu = sum(r["cpu"] for r in self.records) / len(self.records)
        avg_ram = sum(r["ram"] for r in self.records) / len(self.records)
        avg_gpu = sum(r["gpu"] for r in self.records) / len(self.records)
        return {"avg_cpu": round(avg_cpu, 1), "avg_ram": round(avg_ram, 1), "avg_gpu": round(avg_gpu, 1)}

def get_grid():
    devices = ["gpu", "cpu"]
    batch_sizes = [16, 32, 64] # Focused grid for 3-epoch runs
    worker_options = [0, 4]
    precision_options = ["fp32", "fp16"]
    grid = []
    for dev in devices:
        for bs in batch_sizes:
            for nw in worker_options:
                for pr in precision_options:
                    grid.append({"device": dev, "bs": bs, "num_workers": nw, "pin_memory": False, "precision": pr})
    return grid

def load_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                content = json.load(f)
                return content if isinstance(content, list) else []
        except: return []
    return []

def save_log(entry):
    logs = load_logs()
    for i, l in enumerate(logs):
        if l.get("params") == entry["params"]:
            logs[i] = entry
            with open(LOG_FILE, 'w') as f: json.dump(logs, f, indent=4)
            return
    logs.append(entry)
    with open(LOG_FILE, 'w') as f: json.dump(logs, f, indent=4)

def run_benchmark(path, params, monitor):
    dev_type = params["device"]
    bs, nw, pm, precision = params["bs"], params["num_workers"], params["pin_memory"], params["precision"]
    
    target_device = setup_dml(dev_type)
    subset_size = GPU_DATA_SUBSET if dev_type == "gpu" else CPU_DATA_SUBSET
    
    print(f"\n[BENCHMARK] {dev_type.upper()} | BS={bs} | Workers={nw} | Bits={precision} | Epochs={NUM_EPOCHS}")
    
    try:
        images = get_image_files(path/"images")[:subset_size]
        dls_base = ImageDataLoaders.from_name_func(
            path, images, valid_pct=0.2, seed=42, label_func=label_func,
            item_tfms=Resize(224), bs=bs, num_workers=0, device="cpu"
        )
        
        dls = DataLoaders(
            dls_base.train.new(device=target_device),
            dls_base.valid.new(device=target_device),
            path=dls_base.path, device=target_device
        )
        dls = optimize_dls(dls, num_workers=nw, pin_memory=pm, persistent_workers=False)
        
        learn = vision_learner(dls, resnet18, metrics=accuracy)
        if precision == "fp16": learn = learn.to_fp16()
            
        monitor.start()
        start_time = time.perf_counter()
        learn.fit(NUM_EPOCHS, lr=1e-3)
        elapsed = time.perf_counter() - start_time
        stats = monitor.stop()
        
        # Get final accuracy
        final_acc = float(learn.recorder.metrics[0].value) if learn.recorder.metrics else 0.0
        
        result = {
            "params": params, "status": "success", "time_sec": round(elapsed, 2),
            "img_per_sec": round((subset_size * NUM_EPOCHS) / elapsed, 2), 
            "utilization": stats, "accuracy": round(final_acc, 4)
        }
        print(f"  -> SUCCESS: {result['img_per_sec']} img/s | Acc: {result['accuracy']}")
        return result

    except Exception as e:
        monitor.stop()
        error_msg = str(e)
        status = "failed"
        if "quota" in error_msg.lower(): status = "failed_vram"
        elif "OpaqueTensorImpl" in error_msg: status = "unsupported_config"
        print(f"  -> FAILED: {status}")
        return {"params": params, "status": status, "error": error_msg}

def print_rich_matrix(logs):
    if not HAS_RICH:
        print("\n[Note] Install 'rich' for the Wonderful Matrix view.")
        return

    console = Console()
    table = Table(title="DIRECTML HARDWARE TUNING MATRIX (v7.0)", header_style="bold magenta", border_style="cyan")
    
    table.add_column("Device", justify="center", style="bold")
    table.add_column("BS", justify="center")
    table.add_column("NW", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Speed (img/s)", justify="right", style="green")
    table.add_column("Acc (3ep)", justify="right", style="yellow")
    table.add_column("GPU %", justify="right")
    table.add_column("CPU %", justify="right")
    table.add_column("Status", justify="left")

    logs.sort(key=lambda x: (x.get("params", {}).get("device", "gpu") == "cpu", -x.get("img_per_sec", 0)))

    for l in logs:
        p = l.get("params", {})
        u = l.get("utilization", {})
        status = l.get("status", "unknown")
        
        color = "white"
        if status == "success": color = "green"
        elif "quota" in status: color = "red"
        elif "unsupported" in status: color = "yellow"

        table.add_row(
            p.get("device", "").upper(),
            str(p.get("bs", "")),
            str(p.get("num_workers", "")),
            p.get("precision", ""),
            f"{l.get('img_per_sec', 0.0):.2f}" if status == "success" else "-",
            f"{l.get('accuracy', 0.0):.4f}" if status == "success" else "-",
            f"{u.get('avg_gpu', 0.0)}%" if status == "success" else "-",
            f"{u.get('avg_cpu', 0.0)}%" if status == "success" else "-",
            f"[{color}]{status.upper()}[/{color}]"
        )

    console.print("\n")
    console.print(Panel(table, expand=False))
    console.print("\n[bold cyan]Winner Strategy:[/bold cyan] Check BS=64/FP16 for GPU stability vs BS=128/FP32 for CPU raw speed.")

def main():
    path = untar_data(URLs.PETS, data=get_local_path())
    grid, current_logs = get_grid(), load_logs()
    
    def is_done(p):
        for l in current_logs:
            if l.get("params") == p and l.get("status") in ["success", "unsupported_config", "failed_vram"]:
                return True
        return False

    pending = [p for p in grid if not is_done(p)]
    monitor = SystemMonitor()
    
    print(f"\n[STARTING V7.0 TUNER] Total Tests: {len(grid)} | Remaining: {len(pending)}")
    
    try:
        for i, params in enumerate(pending):
            print(f"\nProgress: {len(grid) - len(pending) + i + 1}/{len(grid)}")
            result = run_benchmark(path, params, monitor)
            save_log(result)
            time.sleep(1)
    except KeyboardInterrupt: print("\n[!] Interrupted.")
    
    print_rich_matrix(load_logs())

if __name__ == "__main__":
    main()
