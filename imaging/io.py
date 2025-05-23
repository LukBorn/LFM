import threading
import queue
import h5py
import numpy as np
import time
import sys

class Reader:
    def __init__(self, 
                 file_path, 
                 max_workers=4, 
                 function=lambda x: x, 
                 output_dtype=np.float32):
        self.file_path = file_path
        self.max_workers = max_workers
        self.function = function
        self.output_dtype = output_dtype
        self.preload_queue = queue.Queue(maxsize=max_workers)
        self.preload_thread = None
        self.stop_event = threading.Event()
        self.preload_started = False

    def _preload_worker(self, idx_list):
        for idx in idx_list:
            if self.stop_event.is_set():
                break
            with h5py.File(self.file_path, 'r') as f:     
                arr = np.array(f["data"][idx], dtype=self.output_dtype)
            arr = self.function(arr)
            self.preload_queue.put((idx, arr))
        self.preload_started = False

    def start_preloading(self, idx_list):
        """Preload and preprocess frames in idx_list asynchronously."""
        self.stop_event.clear()
        self.preload_thread = threading.Thread(target=self._preload_worker, args=(idx_list,), daemon=True)
        self.preload_thread.start()
        self.preload_started = True

    def get(self, idx):
        """Get preloaded frame by idx (blocks until available)."""
        while True:
            i, arr = self.preload_queue.get()
            if i == idx:
                return arr
            else:
                # Put back if not the requested idx
                self.preload_queue.put((i, arr))
                time.sleep(0.01)

    def time_one(self, idx):
        """Time and report memory usage for one I/O + preprocessing operation."""
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss
        t0 = time.time()
        with h5py.File(self.file_path, 'r') as f:
            arr = np.array(f["data"][idx], dtype=self.output_dtype)
            if self.function:
                arr = self.function(arr)
        t1 = time.time()
        mem_after = process.memory_info().rss
        print(f"Reader: idx={idx}, time={t1-t0:.4f}s, RAM used={mem_after-mem_before} bytes, arr.nbytes={arr.nbytes}")
        return arr

    def stop(self):
        self.stop_event.set()
        if self.preload_thread is not None:
            self.preload_thread.join()

class Writer:
    def __init__(self, 
                 file_path, 
                 dataset_shape, 
                 dtype=np.float32, 
                 max_workers=2, 
                 function=lambda x: x, 
                 ):
        self.file_path = file_path
        self.dataset_shape = dataset_shape
        self.dtype = dtype
        self.function = function
        self.task_queue = queue.Queue(maxsize=max_workers)
        self.stop_event = threading.Event()
        self.workers = []
        self._start_workers()
        # Create file and dataset if not exists
        with h5py.File(self.file_path, 'a') as f:
            if "data" not in f:
                f.create_dataset("data", shape=dataset_shape, dtype=dtype)

    def _worker(self):
        with h5py.File(self.file_path, 'a') as f:
            dset = f["data"]
            while not self.stop_event.is_set():
                try:
                    idx, arr = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue    
                arr = self.function(arr)
                dset[idx] = arr
                dset.flush()
                self.task_queue.task_done()

    def _start_workers(self):
        for _ in range(2):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def write(self, arr, idx):
        self.task_queue.put((idx, arr))

    def time_one(self, arr, idx):
        """Time and report memory usage for one write operation."""
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss
        t0 = time.time()
        with h5py.File(self.file_path, 'a') as f:
            dset = f["data"]
            if self.function:
                arr = self.function(arr)
            dset[idx] = arr
            dset.flush()
        t1 = time.time()
        mem_after = process.memory_info().rss
        print(f"Writer: idx={idx}, time={t1-t0:.4f}s, RAM used={mem_after-mem_before} bytes, arr.nbytes={arr.nbytes}")

    def stop(self):
        self.stop_event.set()
        for t in self.workers:
            t.join()

    def wait(self):
        self.task_queue.join()

    def __del__(self):
        self.stop()