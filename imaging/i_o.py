import threading
import queue
import h5py
import numpy as np
import time
import sys
from functools import partial
import os, pathlib, socket, glob
import threading, queue
from tqdm import tqdm
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
import os, json
import numpy as np
from collections import defaultdict
from daio.h5 import lazyh5


class Paths():
    def __init__(self, 
                 dataset_name, 
                 url_home, 
                 psf_name='',
                 bg_name='',
                 pn_rec='',
                 pn_psfs='',
                 pn_bg='',
                 pn_out='', 
                 pn_scratch='', 
                 verbosity=0, 
                 expand=True, 
                 create_dirs=True):
        expand = lambda p: str(pathlib.Path(p).expanduser()) if expand else lambda p:str(p)
        
        
        # paths
        self.hostname = socket.gethostname()
        self.dataset_name = dataset_name
        self.pn_recordings = expand(pn_rec)
        self.pn_rec = expand(pathlib.Path(pn_rec, dataset_name))
        self.pn_out = expand(pathlib.Path(pn_out))
        self.pn_outrec = expand(pathlib.Path(self.pn_out, dataset_name))
        scratch = pn_out if not len(pn_scratch) else pn_scratch
        self.pn_scratch = expand(pathlib.Path(scratch, dataset_name))
        self.psf_name = psf_name
        self.pn_psfs = expand(pathlib.Path(pn_psfs))
        self.pn_otfs = expand(pathlib.Path(pn_psfs, 'otfs'))
        self.pn_psf = expand(pathlib.Path(self.pn_psfs, psf_name))
        self.pn_bg = expand(pathlib.Path(pn_bg))
        
        # create directories
        if create_dirs:
            pathlib.Path(self.pn_outrec).mkdir(parents=True, exist_ok=True)

        # files
        self.bgnpy = os.path.join(self.pn_bg, bg_name)
        self.bg = self.bgnpy[:-3]+"h5"
        self.psf = os.path.join(self.pn_psf, 'psf.h5')
        self.meta = os.path.join(self.pn_rec, 'meta.json')
        self.raw = os.path.join(self.pn_rec, 'data.h5')
        self.deconvolved = os.path.join(self.pn_outrec, 'deconvolved.h5')
        self.reg_mask = os.path.join(self.pn_outrec, 'reg_mask.h5')
        self.registered = os.path.join(self.pn_outrec, 'registered.h5')
        self.reg_recipe = os.path.join(self.pn_outrec, 'reg_recipe.h5')
        self.segmentation = os.path.join(self.pn_outrec, 'segmentation.h5')
        self.traces = os.path.join(self.pn_outrec, 'traces.h5')

        #URLs
        self.url_home = url_home
        self.out_url = self.pn_outrec.replace(expand('~'), url_home)
    




class lazymap:
    '''
    lazymap returns an Iterable, functionally related to map (except that outputs are prefetched by a threadpool) 
    and to ThreadPoolExecutor.map (except that the function is being executed lazily, when needed for prefetch).
    
    Args:
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable that maps over the function, providing items as arguments
        prefetch (int): number of threads to prefetch data (default: 1)
        **kwargs: optional, being passed to fcn

    Returns: 
        (iterable): 
    '''

    def __init__(self, fcn, it, prefetch=1, **kwargs):
        self.it = it
        self.generator = self.lazymap_generator(fcn, it, prefetch, **kwargs)

    def __len__(self):
        return len(self.it)

    def __iter__(self):
        return self.generator

    @staticmethod
    def lazymap_generator(fcn, it, prefetch=1, **kwargs):
        it = iter(it)
        with ThreadPoolExecutor(max_workers=prefetch) as executor:
            futures = [executor.submit(fcn, next(it), **kwargs) for i in range(prefetch)]
            k = -1
            for k, item in enumerate(it):
                res = futures[k % prefetch].result()
                futures[k % prefetch] = executor.submit(fcn, item, **kwargs)
                yield res
            for k in range(k + 1, k + 1 + prefetch):
                yield futures[k % prefetch].result()


def lazychain(fcn_list, it):
    '''
    Chain multiple lazymaps.
    
    Args:
        fcn_list (list of callables): functions that are being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable for mapping the first function
    '''
    out = lazymap(fcn_list[0], it)
    for fcn in fcn_list[1:]:
        out = lazymap(fcn, out)
    return out


class VolumeReader:    
    """Iterable. Reads one volume at a time from HDF5 file (with prefetch)

    Args:
        fn (string): HDF5 file path
        key (string): dataset key
        i_frames (iterable): list of frames to include
    """

    def __init__(self, fn, key="data", i_frames=None, prefetch=1, verbose=False):
        self.fn = expandhome(fn)
        self.key = key
        self.verbose = verbose
        if i_frames is None:
            with h5py.File(self.fn, 'r') as f_in:
                i_frames = range(len(f_in[key]))
        self.i_frames = list(i_frames)
        self.len = len(self.i_frames)
        self.prefetch = prefetch
        self._lock = threading.Lock()
        self._gen = self.volume_reader_gen(self.fn, self.key, self.i_frames, self.prefetch, self.verbose)
        self.finished = False  

    def get_next(self):
        """Thread-safe: Get the next item from the generator, returns None after exhaustion."""
        with self._lock:
            if self.finished:
                return None
            try:
                return next(self._gen)
            except StopIteration:
                self.finished = True
                return None

    def __next__(self):
        with self._lock:
            if self.finished:
                raise StopIteration
            try:
                item = next(self._gen)
                return item
            except StopIteration:
                self.finished = True
                raise StopIteration

    def __len__(self):
        return self.len

    def __iter__(self):
        return self
    
    def __del__(self):
        # Attempt to clean up generator and lock references to help GC and avoid tqdm issues
        try:
            if hasattr(self, '_gen'):
                self._gen = None
            if hasattr(self, '_lock'):
                self._lock = None
        except Exception:
            pass

    
    def get_shape(self, key=None):
        """Get the shape of the dataset in the HDF5 file."""
        key = key or self.key
        with h5py.File(self.fn, 'r') as f_in:
            return f_in[key].shape  

    @staticmethod
    def volume_reader_gen(fn, key, i_frames, prefetch, verbose):
        # Debug print to help diagnose argument issues
        with h5py.File(fn, 'r') as f_in:
            lzy_read = lazymap(lambda i: (i, f_in[key][i]), i_frames, prefetch=prefetch)
            for item in lzy_read:
                if verbose:
                    print("Reader: Reading item", item[0])  # Debug 
                    if isinstance(item, tuple) and len(item) == 2:
                        print("Reader: Item data shape", item[1].shape)  # Debug
                yield item
 



def expandhome(path):
    """ Return the full file path including home directory (expand '~')
    """
    return str(pathlib.Path(path).expanduser())


class AsyncH5Writer:
    def __init__(self, filepath, verbose=False):
        self.filepath = filepath
        self._write_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        self._datasets = {}
        self._lock = threading.Lock()
        self.verbose = verbose

    def create_dataset(self, name, shape, dtype=np.float32, **kwargs):
        with h5py.File(self.filepath, 'a') as f:
            if name not in f:
                dset = f.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
            else:
                dset = f[name]
            with self._lock:
                self._datasets[name] = dset.shape
    
    def write_dataset(self, name, data):
        """Write a complete dataset to the file."""
        with h5py.File(self.filepath, 'a') as f:
            if isinstance(data, (str, bytes)):
                # Use string dtype for HDF5
                dt = h5py.string_dtype(encoding='utf-8') if isinstance(data, str) else 'S'
                if name not in f:
                    f.create_dataset(name, data=data, dtype=dt)
                else:
                    f[name][...] = data
                self._write_queue.put(('data', name, slice(None), np.array(data)))
            else:
                if name not in f:
                    f.create_dataset(name, shape=data.shape, dtype=data.dtype)
                self._write_queue.put(('data', name, slice(None), np.array(data)))

    def write(self, dataset_name, data, index):
        self._write_queue.put(('data', dataset_name, index, np.array(data)))

    def write_meta(self, group_name, meta_dict):
        self._write_queue.put(('meta', group_name, meta_dict))

    def _writer_thread(self):
        with h5py.File(self.filepath, 'a') as f:
            while not self._stop_event.is_set() or not self._write_queue.empty():
                try:
                    req = self._write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if req[0] == 'data':
                        _, dataset_name, index, data = req
                        if dataset_name not in f:
                            raise KeyError(f"Dataset {dataset_name} does not exist. Create it first.")
                        f[dataset_name][index] = data
                        f[dataset_name].flush()
                        if self.verbose:
                            print(f"Written data of shape {data.shape} to dataset {dataset_name} at index {index}")
                    elif req[0] == 'meta':
                        _, group_name, meta_dict = req
                        self._write_dict_to_group(f, group_name, meta_dict)
                    f.flush()
                except Exception as e:
                    import traceback
                    print(f"AsyncH5Writer error: {e}")
                    print("Traceback:")
                    traceback.print_exc()
                self._write_queue.task_done()

    def _write_dict_to_group(self, f, group_name, d):
        def recursive_write(g, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    if k not in g:
                        subg = g.create_group(k)
                    else:
                        subg = g[k]
                    recursive_write(subg, v)
                else:
                    # Write as dataset instead of attribute
                    if k in g:
                        del g[k]
                    # Convert to numpy if possible for consistency
                    if isinstance(v, (list, tuple)):
                        v = np.array(v)
                    g.create_dataset(k, data=v)
        if group_name not in f:
            g = f.create_group(group_name)
        else:
            g = f[group_name]
        recursive_write(g, d)

    def flush(self):
        # Wait for all writes to finish and flush file
        self._write_queue.join()
        with h5py.File(self.filepath, 'a') as f:
            f.flush()

    def close(self):
        self._stop_event.set()
        self._write_queue.join()
        self._thread.join()


class InitVolumeManager:
    def __init__(self, size_z, roi_size, reuse_prev_vol):
        self.lock = threading.Lock()
        self.volume = cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
        self.index= -1
        self.reuse_prev_vol = reuse_prev_vol
    
    def update(self, volume, idx):
        if self.reuse_prev_vol:
            if idx > self.index:
                with self.lock:
                    self.volume = volume
                    self.index = idx

    def get(self):   
        with self.lock:
            return self.volume

def get_available_gpus(verbose= True):
    """Robustly detect available GPUs with comprehensive error handling"""
    try:
         # Reset CUDA context before detection
        try:
            cp.cuda.runtime.deviceReset()
            # Add a small delay to let GPU reset
            import time
            time.sleep(1)
        except:
            pass
        
        # Force synchronization
        try:
            cp.cuda.runtime.deviceSynchronize()
        except:
            pass
        
        # Set environment variables for better GPU handling
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        # First check if CUDA is available at all
        print("Checking CUDA availability...") if verbose else None
        
        # Method 1: Check nvidia-smi
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                nvidia_gpu_count = len(gpu_lines)
                print(f"nvidia-smi reports {nvidia_gpu_count} GPUs") if verbose else None
            else:
                nvidia_gpu_count = 0
                print(f"nvidia-smi failed with return code {result.returncode}") if verbose else None
        except Exception as e:
            nvidia_gpu_count = 0
            print(f"nvidia-smi check failed: {e}") if verbose else None
        
        # Method 2: Try to initialize CUDA context
        try:
            print("Attempting CUDA context initialization...") if verbose else None
            
            # Try to reset any existing context
            try:
                cp.cuda.runtime.deviceReset()
            except:
                pass
            
            # Set CUDA device order
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # Try to get device count
            cupy_gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"CuPy reports {cupy_gpu_count} GPUs") if verbose else None
            
            # Test each GPU
            working_gpus = []
            for i in range(cupy_gpu_count):
                try:
                    with cp.cuda.Device(i):
                        # Try a simple operation
                        test_array = cp.ones(10, dtype=cp.float32)
                        result = test_array.sum()
                        print(f"GPU {i}: Working (test sum: {result})") if verbose else None
                        working_gpus.append(i)
                except Exception as e:
                    print(f"GPU {i}: Failed - {e}") 
            
            cupy_gpu_count = len(working_gpus)
            
        except Exception as e:
            print(f"CuPy GPU detection failed: {e}")
            cupy_gpu_count = 0
        
        # Use the minimum of both methods
        n_gpus = min(nvidia_gpu_count, cupy_gpu_count) if nvidia_gpu_count > 0 and cupy_gpu_count > 0 else max(nvidia_gpu_count, cupy_gpu_count)
        
        if n_gpus == 0:
            raise RuntimeError("No GPUs detected or accessible")
        
        print(f"Using {n_gpus} GPU(s)") if verbose else None
        return n_gpus
        
    except Exception as e:
        print(f"GPU detection completely failed: {e}")
        return 1  # Fallback


def get_stimulus(path, fps=40, combinations={}):


    stim_data = lazyh5(os.path.expanduser(os.path.join(path, "stimdata_conditioned.h5")))
    samplerate = stim_data["spec"]["samplerate"]
    reverb_time = int(stim_data["spec"]["reverb_period"] * samplerate)

    stimdict = {}
    stim_names = {}
    for i in range(len(stim_data["stimulus_collection"])):
        stimdict[i] = stim_data["stimulus_collection"][str(i)]["stimulus"]
        stim_names[i] = stim_data["spec"]["stimuli"][str(i)]["label"]

    with open(os.path.expanduser(os.path.join(path, "stimulus_assembly_info.json")), 'r') as f:
        stim_json = json.load(f)

    stimulus_sequence_order = stim_json.get('stimulus_sequence_order', [])
    assembly_info = stim_json.get('assembly_info', [])
    total_duration = stim_json.get("assembly_duration", [])
    final_stimulus = np.zeros(int(total_duration * samplerate))
    final_stimulus_id = np.zeros(int(total_duration * samplerate))

    for i, event_info in enumerate(assembly_info):
        stim_id = stimulus_sequence_order[i]
        if stim_id != 0:
            stim = stimdict[stim_id]
            if stim.shape[0]> 2*reverb_time:
                onset_time = round(event_info.get('onset') * samplerate)
                offset_time = round(event_info.get('offset') * samplerate)
                final_stimulus[onset_time:offset_time] = stim[reverb_time:-reverb_time]
                final_stimulus_id[onset_time:offset_time] = stim_id
            else:
                onset_time = round(event_info.get('onset') * samplerate)
                offset_time = onset_time+stim[reverb_time:].shape[0]
                final_stimulus[onset_time:offset_time] = stim[reverb_time:]
                final_stimulus_id[onset_time:offset_time] = stim_id

    final_stimulus_id = downsample_max(final_stimulus_id, samplerate, fps)
    # trace_mask is not defined in your prompt, so use the same length as final_stimulus_id
    final_stimulus_id_fps = np.zeros_like(final_stimulus_id, dtype=np.uint8)
    final_stimulus_id_fps[:final_stimulus_id.shape[0]] = final_stimulus_id

    # --- Handle combinations ---
    # Reverse lookup: label -> index
    label_to_idx = {v: k for k, v in stim_names.items()}
    stim_names_comb = stim_names.copy()
    stim_idx = max(stim_names.keys()) + 1

    for comb_name, comb_labels in combinations.items():
        stim_names_comb[stim_idx] = comb_name
        stimdict[stim_idx] = comb_labels  # just for completeness, not used further
        stim_idx += 1

    # --- Build bool_stimulus ---
    n_stim = len(stim_names_comb)
    bool_stim = np.zeros((final_stimulus_id_fps.shape[0], n_stim), dtype=bool)

    
    # Individual stimuli (skip silence, which is 0)
    for idx in range(1, len(stim_names)):
        bool_stim[:, idx] = final_stimulus_id_fps == idx
    
    if combinations != {}:
        stim_names[0] = "all"
        # First row: True when not silence
        bool_stim[:, 0] = final_stimulus_id_fps != 0
        
        
        # Combination stimuli
        for i, (comb_name, comb_labels) in enumerate(combinations.items()):
            comb_idx = len(stim_names) + i
            indices = [label_to_idx[lbl] for lbl in comb_labels]
            mask = np.isin(final_stimulus_id_fps, indices)
            bool_stim[:, comb_idx] = mask

    return bool_stim, stim_names_comb, final_stimulus_id_fps, final_stimulus, samplerate

def downsample_max(arr, samplerate, fps):
    window = int(samplerate / fps)
    n_full = arr.shape[0] // window
    arr_main = arr[:n_full * window]
    arr_rest = arr[n_full * window:]
    # Take max in each window
    downsampled = arr_main.reshape(-1, window).max(axis=1)
    # Handle any leftover samples (pad if needed)
    if arr_rest.size > 0:
        downsampled = np.concatenate([downsampled, [arr_rest.max()]])
    return downsampled

def parse_combinations(stim_names, combospec):
    """
    combinations = {
    "R": "angle90.0",
    "L": "angle270.0",
    "400Hz": ("400Hz", "4e+02Hz"),  # OR
    "R_gammatone": ["angle90.0", "gammatone"],  # AND
    "L_400Hz": ["angle90.0", ("400Hz", "4e+02Hz")],  # AND + OR
        } 

    ->

    {'R': ['gammatone_4e+02Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
          'gammatone_8e+02Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
          'gammatone_1.2e+03Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
          'tone_0.9s_400Hz_22PaMonopoleWave2D_distance0.03m_angle90.0',
          'tone_0.9s_800Hz_22PaMonopoleWave2D_distance0.03m_angle90.0',
          'tone_0.9s_1200Hz_22PaMonopoleWave2D_distance0.03m_angle90.0'],
    'L': ['gammatone_4e+02Hz_79Pa-1pi_2.7ms2-1pi_angle270.0',
          'gammatone_8e+02Hz_79Pa-1pi_2.7ms2-1pi_angle270.0',
          'gammatone_1.2e+03Hz_79Pa-1pi_2.7ms2-1pi_angle270.0',
          'tone_0.9s_400Hz_22PaMonopoleWave2D_distance0.03m_angle270.0',
          'tone_0.9s_800Hz_22PaMonopoleWave2D_distance0.03m_angle270.0',
          'tone_0.9s_1200Hz_22PaMonopoleWave2D_distance0.03m_angle270.0'],
     '400Hz': ['gammatone_4e+02Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
              'gammatone_4e+02Hz_79Pa-1pi_2.7ms2-1pi_angle270.0',
              'tone_0.9s_400Hz_22PaMonopoleWave2D_distance0.03m_angle90.0',
              'tone_0.9s_400Hz_22PaMonopoleWave2D_distance0.03m_angle270.0'],
     'R_gammatone': ['gammatone_4e+02Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
                     'gammatone_8e+02Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
                     'gammatone_1.2e+03Hz_79Pa-1pi_2.7ms2-1pi_angle90.0'],
                     'L_400Hz': ['gammatone_4e+02Hz_79Pa-1pi_2.7ms2-1pi_angle90.0',
                     'tone_0.9s_400Hz_22PaMonopoleWave2D_distance0.03m_angle90.0']}
    """
    
    def match(name, cond):
        # cond can be str, tuple, or list
        if isinstance(cond, str):
            return cond in name
        elif isinstance(cond, tuple):
            # OR logic
            return any(match(name, c) for c in cond)
        elif isinstance(cond, list):
            # AND logic
            return all(match(name, c) for c in cond)
        return False

    result = {}
    for key, cond in combospec.items():
        result[key] = [stim_names[i] for i in stim_names if match(stim_names[i], cond)]
    return result















