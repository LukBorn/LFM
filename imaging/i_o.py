import threading
import queue
import h5py
import numpy as np
import time
import sys
from functools import partial
import os, pathlib, socket, glob
import threading, queue
from concurrent.futures import ThreadPoolExecutor

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
        self.pn_rec = expand(pathlib.Path(pn_rec, dataset_name))
        self.pn_out = expand(pathlib.Path(pn_out))
        self.pn_outrec = expand(pathlib.Path(self.pn_out, dataset_name))
        scratch = pn_out if not len(pn_scratch) else pn_scratch
        self.pn_scratch = expand(pathlib.Path(scratch, dataset_name))
        self.psf_name = psf_name
        self.pn_psfs = expand(pathlib.Path(pn_psfs))
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
        self.registered = os.path.join(self.pn_outrec, 'registered.h5')
        self.reg_recipe = os.path.join(self.pn_outrec, 'reg_recipe.json')

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


class volume_reader:
    """Iterable. Reads one volume at a time from HDF5 file (with prefetch)

    Args:
        fn (string): HDF5 file path
        key (string): dataset key
        i_frames (iterable): list of frames to include
    """

    def __init__(self, fn, key, i_frames=None, prefetch=1):
        fn = expandhome(fn)
        if i_frames is None:
            with h5py.File(fn, 'r') as f_in:
                i_frames = range(len(f_in[key]))
        self.len = len(i_frames)
        self.prefetch = prefetch
        self.generator = partial(self.volume_reader_gen, fn, key, i_frames)
        self._queue = queue.Queue(maxsize=prefetch)
        self._stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(fn, key, i_frames), daemon=True)
        self._prefetch_thread.start()

    def _prefetch_worker(self, fn, key, i_frames):
        with h5py.File(fn, 'r', swmr=True) as f_in:
            for i in i_frames:
                if self._stop_event.is_set():
                    break
                vol = f_in[key][i]
                self._queue.put((i, vol))
            self._queue.put(None)  # Sentinel to signal end

    def get_next_prefetched(self, timeout=60):
        """
        Thread-safe: Get the next available prefetched (idx, volume) tuple.
        Blocks if none are ready. Returns None when finished.
        """
        item = self._queue.get(timeout=timeout)
        return item
    
    def put_back(self, item):
        """
        Put an item (idx, volume) back into the prefetch queue.
        Useful if a worker cannot process an item and wants to return it for another worker.
        """
        self._queue.put(item)

    def stop_prefetch(self):
        self._stop_event.set()
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join()

    def __len__(self):
        return self.len

    def __iter__(self):
        return self.generator()
    
    def get_shape(key):
        """Get the shape of the dataset in the HDF5 file."""
        with h5py.File(self.fn, 'r') as f_in:
            return f_in[key].shape  

    @staticmethod
    def volume_reader_gen(fn, key, i_frames):
        with h5py.File(fn, 'r', swmr=True) as f_in:
            lzy_read = lazymap(lambda i: (i, f_in[key][i]), i_frames, 1)
            for item in lzy_read:
                yield item
 





def expandhome(path):
    """ Return the full file path including home directory (expand '~')
    """
    return str(pathlib.Path(path).expanduser())


class AsyncH5Writer:
    def __init__(self, filepath):
        self.filepath = filepath
        self._write_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        self._datasets = {}
        self._lock = threading.Lock()

    def create_dataset(self, name, shape, dtype=np.float32, **kwargs):
        with h5py.File(self.filepath, 'a') as f:
            if name not in f:
                dset = f.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
            else:
                dset = f[name]
            with self._lock:
                self._datasets[name] = dset.shape

    def write(self, data, dataset_name, index):
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
                    elif req[0] == 'meta':
                        _, group_name, meta_dict = req
                        self._write_dict_to_group(f, group_name, meta_dict)
                    f.flush()
                except Exception as e:
                    print(f"AsyncH5Writer error: {e}")
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
                    g.attrs[k] = v
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
