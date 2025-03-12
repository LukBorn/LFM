import multiprocessing as mp
import traceback
import hdf5plugin
import h5py
import numpy as np
import zstandard
from multiprocessing import shared_memory


class VanillaWriter:

    def __init__(self, fn, name, dtype, shape):
        self.h5f = h5py.File(fn, "a")
        self.ds = self.h5f.create_dataset(name=name, dtype=dtype, shape=shape)

    def write_frame(self, frame, iVol, iPlane):
        self.ds[iVol, iPlane] = frame

    def terminate(self):
        self.h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()


class ParallelCompressedWriter:

    def __init__(self, fn, name, dtype, shape, chunk_shape, clevel=1, num_workers=4, queue_bytes=2**30):
        self.input_queue = ByteQueue(queue_bytes)
        self.output_queue = ByteQueue(queue_bytes)
        self.chunk_shape = chunk_shape
        self.wp = WriterProcess(self.output_queue, fn, name=name, dtype=dtype, shape=shape, chunks=chunk_shape, **hdf5plugin.Zstd(clevel=clevel))
        self.wp.start()
        self.cp = [CompressorProcess(self.input_queue, self.output_queue, clevel) for _ in range(num_workers - 1)]
        for p in self.cp:
            p.start()
        self.terminated = False

    def write_chunk(self, frame, chunk_offsets):
        if self.terminated:
            raise RuntimeError("Cannot write to terminated ParallelCompressedWriter")
        assert frame.shape == self.chunk_shape
        assert len(self.chunk_shape) == len(chunk_offsets)
        self.input_queue.put(frame, chunk_offsets)

    def write_frame(self, frame, iVol, iPlane):
        self.write_chunk(frame[None,None], (iVol, iPlane, 0, 0))

    def terminate(self):
        for _ in self.cp:
            self.input_queue.put(np.ones(1, dtype="uint8"), None)
        for p in self.cp:
            p.join()
        self.output_queue.put(np.ones(1, dtype="uint8"), None)
        self.wp.join()
        self.terminated = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def __del__(self):
        if not self.terminated:
            self.terminate()


class CompressorProcess(mp.Process):
    def __init__(self, input_queue, output_queue, clevel):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.clevel = clevel

    def run_(self):
        cctx = zstandard.ZstdCompressor(level=self.clevel, threads=0)
        while True:
            frame, meta = self.input_queue.get()
            if meta is None:
                break
            compressed_frame = cctx.compress(frame)
            self.output_queue.put(np.frombuffer(compressed_frame, dtype="uint8"), meta=meta)

    def run(self):
        try:
            self.run_()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


class WriterProcess(mp.Process):
    def __init__(self, output_queue, fn, **kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs
        self.output_queue = output_queue

    def run_(self):
        import hdf5plugin

        with h5py.File(self.fn, "a") as f:
            ds = f.create_dataset(**self.kwargs)
            did = ds.id

            def callback(arr, meta):
                if meta is None:
                    return
                did.write_direct_chunk(meta, arr.ravel())

            while True:
                _, meta = self.output_queue.get(copy=False, callback=callback)
                if meta is None:
                    break

    def run(self):
        try:
            self.run_()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


class ByteQueue:
    """
    A class representing a byte queue. The queue is implemented as a circular buffer in shared memory.
    """

    def __init__(self, buffer_size):
        """
        Initializes a ByteQueue object.

        Args:
            buffer_size (int): The size of the buffer in bytes.
        """

        self.buffer_size = buffer_size
        self.buffer = mp.Array("c", buffer_size, lock=False)
        self._view = None
        self.queue = mp.Manager().Queue()  # manager helps avoid out-of-order problems
        self.get_lock = mp.Lock()
        self.put_lock = mp.Lock()
        self.head_changed = mp.Condition()
        self.head = mp.Value("i", 0)
        self.tail = mp.Value("i", 0)

    def put(self, array, meta=None):
        """
        Puts a byte array into the queue.

        Args:
            array (numpy.ndarray): The byte array to be put into the queue.
            meta (Any, optional): Additional metadata associated with the byte array.

        Raises:
            AssertionError: If the size of the byte array exceeds the buffer size.
        """

        array_bytes = array.ravel().data
        nbytes = array.nbytes

        assert nbytes < self.buffer_size

        with self.put_lock:
            while self._available_space() < nbytes:
                with self.head_changed:
                    self.head_changed.wait()
            tail = self.tail.value

            if tail + nbytes <= self.buffer_size:
                self.view[tail : tail + nbytes] = array_bytes
                self.tail.value = (tail + nbytes) % self.buffer_size
            else:
                tail_part_size = self.buffer_size - tail
                self.view[tail:] = array_bytes[:tail_part_size]
                self.view[: nbytes - tail_part_size] = array_bytes[tail_part_size:]
                self.tail.value = nbytes - tail_part_size

            array_info = dict(
                dtype=array.dtype.str, shape=array.shape, nbytes=nbytes, head=tail, tail=self.tail.value, meta=meta
            )
            self.queue.put(array_info)

    def get(self, callback=None, copy=None, **kwargs):
        """
        Gets a byte array from the queue.

        Args:
            callback (Callable, optional): A callback function to be called with the byte array (pre-copy, potentially unsafe!) and metadata.
            copy (bool, optional): Whether to make a copy of the byte array. Defaults to None: copy if a callback is not provided.
            **kwargs: Additional keyword arguments to be passed to the queue's get method.

        Returns:
            tuple: A tuple containing the byte array and any metadata provided with put.
        """
        with self.get_lock:
            array_info = self.queue.get(**kwargs)
            head = array_info["head"]
            tail = array_info["tail"]
            try:
                assert head == self.head.value
            except AssertionError:
                print(f"head: {head}, self.head: {self.head.value}")
                raise

            if head < tail:
                array_bytes = self.view[head:tail]
            else:
                array_bytes = np.concatenate((self.view[head:], self.view[:tail]))
            array = np.frombuffer(array_bytes, dtype=array_info["dtype"]).reshape(array_info["shape"])
            if copy or ((copy is None) and (callback is None)):
                array = array.copy()
            if callback is not None:
                callback(array, array_info["meta"])
            self.head.value = (head + array_info["nbytes"]) % self.buffer_size

        with self.head_changed:
            self.head_changed.notify()

        return array, array_info["meta"]

    def _available_space(self):
        """
        Calculates the available space in the buffer.

        Returns:
            int: The available space in the buffer.
        """
        return (self.head.value - self.tail.value - 1) % self.buffer_size

    @property
    def view(self):
        """
        numpy.ndarray: A view of the shared memory array as a numpy array. Lazy initialization to avoid pickling issues.
        """
        if self._view is None:
            self._view = np.frombuffer(self.buffer, "byte")
        return self._view

    def __del__(self):
        self._view = None
