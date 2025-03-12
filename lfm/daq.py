import numpy as np

import nidaqmx
import nidaqmx.stream_writers
from nidaqmx.constants import AcquisitionType, RegenerationMode, Edge


def cfg_samp_clk_timing_and_check(task, rate, **kwargs):
    """Set the sample rate for a task and verify if the actual rate matches the desired rate.

    Args:
        task: The task object for which the sample rate is to be set.
        rate (float): The desired sample rate to be configured for the task.
        **kwargs: Additional keyword arguments to be passed to the task's timing.cfg_samp_clk_timing method.

    Raises:
        AssertionError: If the actual sample rate does not match the desired sample rate.
    """
    task.timing.cfg_samp_clk_timing(rate=rate, **kwargs)
    actual_rate = task.timing.samp_clk_rate
    assert rate == actual_rate, f"mismatch between actual ({actual_rate}) and desired ({rate}) sample rate"


class unifiedDAO:

    def __init__(self, rate, ao_chans, do_port='port0', shutter_line='PFI0', parent=True, buffer_size_seconds=1, n_chunks_in_buffer=5):
        """ Initialize UnifiedDAO (digital and analog output) object and create tasks.

        Args:
            ao_chans (str): String specifying the AO channels.
            do_port (str, optional): The DO port to use for buffered DO signals. Defaults to 'port0'.
            shutter_line (str, optional): The DO channel to use for the shutter. Defaults to 'PFI0'.
            parent (bool, optional): Whether this is the parent device exporting clock and trigger. Defaults to True.
            buffer_size_seconds (int): The size of the buffer in seconds. Defaults to 1.
            n_chunks_in_buffer (int): The number of chunks per buffer (at least 2). Defaults to 5.

        Example:
            dao = unifiedDAO(rate=100000, ao_chans='Dev1/ao0:3', do_port='port0', shutter_line='PFI0')
            with dao.queue_data(ao_data, do_data):
                pass
        """
        self.rate = rate
        self.dev = ao_chans.split('/')[0]
        self.ao_chans = ao_chans
        self.do_port = do_port
        self.parent = parent
        self.buffer_size_seconds = buffer_size_seconds
        self.n_chunks_in_buffer = n_chunks_in_buffer
        # Create shutter task if shutter_line is given
        if shutter_line:
            self.task_shutter = nidaqmx.Task()
            self.task_shutter.do_channels.add_do_chan(f"{self.dev}/{shutter_line}")
            self.writer_shutter = nidaqmx.stream_writers.DigitalSingleChannelWriter(self.task_shutter.out_stream)
            self.shutter_open = lambda: self.writer_shutter.write_one_sample_one_line(1)
            self.shutter_close = lambda: self.writer_shutter.write_one_sample_one_line(0)
            self.shutter_close()

    def queue_data(self, ao_data, do_data=None, finite=True, chunked=False, trigger_delay_sec=0.0):
        """ Queue data to be written to the task.

        Args:
            ao_data (numpy.ndarray): 2D array containing analog data. Shape should be (channels, samples). Skip to requeue previously queued data.
            do_data (numpy.ndarray, optional): 2D array containing digital data. Shape should be (channels, samples).
            finite (bool, optional): Whether or not the task should be finite. Defaults to True.
            chunked (bool, optional): Whether or not to write the data in chunks and use callbacks. Defaults to False.
            trigger_delay_sec (float, optional): The number of seconds to delay the trigger. Defaults to 0.
        """
        acquisition_type = AcquisitionType.FINITE if finite else AcquisitionType.CONTINUOUS
        regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION if finite else RegenerationMode.ALLOW_REGENERATION

        #create AO task
        self.task_ao = nidaqmx.Task()
        self.task_ao.ao_channels.add_ao_voltage_chan(self.ao_chans)
        #self.task_ao.timing.cfg_samp_clk_timing(self.rate, sample_mode=acquisition_type, samps_per_chan=ao_data.shape[1])
        cfg_samp_clk_timing_and_check(self.task_ao, self.rate, sample_mode=acquisition_type, samps_per_chan=ao_data.shape[1])
        self.writer_ao = nidaqmx.stream_writers.AnalogMultiChannelWriter(self.task_ao.out_stream)
        self.task_ao.out_stream.regen_mode = regen_mode
        if self.parent:
            assert trigger_delay_sec == 0.0, "Cannot delay trigger on parent device."
            #self.task_ao.export_signals.export_signal(signal_id=nidaqmx.constants.Signal.TEN_MHZ_REF_CLOCK, output_terminal='PFI5')
            nidaqmx.system.System().connect_terms(f'/{self.dev}/100kHzTimebase', f'/{self.dev}/PFI5')
            self.task_ao.export_signals.export_signal(signal_id=nidaqmx.constants.Signal.START_TRIGGER, output_terminal='PFI6')
            self.task_ao.export_signals.export_signal(signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK, output_terminal='PFI7')
        else:
            self.task_ao.timing.samp_clk_timebase_src = 'PFI5'
            self.task_ao.timing.samp_clk_timebase_rate = 100e3
            self.task_ao.triggers.start_trigger.cfg_dig_edge_start_trig('PFI6', Edge.RISING)
            self.task_ao.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
            self.task_ao.triggers.start_trigger.delay = np.maximum(trigger_delay_sec, 1e-5)
        
        if chunked:
            self.task_ao.out_stream.output_buf_size = int(self.buffer_size_seconds * self.rate * len(self.task_ao.ao_channels.channel_names))
        else:
            self.writer_ao.write_many_sample(ao_data)

        # Create DO task
        if do_data is not None:
            self.task_do = nidaqmx.Task()
            self.task_do.do_channels.add_do_chan(f"{self.dev}/{self.do_port}")
            self.task_do.timing.cfg_samp_clk_timing(self.rate, sample_mode=acquisition_type, samps_per_chan=do_data.shape[1])
            cfg_samp_clk_timing_and_check(self.task_do, self.rate, sample_mode=acquisition_type, samps_per_chan=do_data.shape[1])
            self.task_do.triggers.start_trigger.cfg_dig_edge_start_trig('ao/StartTrigger', Edge.RISING)
            # self.task_do.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
            # self.task_do.triggers.start_trigger.delay = np.maximum(trigger_delay_sec, 1e-6)
            self.task_do.out_stream.regen_mode = regen_mode
            self.writer_do = nidaqmx.stream_writers.DigitalSingleChannelWriter(self.task_do.out_stream)
            if chunked:
                self.task_do.out_stream.output_buf_size = int(self.buffer_size_seconds * self.rate)
            else:
                port_data = np.packbits(do_data, axis=0, bitorder='little')[0].astype('uint32')
                self.writer_do.write_many_sample_port_uint32(port_data)

        # Set up callback for writing in chunks
        if chunked:
            current_index = [0]  # mutable list to hold the current index
            chunk_size = (self.rate * self.buffer_size_seconds) // self.n_chunks_in_buffer

            def callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
                end_idx = min(current_index[0] + chunk_size, ao_data.shape[1])
                if current_index[0] < ao_data.shape[1]:
                    analog_chunk = ao_data[:, current_index[0]:end_idx].copy()
                    self.writer_ao.write_many_sample(analog_chunk)
                if (do_data is not None) and (current_index[0] < do_data.shape[1]):
                    do_chunk = do_data[:, current_index[0]:end_idx].copy()
                    port_data = np.packbits(do_chunk, axis=0, bitorder='little')[0].astype('uint32')
                    self.writer_do.write_many_sample_port_uint32(port_data)
                current_index[0] += chunk_size  # update index
                return 0  # The callback should return an integer

            self.task_ao.register_every_n_samples_transferred_from_buffer_event(chunk_size, callback)
            for _ in range(min(self.n_chunks_in_buffer, int(np.ceil(ao_data.shape[1] / chunk_size)))):
                callback(self.task_ao, None, None, None)

        return self
    

    def __enter__(self):
        self.start()
        if hasattr(self, 'task_shutter'):
            self.shutter_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, 'task_shutter'):
            self.shutter_close()
        self.stop()
        self.close()

    def start(self):
        if hasattr(self, 'task_do'):
            self.task_do.start()
        self.task_ao.start()

    def stop(self):
        if hasattr(self, 'task_ao'):
            self.task_ao.stop()
        if hasattr(self, 'task_do'):
            self.task_do.stop()

    def close(self):
        if hasattr(self, 'task_ao'):
            self.task_ao.close()
        if hasattr(self, 'task_do'):
            self.task_do.close()

    def __del__(self):
        self.stop()
        self.close()
        if hasattr(self, 'task_shutter'):
            self.task_shutter.close()