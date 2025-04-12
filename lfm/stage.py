import pathlib
import os
import time
import libximc.highlevel as ximc
import numpy as np

from opm_acquire.stage import sutterMP285
class SutterStage:
    def __init__(self, verbose=False, overshoot=0.001,com = "COM4"):
        self.stage = sutterMP285(com)
        self.overshoot = overshoot
        self.verbose = verbose

    def __del__(self):
        self.close()

    def close(self):
        del self.stage

    def get_position(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        pos = self.stage.getPosition() * 1000
        if verbose:
            print(f'Current Position: {[f"{p} mm" for p in pos]} (x,y,z)')
        return pos

    def set_origin(self):
        if self.verbose:
            print(f'Setting origin...')
        self.stage.setOrigin()

    def move(self, relative_position, verbose=None):
        if verbose is None:
            verbose = self.verbose
        pos = self.get_position(verbose=verbose)
        new_position = [pos[0] + relative_position[0], pos[1] + relative_position[1], pos[2] + relative_position[2]]
        self.move_to(new_position,verbose=verbose)
        self.get_position(verbose=verbose)

    def move_to(self, new_position, verbose=None):
        if verbose is None:
            verbose = self.verbose
        position = [new_position[0] / 1000, new_position[1] / 1000, new_position[2] / 1000]
        position_over = [position[0] + self.overshoot * np.sign(position[0]),
                         position[1] + self.overshoot * np.sign(position[1]),
                         position[2] + self.overshoot * np.sign(position[2]),]
        self.stage.gotoPosition(position_over)
        self.stage.gotoPosition(position)
        self.get_position(verbose=verbose)


def get_virtual_axes(n):
    uris = {}
    for i in range(n):
        virtual_device_filename = f"virtual_motor_controller_{i+1}.bin"
        virtual_device_file_path = os.path.join(
            pathlib.Path().cwd(),
            virtual_device_filename
        )
        uris[f'VirtualAxis {i+1}'] = f"xi-emu:{virtual_device_file_path}"
    return uris

def get_connected_axes(virtual=0):
    """
    returns list of device URIs. returns virtual one if none are connected
    """
    uris = {}
    devices = ximc.enumerate_devices(
        ximc.EnumerateFlags.ENUMERATE_NETWORK |
        ximc.EnumerateFlags.ENUMERATE_PROBE
    )

    if len(devices) == 0:
        Warning("No devices were found. A virtual device will be used.")
        virtual_device_filename = "virtual_motor_controller_1.bin"
        virtual_device_file_path = os.path.join(
            pathlib.Path().cwd(),
            virtual_device_filename
        )
        uris['VirtualAxis 1'] = f"xi-emu:{virtual_device_file_path}"
    else:
        print("Found {} real device(s):".format(len(devices)))
        for device in devices:
            print("  {}".format(device))
            uris[device["ControllerName"]] = device["uri"]
    return uris

class StandaStage:
    def __init__(self, uris, calibration = [.0025, .0025, .005], verbose=True, overshoot=0.001):
        """
        Class for controlling the StandaStage

        """
        self.verbose = verbose
        if set(uris.keys()) == {'Axis 1', 'Axis 2', 'Axis 3'}:
            self.axis1 = ximc.Axis(uris["Axis 1"]) #x
            self.axis2 = ximc.Axis(uris["Axis 2"]) #y
            self.axis3 = ximc.Axis(uris["Axis 3"]) #z
            self.axes = [self.axis1, self.axis2,self.axis3]

        else:
            self.axes = []
            for axis_name, uri in uris.items():
                clean_name = axis_name.lower().replace(" ", "")
                setattr(self, clean_name, ximc.Axis(uri))
                self.axes.append(getattr(self, clean_name))

        for i,axis in enumerate(self.axes):
            axis.open_device()
            axis.set_calb(calibration[i],axis.get_engine_settings().MicrostepMode)
        self.overshoot = overshoot

    def __del__(self):
        for axis in self.axes:
            axis.close_device()

    def close(self):
        for axis in self.axes:
            axis.close_device()

    def get_position(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        pos = [axis.get_position_calb().Position for axis in self.axes]
        if verbose:
            print(f'Current Position:\n', f"X: {pos[0]:.3f}\n", f"Y: {pos[1]:.3f}\n",f"Z: {pos[2]:.3f}\n")
        return pos

    def set_origin(self):
        if self.verbose:
            print(f'Setting origin...')
        _ = self.get_position()
        for axis in self.axes:
            axis.command_zero()

    def move(self, relative_position, verbose=None):
        if verbose is None:
            verbose = self.verbose
        assert len(relative_position) == len(self.axes), "input must have same shape as .axes"
        for i,axis in enumerate(self.axes):
            if relative_position[i] != 0:
                overshoot = self.overshoot * np.sign(relative_position[i])
                start_time = time.time()
                axis.command_movr_calb(relative_position[i]+overshoot)
                axis.command_wait_for_stop(10)
                axis.command_movr_calb(-overshoot)
                if verbose:
                    print(f"Finished moving axis{i} after {((time.time() - start_time) * 1000):.2f} ms")
            self.get_position(verbose=verbose)

    def move_to(self, new_position,verbose=None):
        if verbose is None:
            verbose = self.verbose
        old_position = self.get_position(verbose=False)
        assert len(new_position) == len(self.axes), "input must have same shape as .axes"
        for i,axis in enumerate(self.axes):
            if new_position[i] != old_position[i]:
                overshoot = self.overshoot * np.sign(new_position[i]-old_position[i])
                start_time = time.time()
                axis.command_move_calb(new_position[i])
                axis.command_wait_for_stop(10)
                axis.command_movr_calb(-overshoot)
                if verbose:
                    print(f"Finished moving axis {i} after {((time.time() - start_time) * 1000):.2f} ms")
            self.get_position(verbose=verbose)

    def get_velocity(self):
        print("Warning: doesnt work for some reason")
        vel = [axis.get_move_settings_calb().Speed for axis in self.axes]
        if self.verbose:
            print(f'Current Velocity: {[f"{v} mm/s" for i,v in vel]} (xyz)')
        return vel

    def set_velocity(self, new_velocity):
        assert len(new_velocity) == len(self.axes), "input must have same shape as .axes"
        for i,axis in enumerate(self.axes):
            move_settings = axis.get_move_settings_calb()
            if self.verbose:
                print(f'Setting velocity for axis {i} from {move_settings.Speed}mm/s to {new_velocity[i]}mm/s')
            move_settings.Speed = new_velocity[i]
            axis.set_move_settings_calb(move_settings)

    def adjust_velocity(self, velocity_factors):
        assert len(velocity_factors) == len(self.axes), "input must have same shape as .axes"
        vel = self.get_velocity()
        new_vel = [vel[i] * velocity_factors[i] for i in range(len(vel))]
        self.set_velocity(new_vel)
