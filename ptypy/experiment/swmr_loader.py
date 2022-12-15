# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the Diamond beamlines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import h5py as h5

from ptypy.experiment import register
from ptypy.experiment.hdf5_loader import Hdf5Loader
from ptypy.utils.verbose import log

try:
    from swmr_tools import KeyFollower

except ImportError:
    log(3, "The SWMR loader requires swmr_tools to be installed,"
           " try pip install swmr_tools")
    raise ImportError


@register()
class SwmrLoader(Hdf5Loader):
    """
    This is an attempt to load data from a live SWMR file that is still being written to.

    Defaults:

    [name]
    default = 'SwmrLoader'
    type = str
    help =

    [intensities.live_key]
    default = None
    type = str
    help = Key to live keys inside the intensities file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.live_fast_key]
    default = None
    type = str
    help = Key to live key for fast axis inside the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.live_slow_key]
    default = None
    type = str
    help = Key to live key for slow axis inside the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)

    def setup(self, *args, **kwargs):
        self._is_swmr = True
        self._use_keyfilter = False if self.p.framefilter else True
        self.available = None
        # if no framefilter passed, use key follower to filter frames by index
        super().setup(*args, **kwargs)
        # Check if we have been given the live keys
        if None in [self.p.intensities.live_key,
                    self.p.positions.live_slow_key,
                    self.p.positions.live_fast_key]:
            raise RuntimeError("Missing live keys to intensities or positions")

        # Check that intensities and positions (and their live keys)
        # are loaded from the same file
        if self.p.intensities.file != self.p.positions.file:
            raise RuntimeError("Intensities and positions file should be same")

        # Initialize KeyFollower

        intensity_file = h5.File(self.p.intensities.file, 'r', swmr=self._is_swmr)
        positions_file = h5.File(self.p.positions.file, 'r', swmr=self._is_swmr)
        self.kf = KeyFollower((intensity_file[self.p.intensities.live_key],
                               positions_file[self.p.positions.live_slow_key],
                               positions_file[self.p.positions.live_fast_key]),
                              timeout=5)

        # Get initial value of maximum number of frames to be loaded before
        # marking scan finished

    def get_data_chunk(self, *args, **kwargs):
        self.kf.refresh()
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        # refreshing here to update before Ptyscan.get_data_chunk calls check
        # and load
        return super().get_data_chunk(*args, **kwargs)

    def check(self, frames=None, start=None):
        """
        Check the live SWMR file for available frames.
        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        self.available = min(self.kf.get_current_max() + 1, self.num_frames)
        new_frames = self.available - start
        if new_frames <= frames:
            # not reached expected nr. of frames,
            # but its last chunk of scan so load it anyway
            if self.available == self.num_frames:
                frames_accessible = new_frames
                end_of_scan = 1
            # reached expected nr. of frames
            # but first block must be of maximum size
            elif start != 0 and new_frames >= self.min_frames:
                frames_accessible = self.min_frames
                end_of_scan = 0
            # not reached required nr. of frames, do nothing
            else:
                end_of_scan = 0
                frames_accessible = 0
        # reached expected nr. of frames
        else:
            frames_accessible = frames
            end_of_scan = 0

        return frames_accessible, end_of_scan