# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the Diamond beamlines.

"""
import h5py as h5
import numpy as np

from ptypy.experiment import register
from ptypy.experiment.hdf5_loader import Hdf5Loader
from ptypy.utils.verbose import log
from ptypy.utils import parallel
from ptypy.utils.verbose import logger
from ptypy.core.data import EOS, WAIT

try:
    from ptypy.utils.swmr_utils import KeyFollowerV2
except ImportError:
    log(3, "The SWMR loader requires swmr_tools to be installed,"
           " try pip install swmr_tools")
    raise ImportError

@register()
class SwmrLoader(Hdf5Loader):
    """
    An attempt to specialise the Hdf5Loader for the case where the loading
    is being done during data creation.

    Defaults:

    [name]
    default = 'SwmrLoader'
    type = str
    help =

    [max_frames]
    default = None
    type = int
    help = Maximum number of frames to load before marking end of scan.
    doc = At initialisation, used to limit SwmrLoader.num_frames. When this
          many frames are loaded in total, end of scan is marked.
          Mainly useful for debugging purposes.

    [intensities.live_key]
    default = None
    type = str
    help = Key to live keys inside the intensities file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start,
          but non-zero when the position is complete.

    [positions.live_fast_key]
    default = None
    type = str
    help = Key to live keys for the fast position axis in the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start,
          but non-zero when the position is complete.

    [positions.live_slow_key]
    default = None
    type = str
    help = Key to live keys for the slow position axis in the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start,
          but non-zero when the position is complete.

    [keyfollower_timeout]
    default = 10
    type = int, float
    help = Timeout in seconds after which KeyFollower marks scan finished
           if no new frames readied.

    [checkpoints]
    default = None
    type = list
    help = A list of numbers of frames at which loading should be paused to
           peform a block of engine iterations
    doc = A list of numbers of frames at which loading should be paused to
           peform a block of engine iterations. The first value should be
           higher than the frames_per_block parameter. 
           Note that checkpoints take priority over min_frames.


    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)

    def setup(self, *args, **kwargs):
        self.checkpoints = self.p.checkpoints or []
        self.checkpoint_reached = None
        log(3, f"Checkpoints: {self.checkpoints}")
        self.available = None
        self._is_swmr = True
        self._use_keyfilter = True if None in [self.p.framefilter.key,
                                               self.p.framefilter.file] else False
        # if no framefilter passed, use key follower to filter frames by index
        super().setup(*args, **kwargs)
        # Check if we have been given the live keys
        if None in [self.p.intensities.live_key,
                    self.p.positions.live_slow_key,
                    self.p.positions.live_fast_key]:
            raise RuntimeError("Missing the live keys to intensities or positions!")

        # Check that intensities and positions (and their live keys) are loaded from the same file
        if self.p.intensities.file != self.p.positions.file:
            raise RuntimeError("Intensities and positions file should be the same")

        intensity_file = h5.File(self.p.intensities.file, 'r',
                                 swmr=self._is_swmr)
        positions_file = h5.File(self.p.positions.file, 'r',
                                 swmr=self._is_swmr)

        # Initialize KeyFollower
        self.kf = KeyFollowerV2((intensity_file[self.p.intensities.live_key],
                                 positions_file[self.p.positions.live_slow_key],
                                 positions_file[self.p.positions.live_fast_key]),
                                timeout=self.p.keyfollower_timeout)

        # Get initial value of maximum number of frames to be loaded before
        # marking scan finished

        self.num_frames = min(f for f in [self.p.max_frames,
                                          self.num_frames,
                                          self.kf.get_max_possible()] if f is not None)

    def get_data_chunk(self, *args, **kwargs):
        '''
        Calls PtyScan's get_data_chunk after calling the refresh method on the
        intensity and position datasets so that newly created data is seen by
        the SwmrLoader. 
        The KeyFollower is also refreshed to update the number of available
        frames as seen by the check function.
        num_frames is lowered if any additional skipped frames are discovered.
        '''
        self.kf.refresh()
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        # refreshing here to update before Ptyscan.get_data_chunk calls check
        # and load
        self.num_frames = min(self.kf.get_max_possible(), self.num_frames)
        return super().get_data_chunk(*args, **kwargs)

    def load_unmapped_raster_scan(self, *args, **kwargs):
        raise NotImplementedError("framefilter not supported for"
                                  " unmapped raster scans (see hdf5 loader)")

    def load_mapped_and_raster_scan(self, *args, **kwargs):
        if self._use_keyfilter:
            filter_shape = self.intensities.shape[:-2]
            filter = self.kf.get_framefilter(filter_shape)
            skip = self.p.positions.skip
            self.preview_indices = self.unfiltered_indices[:, filter[::skip,::skip].flatten()]
            # print(time.time() - start, "time taken to get preview indices")
        # print(self.preview_indices)
        return super().load_mapped_and_raster_scan(*args, **kwargs)

    def load_mapped_and_arbitrary_scan(self, *args, **kwargs):
        if self._use_keyfilter:
            filter = self.kf.get_framefilter()
            self.preview_indices = self.unfiltered_indices[filter[::self.p.positions.skip]]
        return super().load_mapped_and_arbitrary_scan(*args, **kwargs)

    def check(self, frames=None, start=None):
        """
        Check the live SWMR file for available frames.
        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        self.available = min(self.kf.get_number_frames(), self.num_frames)
        frames_accessible = min(frames, self.available - start)

        return frames_accessible, None

    def _mpi_check(self, chunksize, start=None):
        """
        Executes the check() function on master node and communicates
        the result with the other nodes.
        This function determines if the end of the scan is reached
        or if there is more data after a pause.

        returns:
            - codes WAIT or EOS
            - or (start, frames) if data can be loaded
        """
        # Take internal counter if not specified
        s = self.framestart if start is None else int(start)

        # Check may contain data system access, so we keep it to one process
        if parallel.master:
            # If checkpoint reached in last check, force to return no frames
            # to yield control to main loop in Ptycho.run() which performs
            # engines iterations
            if self.checkpoint_reached is True:
                self.checkpoint_reached = False
                self.frames_accessible, eos = 0, self.end_of_scan
            else:
                frames_accessible, eos = self.check(chunksize, start=s)
                for c in self.checkpoints:
                    if c > s and c > chunksize and frames_accessible + s >= c:
                        # frames_accessible = c - s
                        self.checkpoint_reached = True
                        break
                self.frames_accessible = frames_accessible
                if self.num_frames is None and eos is None:
                    logger.warning('Number of frames not specified and'
                                   ' .check() cannot determine end-of-scan.'
                                   ' Aborting..')
                    self.abort = True

                if eos is None:
                    self.end_of_scan = (s + self.frames_accessible
                                        >= self.num_frames)
                else:
                    self.end_of_scan = eos

        # Wait for master
        parallel.barrier()
        # Communicate result
        self._flags = parallel.bcast(self._flags)

        # Abort here if the flag was set
        if self.abort:
            raise RuntimeError(
                'Load routine incapable to determine the end-of-scan.')

        frames_accessible = self.frames_accessible
        # first block must be of maximum size as it defines maximum memory
        # allocation when using GPU accelerated engine
        if self.end_of_scan is not True and \
            ((frames_accessible < self.min_frames and not self.checkpoint_reached) or
             (s == 0 and frames_accessible < chunksize)):
            return WAIT
        elif self.end_of_scan and frames_accessible <= 0:
            return EOS
        else:
            # Move forward, set new starting point
            self.framestart += frames_accessible
            return s, frames_accessible
