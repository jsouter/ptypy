from swmr_tools import KeyFollower
from swmr_tools.utils import refresh_dataset
import numpy as np

class KeyFollowerV2(KeyFollower):
    """
    Subclass of KeyFollower that does not hang on skipped keys. Skipped frames
    are stored in the .skipped member variable after .refresh() is called.
    The number of prepared frames is given by .get_number_frames().
    Framefilters derived from the KeyFollower are returned by
    .get_framefilter().
    """
    def __init__(self, *args, **kwargs):
        self.missing_frames = set()
        # self.nonzeros_after_zeros = set()
        self.skipped = set()
        self.block_start = 0
        self.final_frame_checked = False
        self.last_frame_timeout = None
        super().__init__(*args, **kwargs)

    @property
    def max_size(self):
        sizes = []
        for k in self.key_datasets:
            refresh_dataset(k)
            sizes.append(np.prod(k.shape))
        return max(sizes)

    def get_number_frames(self):
        if len(self.skipped):
            print(f"skipping {len(self.skipped)} frames between {min(self.skipped)} and {max(self.skipped)}")
        return self.get_current_max() + 1 - len(self.skipped)

    def get_max_possible(self):
        # this should be the highest number of frames we can expect
        # once we remove all the known skipped frames
        return self.max_size - len(self.skipped)

    @property
    def all_frames_made(self):
        return (self.max_size == self.get_current_max() + 1)

    def get_framefilter(self, shape=None):
        flat_array = np.array([i not in self.skipped for i in range(self.max_size)])
        if shape:
            return flat_array.reshape(shape)
        else:
            return flat_array

    def _is_next(self):
        karray = self._get_keys()
        if not karray:
            return False

        if len(karray) == 1:
            merged = karray[0]
            max_size = merged.size
        else:
            max_size = self.max_size
            merged = np.zeros(max_size)
            first = karray[0]
            merged[: first.size] = merged[: first.size] + first

            for k in karray[1:]:
                padded = np.zeros(max_size)
                padded[: k.size] = k
                merged = merged * padded
        remaining = merged[self.current_max + 1:]
        if len(remaining) < 1:
            # end of scan
            return False
        try:
            new_frames = remaining[:np.argwhere(remaining != 0).flatten()[-1] + 1]
            self._timer_reset()
        except IndexError:
            # if no new nonzero keys after current max.
            if self._timeout():
            # if timeout, mark zeroed frames as finished
                new_frames = remaining
            else:
                new_frames = np.array([])
        new_skipped = np.argwhere(new_frames == 0).flatten() + self.current_max + 1
        self.skipped.update(new_skipped)
        new_max = self.current_max + new_frames.size

        if new_max < 0 and merged[0] != 0:
            # all keys non zero
            new_max = merged.size - 1
        if self.current_max == new_max:
            return False

        self.current_max = new_max
        return True
