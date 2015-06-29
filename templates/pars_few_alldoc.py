"""
This Script was autogenerated using
``u.create_default_template("pars_few_alldoc.py",0,2)``
It is only a TEMPLATE and not a working reconstruction script.
"""

import numpy as np
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

#### Ptypy Parameter Tree ######################################################

## (00) Verbosity level
# Verbosity level for information logging.
#  - ``0``: Only errors
#  - ``1``: Warning
#  - ``2``: Process Information
#  - ``3``: Object Information
#  - ``4``: Debug
p.verbose_level = 1

## (02) Reconstruction identifier
# Reconstruction run identifier. If ``None``, the run name will be constructed at run
# time from other information.
p.run = None

## (04) Global parameters for I/O
p.io = u.Param()

## (07) Auto-save options
p.io.autosave = u.Param()

## (11) Server / Client parameters
# If ``None`` or ``False`` is passed here in script instead of a Param, it translates to
#  ``active=False`` i.e. no ZeroMQ interaction server.
p.io.interaction = u.Param()

## (12) Activation switch
# Set to ``False`` for no  ZeroMQ interaction server
p.io.interaction.active = True

## (16) Plotting client parameters
# In script you may set this parameter to ``None`` or ``False`` for no automatic plotting.
# 
p.io.autoplot = u.Param()

## (23) Scan parameters
# This categrogy specifies defaults for all scans. Scan-specific parameters are stored
# in scans.scan_%%
p.scan = u.Param()

## (26) Data preparation parameters
p.scan.data = u.Param()

## (27) Data preparation recipe container
p.scan.data.recipe = None

## (28) Describes where to get the data from.
# Accepted values are:
#  - ``'file'``: data will be read from a .ptyd file.
#  - any valid recipe name: data will be prepared using the recipe.
#  - ``'sim'`` : data will be simulated according to parameters in simulation
p.scan.data.source = None

## (29) Prepared data file path
# If source was ``None`` or ``'file'``, data will be loaded from this file and processing
# as well as saving is deactivated. If source is the name of an experiment recipe or path
# to a file, data will be saved to this file
p.scan.data.dfile = None

## (34) Detector pixel size
# Dimensions of the detector pixels (in meters)
p.scan.data.psize = None

## (35) Sample-to-detector distance
# In meters.
p.scan.data.distance = None

## (38) Photon energy of the incident radiation
p.scan.data.energy = None

## (40) Maximum number of frames to be prepared
# If `positions_theory` are provided, num_frames will be ovverriden with the number
# of positions available
p.scan.data.num_frames = None

## (42) Determine if center in data is calculated automatically
# - ``False``, no automatic centering 
#  - ``None``, only if :py:data:`center` is ``None`` 
#  - ``True``, it will be enforced
p.scan.data.auto_center = None

## (43) Determines what will be loaded in parallel
# Choose from ``None``, ``'data'``, ``'common'``, ``'all'``
p.scan.data.load_parallel = "data"

## (49) Scan sharing options
p.scan.sharing = u.Param()

## (54) Physical parameters
# All distances are in meters. Other units are specified in the documentation strings.
# 
p.scan.geometry = u.Param()

## (55) Energy (in keV)
# If ``None``, uses `lam` instead.
p.scan.geometry.energy = 6.2

## (56) Wavelength
# Used only if `energy` is ``None``
p.scan.geometry.lam = None

## (57) Distance from object to detector
p.scan.geometry.distance = 7.19

## (61) Parameters for scan patterns
# These parameters are useful in two cases:
#  - When the experimental positions are not known (no encoders)
#  - When using the package to simulate data.
# In script an array of shape *(N,2)* may be passed here instead of a Param or dictionary
# as an **override**
p.scan.xy = u.Param()

## (62) Scan pattern type
# The type must be one of the following:
#  - ``None``: positions are read from data file.
#  - ``'raster'``: raster grid pattern
#  - ``'round'``: concentric circles pattern
#  - ``'spiral'``: spiral pattern
# In script an array of shape *(N,2)* may be passed here instead
p.scan.xy.model = None

## (63) Pattern spacing
# Spacing between scan positions. If the model supports asymmetric scans, a tuple passed
# here will be interpreted as *(dy,dx)* with *dx* as horizontal spacing and *dy* as vertical
# spacing. If ``None`` the value is calculated from `extent` and `steps`
p.scan.xy.spacing = 1.5e-06

## (64) Pattern step count
# Number of steps with length *spacing* in the grid. A tuple *(ny,nx)* provided here can
# be used for a different step in vertical ( *ny* ) and horizontal direction ( *nx* ). If ``None``
# the, step count is calculated from `extent` and `spacing`
p.scan.xy.steps = 10

## (65) Rectangular extent of pattern
# Defines the absolut maximum extent. If a tuple *(ly,lx)* is provided the extent may be
# rectangular rather than square. All positions outside of `extent` will be discarded.
# If ``None`` the extent will is `spacing` times `steps`
p.scan.xy.extent = 1.5e-05

## (69) Illumination model (probe)
# In script, you may pass directly a three dimensional  numpy.ndarray here instead of a
# `Param`. This array will be copied to the storage instance with no checking whatsoever.
# Used in `~ptypy.core.illumination`
p.scan.illumination = u.Param()

## (70) Type of illumination model
# One of:
#  - ``None`` : model initialitziation defaults to flat array filled with the specified
# number of photons
#  - ``'recon'`` : load model from previous reconstruction, see `recon` Parameters
#  - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
#  - *<resource>* : one of ptypys internal image resource strings
#  - *<template>* : one of the templates inillumination module
# In script, you may pass a numpy.ndarray here directly as the model. It is considered as
# incoming wavefront and will be propagated according to `propagation` with an optional
# `aperture` applied before
p.scan.illumination.model = None

## (72) Parameters to load from previous reconstruction
p.scan.illumination.recon = u.Param()

## (73) Path to a ``.ptyr`` compatible file
p.scan.illumination.recon.rfile = "\*.ptyr"

## (74) ID (label) of storage data to load
# ``None`` means any ID
p.scan.illumination.recon.ID = None

## (75) Layer (mode) of storage data to load
# ``None`` means all layers, choose ``0`` for main mode
p.scan.illumination.recon.layer = None

## (78) Beam aperture parameters
p.scan.illumination.aperture = u.Param()

## (79) One of None, 'rect' or 'circ'
# One of:
#  - ``None`` : no aperture, this may be useful for nearfield
#  - ``'rect'`` : rectangular aperture
#  - ``'circ'`` : circular aperture
p.scan.illumination.aperture.form = "circ"

## (81) Aperture width or diameter
# May also be a tuple *(vertical,horizontal)* in case of an asymmetric aperture
p.scan.illumination.aperture.size = None

## (85) Parameters for propagation after aperture plane
# Propagation to focus takes precedence to parallel propagation if `foccused` is not
# ``None``
p.scan.illumination.propagation = u.Param()

## (86) Parallel propagation distance
# If ``None`` or ``0`` : No parallel propagation
p.scan.illumination.propagation.parallel = None

## (87) Propagation distance from aperture to focus
# If ``None`` or ``0`` : No focus propagation
p.scan.illumination.propagation.focussed = None

## (90) Probe mode(s) diversity parameters
# Can be ``None`` i.e. no diversity
p.scan.illumination.diversity = u.Param()

## (94) Initial object modelization parameters
# In script, you may pass a numpy.array here directly as the model. This array will be passed
# to the storage instance with no checking whatsoever. Used in `~ptypy.core.sample`
# 
p.scan.sample = u.Param()

## (95) Type of initial object model
# One of:
#  - ``None`` : model initialitziation defaults to flat array filled `fill`
#  - ``'recon'`` : load model from STXM analysis of diffraction data
#  - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
#  - *<resource>* : one of ptypys internal model resource strings
#  - *<template>* : one of the templates in sample module
# In script, you may pass a numpy.array here directly as the model. This array will be processed
# according to `process` in order to *simulate* a sample from e.g. a thickness profile.
# 
p.scan.sample.model = None

## (96) Default fill value
p.scan.sample.fill = 1

## (97) Parameters to load from previous reconstruction
p.scan.sample.recon = u.Param()

## (98) Path to a ``.ptyr`` compatible file
p.scan.sample.recon.rfile = "\*.ptyr"

## (103) Model processing parameters
# Can be ``None``, i.e. no processing
p.scan.sample.process = u.Param()

## (111) Probe mode(s) diversity parameters
# Can be ``None`` i.e. no diversity
p.scan.sample.diversity = u.Param()

## (115) Coherence parameters
p.scan.coherence = u.Param()

## (116) Number of probe modes
p.scan.coherence.num_probe_modes = 1

## (117) Number of object modes
p.scan.coherence.num_object_modes = 1

## (121) Param container for instances of `scan` parameters
# If not specified otherwise, entries in *scans* will use parameter defaults from :py:data:`.scan`
# 
p.scans = u.Param()

## (122) Default first scans entry
# If only a single scan is used in the reconstruction, this entry may be left unchanged.
# If more than one scan is used, please make an entry for each scan. The name *scan_00* is
# an arbitrary choice and may be set to any other string.
p.scans.scan_00 = None

## (123) Reconstruction engine parameters
p.engine = u.Param()

## (124) Parameters common to all engines
p.engine.common = u.Param()

## (125) Name of engine.
# Dependent on the name given here, the default parameter set will be a superset of `common`
# and parameters to the entry of the same name.
p.engine.common.name = "DM"

## (126) Total number of iterations
p.engine.common.numiter = 2000

## (128) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_support = 0.7

## (130) Parameters for Difference map engine
p.engine.DM = u.Param()

## (140) Maximum Likelihood parameters
p.engine.ML = u.Param()

## (143) A rescaling of the intensity so they can be interpreted as Poisson counts.
p.engine.ML.intensity_renormalization = 1

## (144) Whether to use a Gaussian prior (smoothing) regularizer.
p.engine.ML.reg_del2 = True

## (145) Amplitude of the Gaussian prior if used.
p.engine.ML.reg_del2_amplitude = 0.01

## (150) Container for instances of "engine" parameters
# All engines registered in this structure will be executed sequentially.
p.engines = u.Param()

## (151) Default first engines entry
# Default first engine is difference map (DM)
p.engines.engine_00 = None


Ptycho(p,level=5)
