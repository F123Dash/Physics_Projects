"""Settings shared by the solver, data pipeline, training and evaluation.

Change values here, not in the individual modules.
"""
import os

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_ROOT = os.path.join(PROJECT_ROOT, "snapshots")       # training data
OOD_ROOT      = os.path.join(PROJECT_ROOT, "snapshots_ood")   # unseen-Re test data
IMAGE_ROOT    = os.path.join(PROJECT_ROOT, "img_out")
RUNS_ROOT     = os.path.join(PROJECT_ROOT, "runs")

# Reynolds numbers
RE_VALUES     = [100, 200, 400, 600, 800, 1000, 1200, 1500, 1700, 2000, 2500, 3200]
OOD_RE_VALUES = [1800, 2200, 2800, 4000]
VAL_RE        = [600, 2000]     # held out of training, used for model selection

# Fields and super-resolution
CHANNEL_NAMES = ["u-velocity", "v-velocity", "pressure", "vorticity ω"]   # indexed [x, y]
FINE_SIZE     = 64              # cavity: nodes per side, walls included (dx = 1/(FINE_SIZE-1))
COARSE_FACTOR = 4               # input = block averages over COARSE_FACTOR × COARSE_FACTOR cells
NOISE_STD     = 0.05            # input noise on training samples, normalised units

# 2-D periodic turbulence (solver/turbulence.py)
TURB_ROOT          = os.path.join(PROJECT_ROOT, "snapshots_turb")       # <kind>/Re_*/seed_*/
TURB_OOD_ROOT      = os.path.join(PROJECT_ROOT, "snapshots_turb_ood")
TURB_FINE_SIZE     = 128        # saved grid (spectrally truncated DNS); model input is 32² block averages
TURB_RE_VALUES     = [250, 400, 630, 1000, 1600, 2500]    # log-spaced, ×1.58
TURB_VAL_RE        = [630]                                 # held out of training, used for model selection
TURB_OOD_RE_VALUES = [320, 800, 2000, 4000]                # 320/800/2000 interpolate, 4000 extrapolates
TURB_SEEDS         = [0, 1, 2]
TURB_OOD_SEEDS     = [10, 11]
# DNS grid per flow kind, chosen so the enstrophy spectrum at the dealiasing cut-off is
# < 1e-3 of its peak (checked with solver/turbulence.py's resolution diagnostic).
TURB_DNS_GRID      = {"forced":   [(2500, 256), (10**9, 512)],
                      "decaying": [(1000, 256), (10**9, 512)]}

# Datasets per flow: train.py/evaluate.py --flow picks one (explicit --data_dir etc. override)
# noise_std: input noise on training samples. Turbulence uses none: noise of 0.05 (normalised)
#            carried 1.1-1.4x the energy of the true fine scales (k >= 16) it had to reconstruct.
# recon:     "mse" on normalised fields, or "relative" (per sample and channel, physical units) so that
#            weak fields (late decaying turbulence) are not swamped by strong ones.
FLOWS = {
    "cavity":     dict(train_dirs=[SNAPSHOT_ROOT], test_dirs=[OOD_ROOT],      val_re=VAL_RE,
                       noise_std=NOISE_STD, recon="mse"),
    "turbulence": dict(train_dirs=[TURB_ROOT],     test_dirs=[TURB_OOD_ROOT], val_re=TURB_VAL_RE,
                       noise_std=0.0,       recon="relative"),
}
DEFAULT_FLOW = "turbulence"
