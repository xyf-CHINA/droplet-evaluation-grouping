"""Frozen model / task specification shared by all stages.

Single source of truth: stage1a/1b/2 and tests all import from here.
"""
# Author 8-feature specification (kept UNCHANGED across all protocols).
FEATURE_COLS = [
    "Orifice width (um)",
    "Normalized channel depth",
    "Flow rate ratio",
    "Capillary number",
    "Normalized continuous inlet",
    "Normalized dispersed inlet",
    "Normalized outlet width",
    "viscosity ratio",
]
TARGET_COL = "Normalized droplet diameter"
DENORM_COL = "Hydraulic diameter"
OBS_COL = "Observed droplet diameter (um)"

# Explicitly locked XGBoost specification.
# Never tuned; identical in every protocol and every model experiment.
LOCKED_XGB_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.3,
    max_depth=6,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=0,
)

# Locked Stage 3 model specs (frozen BEFORE running, never tuned).
LOCKED_RF_PARAMS = dict(n_estimators=100, random_state=0)  # sklearn defaults otherwise
LOCKED_MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=0,
)

TEST_SIZE = 0.20
SEEDS_R = list(range(100))  # Protocol R seeds
