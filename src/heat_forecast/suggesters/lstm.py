
## Suggesters for LSTM

import optuna
import copy

from heat_forecast.utils.optuna import (
    register_suggester, 
    SuggesterDoc,
    ParamCat, ParamInt, ParamFloat
)
from heat_forecast.pipeline.lstm import LSTMRunConfig

#--------------------------------
# PRELIMINARY STUDY SUGGESTERS
# --------------------------------

_lstm_v1_doc = SuggesterDoc(
    summary="Baseline LSTM search space for 24-hour ahead forecasting.",
    params=[
        # Model
        ParamCat("model.head", ["linear", "mlp"]),
        ParamCat("model.hidden_size", [32, 64, 128]),
        ParamCat("model.num_layers", [1, 2]),
        ParamCat("model.dropout", [0.0, 0.15, 0.3]),

        # Windows
        ParamCat("model.input_len", [72, 168]),

        # Data
        ParamCat("data.batch_size", [64, 128]),

        # Training
        ParamFloat("train.learning_rate", 1e-4, 5e-3, log=True),
    ],
    notes=[
        "`norm.mode` is fixed to `global`.",
        "`train.n_epochs` is fixed to 35; early stopping trims excess epochs.",
        "The heavy combo of `model.hidden_size=128`, `model.num_layers=2`, and `model.input_len=168` is skipped (pruned early).",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number (and base_cfg.seed if provided).",
        "Other params use defaults.",
    ],
)

@register_suggester("lstm_v1", doc=_lstm_v1_doc)
def suggest_config_v1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model ----
    cfg.model.input_len  = trial.suggest_categorical("model.input_len", [72, 168])
    cfg.model.num_layers = trial.suggest_categorical("model.num_layers", [1, 2])
    cfg.model.hidden_size = trial.suggest_categorical("model.hidden_size", [32, 64, 128])

    # Forbid a heavy combo, prune early
    if cfg.model.num_layers == 2 and cfg.model.input_len == 168 and cfg.model.hidden_size == 128:
        trial.set_user_attr("invalid_combo", True)
        raise optuna.TrialPruned("Skip hidden_size=128 for 2 layers & 168 input_len")

    cfg.model.head    = trial.suggest_categorical("model.head", ["linear", "mlp"])
    cfg.model.dropout = trial.suggest_categorical("model.dropout", [0.0, 0.15, 0.3])

    # ---- Training/Data ----
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 5e-4, 5e-3, log=True)
    cfg.train.n_epochs = 35
    cfg.data.batch_size = trial.suggest_categorical("data.batch_size", [64, 128])
    cfg.norm.mode = "global"

    return cfg

preliminary_v2_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v2 (t vs thpw comparison).",
    params=[
        ParamCat("features.exog_vars_key", ["t", "thpw"]),
        ParamCat("repeat_idx", [0, 1, 2, 3]),
    ],
    notes=[
        "`repeat_idx` repeats each config 4×.",
        "Other fixed params: input_len=72, num_layers=2, hidden_size=64, head='linear', "
        "dropout=0.1, lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on).",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)
@register_suggester("preliminary_v2", doc=preliminary_v2_doc)
def suggester_preliminary_v2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v2') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3])  # dummy to repeat trials

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len  = 72
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = 64

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    # ---- Features ----
    exog_vars_key = trial.suggest_categorical("features.exog_vars_key", ["t", "thpw"])
    cfg.features.exog_vars = ("temperature",) if exog_vars_key == "t" else ("temperature", "humidity", "pressure", "wind_speed")
    trial.set_user_attr("repeat_idx", idx)

    return cfg

preliminary_v2_second_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v2 for F4 (t vs thpw comparison; es activated at epoch 12).",
    params=[
        ParamCat("features.exog_vars_key", ["t", "thpw"]),
        ParamCat("repeat_idx", [0, 1, 2, 3]),
    ],
    notes=[
        "`repeat_idx` repeats each config 4×.",
        "Other fixed params: input_len=72, num_layers=2, hidden_size=64, head='linear', "
        "dropout=0.1, lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25, es_start_epoch=12s.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)
@register_suggester("preliminary_v2_second", doc=preliminary_v2_doc)
def suggester_preliminary_v2_second(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v2_second') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3])  # dummy to repeat trials

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len  = 72
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = 64

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.train.es_start_epoch = 12
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    # ---- Features ----
    exog_vars_key = trial.suggest_categorical("features.exog_vars_key", ["t", "thpw"])
    cfg.features.exog_vars = ("temperature",) if exog_vars_key == "t" else ("temperature", "humidity", "pressure", "wind_speed")
    trial.set_user_attr("repeat_idx", idx)

    return cfg

preliminary_v3_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v3 (long context vs short context + lags).",
    params=[
        ParamCat("model.input_len", [24, 72, 168, 504]),
        ParamCat("features.endog_hour_lags", [(), (168,), (168, 24)]),
        ParamCat("model.hidden_size", [64, 128]),
        ParamCat("repeat_idx", [0, 1, 2]),
    ],
    notes=[
        "`repeat_idx` repeats each config 3×.",
        "Other fixed params: input_len=72, num_layers=2, head='linear', dropout=0.1, "
        "lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on).",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)
@register_suggester("preliminary_v3", doc=preliminary_v3_doc)
def suggester_preliminary_v3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v3') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len = trial.suggest_categorical("model.input_len", [24, 72, 168, 504])
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = trial.suggest_categorical("model.hidden_size", [64, 128])

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Features ----
    cfg.features.endog_hour_lags = trial.suggest_categorical("features.endog_hour_lags", [(), (168,), (168, 24)])
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    return cfg

preliminary_v4_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v4 (autoregressive vs non-autoregressive dec).",
    params=[
        ParamCat("model.use_ar_prev", [False, True]),
        ParamCat("repeat_idx", [0, 1, 2, 3]),
    ],
    notes=[
        "`repeat_idx` repeats each config 4×.",
        "Other fixed params: input_len=72, num_layers=2, head='linear', dropout=0.1, hidden_size=64, "
        "lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on), es_start_epoch=12.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)

@register_suggester("preliminary_v4", doc=preliminary_v4_doc)
def suggester_preliminary_v4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v4') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = 64
    cfg.model.use_ar_prev = trial.suggest_categorical("model.use_ar_prev", [False, True])

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.train.es_start_epoch = 12
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    return cfg

preliminary_v5_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v5 (appropriate complexity study).",
    params=[
        ParamCat("model.hidden_size", [2, 4, 8, 12, 16, 32, 48, 64, 96, 128, 160, 196]),
        ParamCat("repeat_idx", [0, 1]),
    ],
    notes=[
        "`repeat_idx` repeats each config 2x.",
        "Other fixed params: input_len=72, num_layers=1, head='linear', dropout=0.1, hidden_size=64, "
        "lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on), es_start_epoch=6.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)

@register_suggester("preliminary_v5", doc=preliminary_v5_doc)
def suggester_preliminary_v5(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v5') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers = 1
    cfg.model.head       = "linear"
    cfg.model.hidden_size = trial.suggest_categorical("model.hidden_size", [2, 4, 8, 12, 16, 32, 48, 64, 96, 128, 160, 196])
    cfg.model.use_ar = "none"

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.train.es_start_epoch = 7
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    return cfg

final_v1_F1_doc = SuggesterDoc(
    summary="Final LSTM suggester v1 (F1)",
    params=[
        ParamInt("model.num_layers", 1, 4),
        ParamInt("model.hidden_size", 32, 128, step=32),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
        ParamFloat("train.weight_decay_pos", 1e-6, 1e-2, log=True, condition="train.use_weight_decay == True"),
        ParamInt("train.drop_epoch", 5, 8),
        ParamInt("data.batch_size", 32, 96, step=32),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', use_ar=24h, "
        "use_lr_drop=True (factor=0.3), norm.mode='global', "
        "n_epochs=25 with es (es_start_epoch=6), max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.lr_drop_epoch and train.tf_drop_epochs are both set to `train.drop_epoch`.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v1_F1", doc=final_v1_F1_doc)
def suggester_final_v1_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 4)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 32, 128, step=32)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay_pos", 1e-6, 1e-2, log=True)
    )
    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 5, 8)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.data.batch_size      = trial.suggest_int("data.batch_size", 32, 96, step=32)
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v1_F2_doc = SuggesterDoc(
    summary="Final LSTM suggester v1 (F2)",
    params=[
        ParamInt("model.num_layers", 1, 3),
        ParamInt("model.hidden_size_exp", 4, 7),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
        ParamFloat("train.weight_decay_pos", 1e-6, 1e-2, log=True, condition="train.use_weight_decay == True"),
        ParamInt("train.drop_epoch", 5, 8),
        ParamInt("data.batch_size", 32, 96, step=32),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', use_ar=24h, "
        "use_lr_drop=True (factor=0.3), norm.mode='global', "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.5%), max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.lr_drop_epoch and train.tf_drop_epochs are both set to `train.drop_epoch`.",
        "model.hidden_size` is set to 2^`model.hidden_size_exp`.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v1_F2", doc=final_v1_F2_doc)
def suggester_final_v1_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 3)
    cfg.model.head        = "linear"
    hidden_size_exp = trial.suggest_int("model.hidden_size_exp", 4, 7)
    cfg.model.hidden_size = 2 ** hidden_size_exp
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay_pos", 1e-6, 1e-2, log=True)
    )
    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 5, 8)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta = 0.005
    cfg.data.batch_size      = trial.suggest_int("data.batch_size", 32, 96, step=32)
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F1_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F1)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 32, 128, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_F1", doc=final_v2_F1_doc)
def suggester_final_v2_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 32, 128, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_NAR_F1_doc = SuggesterDoc(
    summary="Final LSTM suggester v2, non-ar (F1)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 32, 128, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=none, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_NAR_F1", doc=final_v2_NAR_F1_doc)
def suggester_final_v2_NAR_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 32, 128, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "none"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F2_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F2)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 16, 112, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_F2", doc=final_v2_F2_doc)
def suggester_final_v2_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 16, 112, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_NAR_F2_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (NAR F2)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 16, 112, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=none, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_NAR_F2", doc=final_v2_NAR_F2_doc)
def suggester_final_v2_NAR_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 16, 112, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "none"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F3_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F3)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 8, 32, step=8),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
        ParamFloat("train.weight_decay", 1e-6, 1e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_F3", doc=final_v2_F3_doc)
def suggester_final_v2_F3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 32, step=8)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay", 1e-6, 1e-2, log=True)
    )
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch 

    return cfg

final_v2_NAR_F3_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 non-autoregressive (F3)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 8, 32, step=8),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
        ParamFloat("train.weight_decay", 1e-6, 1e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=none, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_NAR_F3", doc=final_v2_NAR_F3_doc)
def suggester_final_v2_NAR_F3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 32, step=8)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay", 1e-6, 1e-2, log=True)
    )
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "none"
    cfg.train.tf_drop_epochs = drop_epoch 

    return cfg

final_v2_F4_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F4)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 8, 24, step=4),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_F4", doc=final_v2_F4_doc)
def suggester_final_v2_F4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 24, step=4)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch               = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F5_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F5)",
    params=[
        ParamCat("model.input_len", [72, 120]),
        ParamInt("model.hidden_size", 8, 32, step=4),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', tf_drop_epochs=4, batch_size=64, "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_epoch=4), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_F5", doc=final_v2_F5_doc)
def suggester_final_v2_F5(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = trial.suggest_categorical("model.input_len", [72, 120])
    cfg.model.num_layers  = 1
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 32, step=4)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch               = 4
    cfg.train.lr_drop_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F5_alt_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F5)",
    params=[
        ParamInt("model.hidden_size", 8, 32, step=4),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', tf_drop_epochs=4, batch_size=64, "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=4), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_alt_F5", doc=final_v2_F5_alt_doc)
def suggester_final_v2_F5_alt(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = 1
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 32, step=4)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v3_F1_doc = SuggesterDoc(
    summary="LSTM final sugg v3 (final candidates with repeats).",
    params=[
        ParamCat("config", ["AR", "NAR"]),
        ParamCat("repeat_idx", list(range(10))),
    ],
    notes=[
        "`repeat_idx` repeats each config 10×.",
        "AR: selected autoregressive configuration; NAR: selected non-autoregressive configuration.",
    ],
)

@register_suggester("final_v3_F1", doc=final_v3_F1_doc)
def suggester_final_v3_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F1') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(10)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    config = trial.suggest_categorical("config", ["AR", "NAR"])
    trial.set_user_attr("config", config)
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 10
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 64
    cfg.model.num_layers = 2
    cfg.train.lr_drop_epoch = 4
    cfg.train.tf_drop_epochs = 4
    cfg.train.learning_rate = 8e-04

    if config == "NAR":
        cfg.model.dropout = 0.1
        cfg.model.use_ar  = "none"
    else:
        cfg.model.dropout = 0.0
        cfg.model.use_ar  = "24h"

    return cfg

final_v4_F1_doc = SuggesterDoc(
    summary="LSTM final sugg v4 (final candidates with repeats).",
    params=[
        ParamFloat("train.learning_rate", 3e-4, 1.5e-3, log=True),
        ParamFloat("model.dropout", 0.0, 0.2),
    ],
    notes=[
        "Other params used the same fixed values as in `final_v3_F1`:",
        " batch_size=64, input_len=72, head='linear', include_exog_lags=True, lr_drop_epoch=4, lr_drop_factor=0.3, "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.0), norm.mode='global', hidden_size=64, num_layers=2, use_ar='none'.",
    ],
)

final_v3_F2_doc = SuggesterDoc(
    summary="LSTM final sugg v3 for F2 (final candidates with repeats).",
    params=[
        ParamCat("config", ["AR", "NAR"]),
        ParamCat("repeat_idx", list(range(10))),
    ],
    notes=[
        "`repeat_idx` repeats each config 10×.",
        "AR: selected autoregressive configuration; NAR: selected non-autoregressive configuration.",
    ],
)

@register_suggester("final_v3_F2", doc=final_v3_F2_doc)
def suggester_final_v3_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F2') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(10)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    config = trial.suggest_categorical("config", ["AR", "NAR"])
    trial.set_user_attr("config", config)
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 10
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 112
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.05
    cfg.train.lr_drop_epoch = 4
    cfg.train.tf_drop_epochs = 4
    cfg.train.learning_rate = 8e-04

    if config == "NAR":
        cfg.model.use_ar  = "none"
    else:
        cfg.model.use_ar  = "24h"

    return cfg

final_v3_F3_doc = SuggesterDoc(
    summary="LSTM final sugg v3 for F3 (final candidates with repeats).",
    params=[
        ParamCat("config", ["AR1", "AR2", "NAR"]),
        ParamCat("repeat_idx", list(range(10))),
    ],
    notes=[
        "`repeat_idx` repeats each config 10×.",
        "AR1/2: selected autoregressive configurations; NAR: selected non-autoregressive configuration.",
    ],
)

@register_suggester("final_v3_F3", doc=final_v3_F3_doc)
def suggester_final_v3_F3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F2') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(10)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    config = trial.suggest_categorical("config", ["AR1", "AR2", "NAR"])
    trial.set_user_attr("config", config)
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 10
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.train.lr_drop_epoch = 4
    cfg.train.tf_drop_epochs = 4
    cfg.train.learning_rate = 8e-04

    if config == "NAR":
        cfg.model.hidden_size = 32
        cfg.model.num_layers = 1
        cfg.model.use_ar  = "none"
        cfg.model.dropout = 0.2
        cfg.train.learning_rate = 6e-04
    elif config == "AR1":
        cfg.model.hidden_size = 32
        cfg.model.num_layers = 1
        cfg.model.use_ar  = "24h"
        cfg.model.dropout = 0.25
    elif config == "AR2":
        cfg.model.hidden_size = 16
        cfg.model.num_layers = 1
        cfg.model.use_ar  = "24h"
        cfg.model.dropout = 0.05
    return cfg

final_v3_F4_doc = SuggesterDoc(
    summary="LSTM final sugg v3 for F4 (final candidates with repeats).",
    params=[
        ParamCat("train.lr_drop_epoch", [4, 6, 8]),
        ParamCat("train.tf_drop_epochs", [4, 6, 8]),
        ParamCat("repeat_idx", list(range(7))),
    ],
    notes=[
        "`repeat_idx` repeats each config 7×.",
    ],
)

@register_suggester("final_v3_F4", doc=final_v3_F4_doc)
def suggester_final_v3_F4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F4') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(7)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 10
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.train.lr_drop_epoch = trial.suggest_categorical("train.lr_drop_epoch", [4, 6, 8])
    cfg.train.tf_drop_epochs = trial.suggest_categorical("train.tf_drop_epochs", [4, 6, 8])
    cfg.train.learning_rate = 4e-04
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.1
    cfg.model.use_ar  = "24h"

    return cfg

final_v3_NAR_F4_doc = SuggesterDoc(
    summary="LSTM final sugg v3 for F4 (final candidates with repeats).",
    params=[
        ParamCat("config", ["AR", "NAR"]),
        ParamCat("repeat_idx", list(range(7))),
    ],
    notes=[
        "`repeat_idx` repeats each config 7×.",
    ],
)

@register_suggester("final_v3_NAR_F4", doc=final_v3_NAR_F4_doc)
def suggester_final_v3_NAR_F4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F4') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(7)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = False
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 10
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.train.tf_drop_epochs = 8
    cfg.train.learning_rate = 4e-04
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.1
    config = trial.suggest_categorical("config", ["AR", "NAR"])
    trial.set_user_attr("config", config)
    if config == "NAR":
        cfg.model.use_ar  = "none"
    else:
        cfg.model.use_ar  = "24h"

    return cfg

final_v3_F5_doc = SuggesterDoc(
    summary="LSTM final sugg v3 for F5 (final candidates with repeats).",
    params=[
        ParamCat("repeat_idx", list(range(10))),
    ],
    notes=[
        "`repeat_idx` repeats each config 10×.",
    ],
)

@register_suggester("final_v3_F5", doc=final_v3_F5_doc)
def suggester_final_v3_F5(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F5') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(10)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 10
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.train.lr_drop_epoch = 4
    cfg.train.tf_drop_epochs = 4
    cfg.train.learning_rate = 1.7e-03
    cfg.model.hidden_size = 8
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.0
    cfg.model.use_ar  = "24h"

    return cfg

@register_suggester("final_v4_F1", doc=final_v4_F1_doc)
def suggester_final_v4_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 24 # 1day ahead

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 64
    cfg.model.num_layers = 2
    cfg.train.lr_drop_epoch = 4
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 3e-4, 1.5e-3, log=True)
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.2)
    cfg.model.use_ar  = "none"

    return cfg

final_v4_F2_doc = SuggesterDoc(
    summary="LSTM final sugg v4 for F2 (24h model finetuning).",
    params=[
        ParamFloat("train.learning_rate", 3e-4, 1.5e-3, log=True),
        ParamFloat("model.dropout", 0.0, 0.2),
    ],
    notes=[
        "Other params used the same fixed values as in `final_v3_F2`:",
        " batch_size=64, input_len=72, head='linear', include_exog_lags=True, lr_drop_epoch=4, lr_drop_factor=0.3, "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.0), norm.mode='global', hidden_size=112, num_layers=1, use_ar='none'.",
    ],
)

@register_suggester("final_v4_F2", doc=final_v4_F2_doc)
def suggester_final_v4_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 24 # 1day ahead

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 112
    cfg.model.num_layers = 1
    cfg.train.lr_drop_epoch = 4
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 3e-4, 1.5e-3, log=True)
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.2)
    cfg.model.use_ar  = "none"

    return cfg

final_v4_F3_doc = SuggesterDoc(
    summary="LSTM final sugg v4 for F3 (24h model finetuning).",
    params=[
        ParamFloat("train.learning_rate", 3e-4, 1.5e-3, log=True),
        ParamFloat("model.dropout", 0.0, 0.4),
    ],
    notes=[
        "Other params used the same fixed values as in `final_v3_F3`:",
        " batch_size=64, input_len=72, head='linear', include_exog_lags=True, lr_drop_epoch=4, lr_drop_factor=0.3, "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.0), norm.mode='global', hidden_size=32, num_layers=1, use_ar='none'.",
    ],
)

@register_suggester("final_v4_F3", doc=final_v4_F3_doc)
def suggester_final_v4_F3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 24 # 1day ahead

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 32
    cfg.model.num_layers = 1
    cfg.train.lr_drop_epoch = 4
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 3e-4, 1.5e-3, log=True)
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.4)
    cfg.model.use_ar  = "none"

    return cfg

final_v4_F4_doc = SuggesterDoc(
    summary="LSTM final sugg v4 for F4 (24h model finetuning).",
    params=[
        ParamFloat("train.learning_rate", 1e-4, 1.5e-3, log=True),
        ParamFloat("model.dropout", 0.0, 0.4),
    ],
    notes=[
        "Other params used the same fixed values as in `final_v3_F4`:",
        " batch_size=64, input_len=72, head='linear', include_exog_lags=True, use_lr_drop=False, "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.0), norm.mode='global', hidden_size=16, num_layers=1, use_ar='none'.",
    ],
)

@register_suggester("final_v4_F4", doc=final_v4_F4_doc)
def suggester_final_v4_F4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 24 # 1day ahead

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = False
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 1e-4, 1.5e-3, log=True)
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.4)
    cfg.model.use_ar  = "none"

    return cfg

final_v4_F5_doc = SuggesterDoc(
    summary="LSTM final sugg v4 for F5 (24h model finetuning).",
    params=[
        ParamFloat("train.learning_rate", 1e-4, 1.5e-3, log=True),
        ParamFloat("model.dropout", 0.0, 0.3),
    ],
    notes=[
        "Other params used the same fixed values as in `final_v3_F5`:",
        " batch_size=64, input_len=72, head='linear', include_exog_lags=True, use_lr_drop=False, "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.0), norm.mode='global', hidden_size=8, num_layers=1, use_ar='none'.",
    ],
)

@register_suggester("final_v4_F5", doc=final_v4_F5_doc)
def suggester_final_v4_F5(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 24 # 1day ahead

    # ---- Config ----
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = False
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.hidden_size = 8
    cfg.model.num_layers = 1
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 1e-4, 1.5e-3, log=True)
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.4)
    cfg.model.use_ar  = "none"

    return cfg
