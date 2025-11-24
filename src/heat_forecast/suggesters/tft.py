
## Suggesters for TFT

import optuna
import copy

from heat_forecast.pipeline.tft import TFTRunConfig

from heat_forecast.utils.optuna import (
    register_suggester, 
    SuggesterDoc,
    ParamCat, ParamInt, ParamFloat
)

_tft_v1_F1_doc = SuggesterDoc(
    summary="TFT search space for 24-hour ahead forecasting.",
    params=[
        # Model
        ParamInt("model.hidden_size_idx", 0, 4),  
        ParamFloat("model.dropout", 0.0, 0.6),
        ParamCat("model.num_attention_heads", [1, 4]),

        # Data
        ParamCat("data.batch_size", [64, 128]),

        # Training
        ParamFloat("train.learning_rate", 1e-4, 1e-2, log=True),
    ],
    notes=[
        "`model.hidden_size_idx` is used to select the hidden size from the list [20, 40, 80, 120, 160].",
        "`model.input_chunk_length` is fixed to 168+72 and `model.output_chunk_length` is fixed to 168.",
        "`train.n_epochs` is fixed to 17; early stopping trims excess epochs.",
        "`train.es_rel_min_delta` is fixed to 0.05 and `train.es_warmup_epochs` is fixed to 5.",
        "After the first approx 50 trials, we noted that in all trials little to no improvement happend after 8 epochs, so we set `train.es_warmup_epochs=3` and `train.n_epochs=8`.",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number (and base_cfg.seed if provided).",
        "Other params use defaults.",
    ],
)

@register_suggester("tft_v1_F1", doc=_tft_v1_F1_doc)
def suggest_config_v1_F1(trial: optuna.trial.Trial, base: TFTRunConfig) -> TFTRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model ----
    cfg.model.input_chunk_length  = 168 + 72
    cfg.model.output_chunk_length = 168
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.6)
    cfg.model.num_attention_heads = trial.suggest_categorical("model.num_attention_heads", [1, 4])


    hidden_size_choices = [20, 40, 80, 120, 160]
    idx = trial.suggest_int("model.hidden_size_idx", 0, len(hidden_size_choices)-1)
    cfg.model.hidden_size = hidden_size_choices[idx]
    trial.set_user_attr("model.hidden_size", cfg.model.hidden_size)

    # ---- Training/Data ----
    cfg.data.batch_size = trial.suggest_categorical("data.batch_size", [64, 128])
    cfg.train.lr = trial.suggest_float("train.learning_rate", 1e-4, 1e-2, log=True)
    cfg.train.n_epochs = 8
    cfg.train.es_rel_min_delta = 0.05
    cfg.train.es_warmup_epochs = 3

    return cfg

_tft_v2_F1_doc = SuggesterDoc(
    summary="Tiny study to choose n_epochs for TFT & week-ahead forecasting.",
    params=[
        ParamCat("repeat_idx", [0, 1, 2, 3, 4]),
    ],
    notes=[
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Params use the final configuration for F1/week from the main study.",
    ],
)

@register_suggester("tft_v2_F1", doc=_tft_v2_F1_doc)
def suggest_config_v2_F1(trial: optuna.trial.Trial, base: TFTRunConfig) -> TFTRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('tft_v2_F1') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3, 4])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model ----
    cfg.model.input_chunk_length  = 168 + 72
    cfg.model.output_chunk_length = 168
    cfg.model.dropout = 0.15
    cfg.model.num_attention_heads = 1
    cfg.model.hidden_size = 40

    # ---- Training/Data ----
    cfg.data.batch_size = 64
    cfg.train.gradient_clip_val = 10.0
    cfg.train.lr = 3e-3
    cfg.train.n_epochs = 10
    cfg.train.es_warmup_epochs = 100 # disable early stopping

    return cfg

_tft_v3_F1_doc = SuggesterDoc(
    summary="Finetuning of the model for the daily forecast task.",
    params=[
        ParamFloat("train.learning_rate", 1e-4, 1e-2, log=True),
        ParamFloat("model.dropout", 0.0, 0.5),
    ],
    notes=[
        "At each trial, the chosen cfg receives a seed derived from study name and trial number (and base_cfg.seed if provided).",
        "Other params use the final configuration for F1/week from the main study.",
    ],
)

@register_suggester("tft_v3_F1", doc=_tft_v3_F1_doc)
def suggest_config_v3_F1(trial: optuna.trial.Trial, base: TFTRunConfig) -> TFTRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model ----
    cfg.model.input_chunk_length  = 168 + 72
    cfg.model.output_chunk_length = 24
    cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.5)
    cfg.model.num_attention_heads = 1
    cfg.model.hidden_size = 40

    # ---- Training/Data ----
    cfg.data.batch_size = 64
    cfg.train.gradient_clip_val = 10.0
    cfg.train.lr = trial.suggest_float("train.learning_rate", 1e-4, 1e-2, log=True)
    cfg.train.n_epochs = 17
    cfg.train.es_rel_min_delta = 0.05
    cfg.train.es_warmup_epochs = 8

    return cfg

_tft_v4_F1_doc = SuggesterDoc(
    summary="Final study to finalize n_epochs for TFT & day-ahead forecasting.",
    params=[
        ParamCat("repeat_idx", [0, 1, 2, 3, 4]),
    ],
    notes=[
        "From the previous study, 4-5 epochs seem the best candidates.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Params use the final configuration for F1/day from the previous study.",
    ],
)

@register_suggester("tft_v4_F1", doc=_tft_v4_F1_doc)
def suggest_config_v4_F1(trial: optuna.trial.Trial, base: TFTRunConfig) -> TFTRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('tft_v4_F1') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3, 4])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model ----
    cfg.model.input_chunk_length  = 168 + 72
    cfg.model.output_chunk_length = 24
    cfg.model.dropout = 0.05
    cfg.model.num_attention_heads = 1
    cfg.model.hidden_size = 40

    # ---- Training/Data ----
    cfg.data.batch_size = 64
    cfg.train.gradient_clip_val = 10.0
    cfg.train.lr = 2.5e-3
    cfg.train.n_epochs = 8
    cfg.train.es_rel_min_delta = 0.05
    cfg.train.es_warmup_epochs = 100

    return cfg
