import json

REQUIRED = ("dataset", "measure", "stage", "epochs", "image_mode", "aux_kernel_sizes")

DEFAULTS = {
    "seed": 666,
    "batch_size": 96,
    "learning_rate": 0.001,
    "metric_loss_norm": "none",
    "eval_batch_size": 1024,
    "eval_chunk_size": 10240,
    "pdt_m": 16,
    "pdt_k": 256,
    "pdt_steps": 1,
    "pdt_heads": 1,
    "pdt_input_norm": False,
    "init_from_continuous": False,
    "pdt_loss_start_epoch": None,
    "pdt_loss_weight": 0.0,
    "adc_ste_start_epoch": 0,
    "adc_ste_warmup_epochs": 0,
    "adc_ste_max_weight": 0.0,
    "reg_start_epoch": None,
    "commitment_weight": 0.0,
    "entropy_weight": 0.0,
    "uniform_weight": 0.0,
    "codebook_refresh_epochs": [],
    "teacher_weight": 0.0,
    "teacher_end_epoch": 0,
    "teacher_codebook_freeze_epochs": 0,
    "late_finetune_start_epoch": None,
    "late_finetune_lr_scale": 1.0,
    "dead_code_restart_epochs": [],
    "dead_code_min_usage": 0,
    "dead_code_max_per_group": 0,
    "dead_code_sample_limit": 0,
    "dead_code_noise": 0.0,
}


def load_config(path):
    with open(path) as handle:
        user = json.load(handle)
    unknown = sorted(set(user) - set(DEFAULTS) - set(REQUIRED))
    missing = [key for key in REQUIRED if key not in user]
    if unknown or missing:
        raise ValueError("{}: unknown keys {}, missing keys {}".format(path, unknown, missing))
    cfg = {**DEFAULTS, **user}
    if cfg["stage"] not in ("continuous", "vq"):
        raise ValueError("stage must be 'continuous' or 'vq'")
    if cfg["metric_loss_norm"] not in ("none", "target_train_mean"):
        raise ValueError("metric_loss_norm must be 'none' or 'target_train_mean'")
    if cfg["stage"] == "vq" and cfg["pdt_loss_start_epoch"] is None:
        raise ValueError("the vq stage needs pdt_loss_start_epoch")
    if cfg["dead_code_restart_epochs"] and (cfg["dead_code_sample_limit"] <= 0 or cfg["dead_code_max_per_group"] <= 0):
        raise ValueError("dead-code restart needs positive dead_code_sample_limit and dead_code_max_per_group")
    return cfg
