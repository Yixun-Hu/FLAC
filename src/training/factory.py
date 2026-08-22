import torch
from torch.nn import Parameter
from ..data import are_anchor
from ..models.factory import create_model_from_config

YAW_AUG_KEYS = ("enabled", "img_w", "seed")


def _parse_yaw_aug_config(training_config):
    """Validate ``training.yaw_aug`` and return the wrapper kwargs it implies.

    exp_15's training-side random-yaw augmentation (plan §§3.1, 6.2). Every
    failure mode here is fail-closed: a malformed block must stop the launch, not
    quietly train the wrong arm — the whole experiment is one treatment against
    one historical control, so a silently-off (or silently-on) augmentation is
    unrecoverable after the fact.

    Returns ``{}`` unless the block is present *and* enabled, so the disabled
    path's construction call stays literally the pre-change call (plan §3.3-4:
    the control was trained through that call).
    """
    if "yaw_aug" not in training_config:
        return {}

    block = training_config["yaw_aug"]
    if not isinstance(block, dict):
        raise ValueError(
            f"training.yaw_aug must be an object with keys {list(YAW_AUG_KEYS)}, "
            f"got {type(block).__name__}"
        )

    unknown = [k for k in block if k not in YAW_AUG_KEYS]
    if unknown:
        raise ValueError(
            f"training.yaw_aug has unknown key(s) {sorted(unknown)}; "
            f"allowed keys are {list(YAW_AUG_KEYS)}"
        )

    enabled = block.get("enabled", None)
    if not isinstance(enabled, bool):
        raise ValueError(
            "training.yaw_aug.enabled must be a literal boolean (true/false), got "
            f"{enabled!r}"
        )

    if not enabled:
        return {}

    # exp_21 widens this to every frame-averaged method (the wrapper's own guard
    # is widened in step). The reason is the method's, not the arm's: the orbit
    # already symmetrises over exactly the subgroup the augmentation would draw
    # from, so composing them is neither a no-op nor anything either experiment
    # declared.
    cond_method = training_config.get("cond_method", "vanilla")
    if cond_method in ("fa_invariant", "fa_cartesian"):
        raise ValueError(
            f"training.yaw_aug.enabled=true with cond_method={cond_method!r} is an "
            "untested combination and out of scope for exp_15 (fa_invariant) and "
            "exp_21 (fa_cartesian): frame averaging already symmetrises over the "
            "yaw subgroup."
        )

    for key in ("img_w", "seed"):
        if key not in block:
            raise ValueError(
                f"training.yaw_aug.enabled=true requires '{key}' (no default is "
                "assumed: the applied rotation must be stated by the config)"
            )
        if isinstance(block[key], bool) or not isinstance(block[key], int):
            raise ValueError(
                f"training.yaw_aug.{key} must be an int, got {block[key]!r}"
            )

    if block["img_w"] <= 0:
        raise ValueError(
            f"training.yaw_aug.img_w must be > 0, got {block['img_w']}"
        )

    return {
        "yaw_aug_enabled": True,
        "yaw_aug_img_w": int(block["img_w"]),
        "yaw_aug_seed": int(block["seed"]),
    }


def _parse_frame_avg_cap_config(training_config):
    """Validate ``training.frame_avg_max_fwd_samples`` and return its wrapper kwargs.

    exp_14's treatment (plan §3.1). The frame-average chunk plan — how many orbit
    angles share one conditioner forward, and therefore one train-mode DINOv3
    RoPE draw — used to be derived from a module constant
    (``yaw_rotation.FRAME_AVG_MAX_FWD_SAMPLES``), which announcement 06 flags as
    "derived, not declared": the same JSON at a different micro-batch, or a month
    apart, trains a different method. Declaring it here puts it in the config that
    ``train.py`` embeds into every checkpoint, so an arm's chunk plan is auditable
    after the fact and comparable across arms.

    Absent key -> ``{}``, so the wrapper construction call is LITERALLY the
    pre-change call and every recipe already in the record is byte-identical.

    Every failure mode is fail-closed on the RAW value: a coerced ``int("32")`` or
    a ``True`` that ``isinstance(_, int)`` happily accepts would arm the wrong
    chunk plan for a six-day run without a word of complaint.
    """
    key = "frame_avg_max_fwd_samples"
    if key not in training_config:
        return {}

    value = training_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"training.{key} must be an int (the frame-average per-forward sample "
            f"cap), got {value!r}"
        )
    if value < 1:
        raise ValueError(f"training.{key} must be >= 1, got {value}")

    return {key: int(value)}


ARE_LAMBDA_KEY = "are_lambda"
ARE_ANCHOR_KEY = "are_anchor"


def _parse_are_config(model_config, training_config):
    """Validate ``training.are_lambda`` / ``training.are_anchor`` -> wrapper kwargs.

    exp_16's treatment (plan §§1-3): FLAC's rectified flow learns
    ``noise -> (z - lambda*A(p))`` instead of ``noise -> z``. That is the arm's
    OBJECTIVE, so it has to be a declared, checkpoint-embedded property rather
    than an ambient constant — the same reasoning announcement 06 applies to the
    frame-average chunk plan.

    Returns ``{}`` unless ``are_lambda`` is present, so an arm that does not
    declare it is constructed through the LITERAL pre-change call. That matters
    concretely here: exp_16's lambda=0 control is P1, a run that already
    happened, and the comparison is only clean if the absent-key path is
    byte-identical to the one P1 was trained through.

    Every check is on the RAW value. ``True`` is an ``int``, ``"1"`` is not a
    number however plausible it looks, and either would arm a ~1.8-day run at a
    lambda nobody chose.
    """
    has_lambda = ARE_LAMBDA_KEY in training_config
    has_anchor = ARE_ANCHOR_KEY in training_config

    if not has_lambda:
        if has_anchor:
            raise ValueError(
                f"training.{ARE_ANCHOR_KEY} is declared but training.{ARE_LAMBDA_KEY} "
                "is not: an anchor block with no lambda would be parsed, recorded and "
                "then ignored. State the lambda (0.0 for a declared control) or drop "
                "the block.")
        return {}

    lam = training_config[ARE_LAMBDA_KEY]
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        raise ValueError(
            f"training.{ARE_LAMBDA_KEY} must be a number in [0, 1] (the weight of the "
            f"analytic anchor in the target reparameterisation), got {lam!r}")
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError(
            f"training.{ARE_LAMBDA_KEY} must be in [0, 1], got {lam!r}")

    if not has_anchor:
        raise ValueError(
            f"training.{ARE_LAMBDA_KEY}={lam!r} requires a training.{ARE_ANCHOR_KEY} "
            "block carrying the calibrated constants "
            f"{list(are_anchor.ANCHOR_REQUIRED_KEYS)}: no default is assumed, because "
            "an anchor placed at an uncalibrated onset is a different method.")

    block = training_config[ARE_ANCHOR_KEY]
    if not isinstance(block, dict):
        raise ValueError(
            f"training.{ARE_ANCHOR_KEY} must be an object with keys "
            f"{list(are_anchor.ANCHOR_REQUIRED_KEYS)}, got {type(block).__name__}")

    allowed = set(are_anchor.ANCHOR_REQUIRED_KEYS) | set(are_anchor.ANCHOR_OPTIONAL_KEYS)
    unknown = sorted(k for k in block if k not in allowed)
    if unknown:
        # sample_rate / sample_size are named separately: they are not "unknown",
        # they are the MODEL config's, and letting the block restate them would
        # let the anchor's time base disagree with the audio the run trains on.
        restated = [k for k in unknown if k in are_anchor.ANCHOR_MODEL_KEYS]
        if restated:
            raise ValueError(
                f"training.{ARE_ANCHOR_KEY} may not restate {restated}: they are read "
                "from the model config (sample_rate / sample_size), so a second copy "
                "could silently disagree with the audio being trained on.")
        raise ValueError(
            f"training.{ARE_ANCHOR_KEY} has unknown key(s) {unknown}; allowed keys are "
            f"{sorted(allowed)}")

    for key in are_anchor.ANCHOR_REQUIRED_KEYS:
        if key not in block:
            raise ValueError(
                f"training.{ARE_ANCHOR_KEY} requires '{key}': it is calibrated on the "
                "AR train split and must be stated by the config, never defaulted.")
    for key, value in block.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"training.{ARE_ANCHOR_KEY}.{key} must be a number, got {value!r}")

    resolved = dict(block)
    resolved["sample_rate"] = model_config["sample_rate"]
    resolved["sample_size"] = model_config["sample_size"]
    return {ARE_LAMBDA_KEY: float(lam), ARE_ANCHOR_KEY: resolved}


def create_training_wrapper_from_config(model_config, model):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderTrainingWrapper
        
        ema_copy = None

        if training_config.get("use_ema", False):
            ema_copy = create_model_from_config(model_config)
            ema_copy = create_model_from_config(model_config) # I don't know why this needs to be called twice but it broke when I called it once
            # Copy each weight to the ema copy
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        use_ema = training_config.get("use_ema", False)

        latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)

        teacher_model = training_config.get("teacher_model", None)
        if teacher_model is not None:
            teacher_model = create_model_from_config(teacher_model)
            teacher_model = teacher_model.eval().requires_grad_(False)

            teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
            if teacher_model_ckpt is not None:
                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
            else:
                raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")

        return AutoencoderTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            warmup_steps=training_config.get("warmup_steps", 0), 
            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
            sample_rate=model_config["sample_rate"],
            loss_config=training_config.get("loss_configs", None),
            eval_loss_config=training_config.get("eval_loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=use_ema,
            ema_copy=ema_copy if use_ema else None,
            force_input_mono=training_config.get("force_input_mono", False),
            latent_mask_ratio=latent_mask_ratio,
            teacher_model=teacher_model
        )
    
    elif model_type == 'diffusion_cond':
       
        from .diffusion import DiffusionCondTrainingWrapper

        # exp_15: absent/disabled block -> {} -> the pre-change call verbatim.
        yaw_aug_kwargs = _parse_yaw_aug_config(training_config)
        # exp_14: absent key -> {} -> likewise verbatim (cap defaults to the
        # module's 64 inside invariant_conditioning, not here).
        frame_avg_cap_kwargs = _parse_frame_avg_cap_config(training_config)
        # exp_16: absent are_lambda -> {} -> likewise verbatim (P1, the lambda=0
        # control, was trained through exactly this call).
        are_kwargs = _parse_are_config(model_config, training_config)

        return DiffusionCondTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            mask_padding=training_config.get("mask_padding", False),
            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
            use_ema = training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
            timestep_sampler_options = training_config.get("timestep_sampler_options", {}),
            p_one_shot=training_config.get("p_one_shot", 0.0),
            test_param = model_config.get("test_setup", None),
            cond_method = training_config.get("cond_method", "vanilla"),
            frame_avg_angles = training_config.get("frame_avg_angles", None),
            **frame_avg_cap_kwargs,
            **yaw_aug_kwargs,
            **are_kwargs,
        )
    
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_metric_callback_from_config(model_config, dataset_id=None, per_scene=False):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    metrics_config = training_config.get('metrics', None)
    assert metrics_config is not None, 'metrics config must be specified in training config'

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    audio_channels = model_config.get("audio_channels", 1)

    from ..metrics.metric_callback import AcousticMetricsCallback
    return AcousticMetricsCallback(
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=audio_channels,
        dataset_name=dataset_id,
        eval_per_scene=per_scene,
        
        dump_dir=metrics_config.get("dump_dir", None),

        eval_T60=metrics_config.get("eval_T60", False),
        eval_C50=metrics_config.get("eval_C50", False),
        eval_EDT=metrics_config.get("eval_EDT", False),
        eval_l1_distance=metrics_config.get("eval_l1_distance", False),
        eval_l1_distance_multires=metrics_config.get("eval_l1_distance_multires", False),
        eval_FD=metrics_config.get("eval_FD", False),
        eval_retrieval = metrics_config.get("eval_retrieval", False),
        eval_env = metrics_config.get("eval_env", False),

        AGREE_ckpt=metrics_config.get("AGREE_ckpt", None),
    )
