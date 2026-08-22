import torch
import json
import os
import pytorch_lightning as pl
from prefigure.prefigure import get_all_args

from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config

# Conditioning methods this entry point must REFUSE to evaluate (exp_21 round-2
# code review, nit 3).
#
# eval_pl drives PL's test loop and writes `{metrics, ckpt_path}` -- and nothing
# else. None of the provenance `eval_FLAC.build_metrics_record` attaches exists
# here: no cond_method, no frame_avg_angles, no orbit_execution or
# frame_avg_fwd_cap, no source_sha, no batch size, seed, dataset config or
# weights source. A row produced here is therefore indistinguishable after the
# fact from a vanilla one, which is the failure announcement 05 exists to
# prevent, and the exp_21 admission validator (plan §3g) has nothing to check.
#
# The guard is needed because the training-wrapper factory ACCEPTS fa_cartesian
# -- it must, training uses it -- so eval_pl inherits the dispatch for free and
# would run the arm and emit an unprovenanced number. Failing closed is cheaper
# than a worklog rule: exp_21 headline rows come from eval_FLAC.py.
#
# The registered methods are deliberately NOT listed: vanilla and fa_invariant
# rows predate this entry point's provenance gap, and adding them would change
# behaviour for runs already in the record.
UNREGISTERED_COND_METHODS = ("fa_cartesian",)


def reject_unregistered_cond_method(model_config):
    """Fail closed BEFORE any model or wrapper is built, or any GPU is touched.

    Position matters as much as the check: raising later would still produce the
    error, but only after a multi-hour test loop had run, and a caller who caught
    it could already hold a half-written record.
    """
    cond_method = ((model_config or {}).get("training") or {}).get("cond_method")
    if cond_method in UNREGISTERED_COND_METHODS:
        raise ValueError(
            f"eval_pl.py refuses to evaluate cond_method={cond_method!r}: this entry "
            "point writes only {metrics, ckpt_path}, so it records no cond_method, "
            "no frame_avg_angles, no orbit execution or cap, and no source SHA -- it "
            "has NO registered-eval provenance, and its output is indistinguishable "
            "from a vanilla row after the fact. exp_21 headline rows must come from "
            "eval_FLAC.py --cond-method fa_cartesian (which records all of the above "
            "and is what the model-comparison admission validator reads)."
        )


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    torch.set_float32_matmul_precision('medium') 
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    # Get model  
    with open(args.model_config) as f:
        model_config = json.load(f)
    reject_unregistered_cond_method(model_config)
    model = create_model_from_config(model_config)

    # Get dataset 
    assert args.val_dataset_config, "You must provide an eval dataset config file."
    with open(args.val_dataset_config) as f:
        eval_dataset_config = json.load(f)

    eval_dl = create_dataloader_from_config(
        eval_dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=False,
    )

    # Metrics: Give test setup
    model_config['test_setup'] = {
        'samples': model_config["sample_size"],
        'cfg_scale': args.cfg_scale,
        'steps': args.steps,
        'sample_rate': model_config["sample_rate"],
        'audio_channels': model_config.get("audio_channels", 1),
        'AGREE_ckpt': model_config['training'].get("AGREE_ckpt", None),
        'store_predictions': args.store_predictions,
        }
    
    model_config['test_setup']['metrics'] = model_config['training']['metrics']

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()
    
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
   
    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": eval_dataset_config})
    args_dict.update({"eval_dataset_config": eval_dataset_config})

    trainer = pl.Trainer(
        devices=args.num_gpus,#"auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[exc_callback, save_model_config_callback],
        log_every_n_steps=100,
        max_steps=1000000,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
    )

    assert args.ckpt_path, "You must provide a checkpoint path to load the model."
    trainer.test(training_wrapper, eval_dl, ckpt_path=args.ckpt_path)

    metrics_dict = training_wrapper.metrics_dict
    metrics_to_save = {
        "metrics": metrics_dict,
        "ckpt_path": args.ckpt_path,
    }
    
    ckpt_name = os.path.basename(args.ckpt_path).replace('.ckpt', '')
    path2save = os.path.join(os.path.dirname(args.ckpt_path), ckpt_name + '_metrics_' + str(args.steps) + '_' + str(args.cfg_scale) + '_' + '.json')
    with open(path2save, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Metrics saved to {path2save}")

    if training_wrapper.store_predictions:
        decoded_samples_all = torch.cat(training_wrapper.preds, dim=0) 
        path2save_preds = os.path.join(os.path.dirname(args.ckpt_path), ckpt_name + '_predictions_' + str(args.steps) + '_' + str(args.cfg_scale)  + '.pt')
        torch.save(decoded_samples_all, path2save_preds)
        print(f"Decoded samples saved to {path2save_preds}")
    
    print('Evaluation complete!')


if __name__ == '__main__':
    main()
