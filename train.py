import argparse
import json
import random

import numpy as np
import torch

from trajvq.config import load_config
from trajvq.data import load_dataset
from trajvq.trainer import Trainer


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate one TrajVQ stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", required=True, help="Split dataset directory, see README.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", help="Continuous-stage model.pt; required when init_from_continuous is set.")
    parser.add_argument("--teacher_dir", help="Output of fit_opq_teacher.py; required when teacher_weight > 0.")
    parser.add_argument("--checkpoint", help="Evaluate this model.pt instead of training.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training = args.checkpoint is None
    if training and cfg["init_from_continuous"] != bool(args.backbone):
        parser.error("--backbone must be given exactly when the config sets init_from_continuous")
    if training and (cfg["teacher_weight"] > 0) != bool(args.teacher_dir):
        parser.error("--teacher_dir must be given exactly when the config sets teacher_weight > 0")

    setup_seed(cfg["seed"])
    trainer = Trainer(cfg, load_dataset(args.data_root, cfg["measure"]), args.device, args.output_dir,
                      backbone=args.backbone if training else None,
                      teacher_dir=args.teacher_dir if training else None)
    (trainer.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    if training:
        trainer.train()
    else:
        trainer.evaluate(trainer.load_checkpoint(args.checkpoint))


if __name__ == "__main__":
    main()
