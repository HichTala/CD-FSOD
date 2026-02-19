import os

import detectron2.utils.comm as comm
from torch import manual_seed

import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from model import add_kd_config
from model.engine.trainer import KdTrainer

# hacky way to register
from model.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from model.modeling.proposal_generator.rpn import PseudoLabRPN
from model.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import model.data.datasets.builtin

from model.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from train_net import main


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_kd_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    seed = os.getenv("REPEAT_ID", 2026)
    manual_seed(seed)
    output_dir = cfg.OUTPUT_DIR
    cfg.merge_from_list(['OUTPUT_DIR', os.path.join(output_dir, str(seed)), 'MODEL.WEIGHTS',
                         os.path.join('runs', f"{cfg.DATASETS.TRAIN[0]}_{str(seed)}", "model_final.pth")])
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
