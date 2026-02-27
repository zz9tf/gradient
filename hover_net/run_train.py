"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>] [--resume=<path>]
    [--batch-train=<n>] [--batch-valid=<n>]
    [--grad-mode=<m>] [--grad-eps=<f>] [--grad-common-gate-rho-thr=<f>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help               Show this string.
  --version               Show version.
  --gpu=<id>              Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>           Visualise images after augmentation. Choose 'train' or 'valid'.
  --resume=<path>         Resume training from this log dir (same grad overrides; set nr_epochs in opt to target total epochs).
  --batch-train=<n>       Batch size for training (per GPU). Overrides phase config.
  --batch-valid=<n>       Batch size for validation and test (per GPU). Overrides phase config.
  --grad-mode=<m>         Gradient aggregation mode: sum|pcgrad|graddrop|mgda|cagrad|dwa|gradnorm|uw_heuristic|nash_mtl (overrides config).
  --grad-eps=<f>          Epsilon for gradient aggregation (GradAggConfig.eps override).
  --grad-common-gate-rho-thr=<f>  Enable Gpop common gate and set rho_thr in [-1,1] (GradAggConfig.gpop_enabled=True, gpop_rho_thr override).
"""

import cv2

cv2.setNumThreads(0)
import argparse
import glob
import importlib
import inspect
import json
import os
import shutil

import matplotlib
import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from gradient_wrapper.grad_wrapper import GradAggregator, GradAggConfig

from config import Config
from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine, Events
from run_utils.utils import (
    check_log_dir,
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
)
from run_utils.callbacks.base import (
    AccumulateRawOutput,
    ProcessAccumulatedRawOutput,
)
from run_utils.callbacks.logging import LoggingEpochOutput
from models.hovernet.run_desc import proc_valid_step_output, valid_step
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#### have to move outside because of spawn
# * must initialize augmentor per worker, else duplicated rng generators may happen
def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


def _parse_grad_overrides(args):
    """
    Build gradient override dict from CLI args. Omitted options keep config defaults.

    Args:
        args: dict from docopt (keys --grad-mode, --grad-beta, etc.).

    Returns:
        dict: keys compatible with GradAggConfig (mode, eps, gpop_enabled, gpop_rho_thr); only present if given on CLI.
    """
    overrides = {}
    if args.get("--grad-mode"):
        overrides["mode"] = args["--grad-mode"]
    if args.get("--grad-common-gate-rho-thr"):
        overrides["gpop_enabled"] = True
        overrides["gpop_rho_thr"] = float(args["--grad-common-gate-rho-thr"])
    if args.get("--grad-eps"):
        overrides["eps"] = float(args["--grad-eps"])
    return overrides


def _grad_overrides_to_log_suffix(grad_overrides):
    """
    Build a log subdir suffix from gradient overrides so results go to a different path.

    Args:
        grad_overrides: dict with keys mode, eps, gpop_enabled, gpop_rho_thr (any subset).

    Returns:
        str: e.g. "grad_pgrs" or "grad_pgrs_lpf1_gknp", or "" if empty.
    """
    if not grad_overrides:
        return ""
    parts = ["grad", grad_overrides.get("mode", "sum")]
    if "eps" in grad_overrides:
        parts.append("e%g" % grad_overrides["eps"])
    if grad_overrides.get("gpop_enabled"):
        parts.append("gpop")
    if "gpop_rho_thr" in grad_overrides:
        parts.append("rho%g" % grad_overrides["gpop_rho_thr"])
    return "_".join(parts)


def _batch_overrides_to_log_suffix(batch_overrides):
    """
    Build a log dir suffix from batch overrides (only when any override is set).
    E.g. {"train": 16} -> "_bt_16", {"valid": 8} -> "_bv_8", both -> "_bt_16_bv_8".

    Returns:
        str: suffix including leading underscore, or "" if no overrides.
    """
    if not batch_overrides:
        return ""
    parts = []
    if "train" in batch_overrides:
        parts.append("bt_%d" % int(batch_overrides["train"]))
    if "valid" in batch_overrides:
        parts.append("bv_%d" % int(batch_overrides["valid"]))
    return "_" + "_".join(parts) if parts else ""


####
class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""

    def __init__(self, grad_overrides=None, resume_path=None, batch_overrides=None):
        """
        Args:
            grad_overrides: Optional dict to override GradAggConfig
                (mode, eps, gpop_enabled, gpop_rho_thr, and other GradAggConfig fields).
            resume_path: If set, resume training from this log dir (same grad overrides; set nr_epochs in opt to target total).
            batch_overrides: Optional dict to override batch_size, e.g. {"train": 16, "valid": 8}.
        """
        super().__init__()
        self.grad_overrides = grad_overrides or {}
        self.resume_path = resume_path
        self.batch_overrides = batch_overrides or {}
        return

    def _apply_batch_overrides(self, batch_size_dict):
        """Return a copy of batch_size_dict with self.batch_overrides applied (only for existing keys)."""
        out = dict(batch_size_dict)
        for k, v in self.batch_overrides.items():
            if k in out:
                out[k] = v
        return out

    ####
    def view_dataset(self, mode="train"):
        """
        Manually change to plt.savefig or plt.show 
        if using on headless machine or not
        """
        self.nr_gpus = 1
        import matplotlib.pyplot as plt
        check_manual_seed(self.seed)
        # TODO: what if each phase want diff annotation ?
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for batch_data in dataloader:  
            # convert from Tensor to Numpy
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            out = f"debug_view_{mode}.png"
            plt.imsave(out, viz)
            print("saved:", out, "viz shape:", viz.shape, "dtype:", viz.dtype, "min/max:", viz.min(), viz.max())
            break
        self.nr_gpus = -1
        return

    ####
    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        """
        Get dataloader for specified run mode.
        
        Args:
            batch_size: Batch size per GPU
            run_mode: One of 'train', 'valid', or 'test'
            target_gen: Target generation function
            nr_procs: Number of worker processes
            fold_idx: Fold index for cross-validation
            
        Returns:
            DataLoader instance
        """
        nr_procs = nr_procs if not self.debug else 0

        # ! Hard assumption on file type
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        elif run_mode == "valid":
            data_dir_list = self.valid_dir_list
        elif run_mode == "test":
            data_dir_list = self.test_dir_list
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")
            
        for dir_path in data_dir_list:
            if not os.path.isabs(dir_path):
                dir_path = os.path.join(os.path.dirname(__file__), dir_path)
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # to always ensure same input ordering

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode]
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0, resume_log_dir=None):
        """
        Run one phase of training/validation.

        Args:
            opt: Phase option (run_info, target_info, batch_size, nr_epochs).
            run_engine_opt: Engine config (train/valid callbacks etc.).
            log_dir: Directory for checkpoints and stats.
            prev_log_dir: Previous phase log dir (for pretrained=-1).
            fold_idx: Fold index for dataloader.
            resume_log_dir: If set and equals log_dir, resume from last checkpoint in log_dir
                (restore model, optimizer, scheduler, grad_agg and continue from next epoch).
                Set nr_epochs in opt to the desired total epoch count (e.g. 60 to add 20 more after 40).
        """
        check_manual_seed(self.seed)

        resume_mode = (
            resume_log_dir is not None
            and os.path.abspath(resume_log_dir) == os.path.abspath(log_dir)
            and os.path.exists(os.path.join(log_dir, "stats.json"))
        )
        resume_start_epoch = 0
        log_info = {}
        if self.logging:
            if resume_mode:
                json_log_file = log_dir + "/stats.json"
                with open(json_log_file) as json_file:
                    stats = json.load(json_file)
                resume_start_epoch = max(int(k) for k in stats.keys())
                tfwriter = SummaryWriter(log_dir=log_dir)
                log_info = {"json_file": json_log_file, "tfwriter": tfwriter}
                print("[resume] Resuming from %s at epoch %d (will run until nr_epochs=%d)" % (log_dir, resume_start_epoch, opt["nr_epochs"]))
            else:
                rm_n_mkdir(log_dir)
                tfwriter = SummaryWriter(log_dir=log_dir)
                json_log_file = log_dir + "/stats.json"
                with open(json_log_file, "w") as json_file:
                    json.dump({}, json_file)
                log_info = {"json_file": json_log_file, "tfwriter": tfwriter}

        ####
        batch_size = self._apply_batch_overrides(opt["batch_size"])
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            print(f"Runner name: {runner_name}")
            loader_dict[runner_name] = self._get_datagen(
                batch_size[runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                prev_phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            # TODO: customize print-out for each run ?
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            full_chkpt = None  # full .tar for grad_agg / optimizer / scheduler restore
            if resume_mode:
                pretrained_path = get_last_chkpt_path(log_dir, net_name)
                if not os.path.exists(pretrained_path):
                    raise FileNotFoundError("Resume dir %s has no checkpoint for net %s (e.g. %s)" % (log_dir, net_name, pretrained_path))
                full_chkpt = torch.load(pretrained_path)
                net_state_dict = full_chkpt["desc"]
                colored_word = colored(net_name, color="red", attrs=["bold"])
                print("Model `%s` resume from: %s" % (colored_word, pretrained_path))
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])
            elif pretrained_path is not None:
                if pretrained_path == -1:
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    full_chkpt = torch.load(pretrained_path)
                    net_state_dict = full_chkpt["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":
                        full_chkpt = torch.load(pretrained_path)
                        net_state_dict = full_chkpt["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # * extremely slow to pass this on DGX with 1 GPU, why (?)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            # TODO: expand for external aug for scheduler
            nr_iter = opt["nr_epochs"] * len(loader_dict["train"])
            scheduler = net_info["lr_scheduler"](optimizer)
            extra_info = dict(net_info["extra_info"])
            grad_mode = self.grad_overrides.get("mode", extra_info["grad_mode"])
            # start from config grad_cfg and apply any CLI overrides except mode
            grad_cfg = dict(extra_info.get("grad_cfg", {}))
            for k, v in self.grad_overrides.items():
                if k == "mode":
                    continue
                grad_cfg[k] = v
            # Store effective config so logs/checkpoints record what is actually used
            extra_info["grad_mode"] = grad_mode
            extra_info["grad_cfg"] = grad_cfg
            # def keeper(name, p):
            #     name = name[7:] if name.startswith("module.") else name
            #     return not name.startswith("decoder.")
            def common_param_filter(name, p):
                name = name[7:] if name.startswith("module.") else name
                return not name.startswith("decoder.")
            grad_agg = GradAggregator(
                net_desc,
                GradAggConfig(mode=grad_mode, **grad_cfg),
                # param_filter=keeper,
                common_param_filter=common_param_filter,
                verbose=True,
            )
            if full_chkpt is not None and "grad_agg" in full_chkpt:
                grad_agg.load_state_dict(full_chkpt["grad_agg"], strict=False)
                print("[gradient] restored grad_agg state (Gpop) from checkpoint")
            if resume_mode and full_chkpt is not None:
                if "optimizer" in full_chkpt:
                    optimizer.load_state_dict(full_chkpt["optimizer"])
                    print("[resume] restored optimizer state")
                if "lr_scheduler" in full_chkpt:
                    scheduler.load_state_dict(full_chkpt["lr_scheduler"])
                    print("[resume] restored lr_scheduler state")
            if self.grad_overrides:
                print(
                    "[gradient] effective: mode=%s, grad_cfg=%s"
                    % (grad_mode, grad_cfg)
                )
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "extra_info": extra_info,
                "grad_agg": grad_agg,
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        start_epoch = resume_start_epoch if resume_mode else 0
        if resume_mode and resume_start_epoch >= opt["nr_epochs"]:
            raise ValueError(
                "Resume start epoch (%d) >= nr_epochs (%d). Increase nr_epochs in opt (e.g. 60) to train more."
                % (resume_start_epoch, opt["nr_epochs"])
            )
        main_runner.run(opt["nr_epochs"], start_epoch=start_epoch)

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run_test(self, log_dir, max_epoch=None):
        """
        Run test evaluation on the test dataset after training.

        Args:
            log_dir: Directory containing the trained model checkpoints.
            max_epoch: If set, choose best checkpoint only among epochs <= max_epoch (inclusive).
        """
        print("\n" + "="*60)
        print("Starting Test Evaluation")
        if max_epoch is not None:
            print("Best checkpoint limited to epoch <= %d" % max_epoch)
        print("="*60 + "\n")

        check_manual_seed(self.seed)
        
        # Get the last phase's checkpoint
        phase_list = self.model_config["phase_list"]
        last_phase_info = phase_list[-1]
        
        # Setup logging for test results
        test_log_dir = log_dir.rstrip("/") + "_test"
        rm_n_mkdir(test_log_dir)
        
        tfwriter = SummaryWriter(log_dir=test_log_dir)
        json_log_file = test_log_dir + "/stats.json"
        with open(json_log_file, "w") as json_file:
            json.dump({}, json_file)
        
        log_info = {
            "json_file": json_log_file,
            "tfwriter": tfwriter,
        }
        
        # Get test dataloader (use valid batch_size as default)
        phase_batch = self._apply_batch_overrides(last_phase_info["batch_size"])
        test_batch_size = phase_batch.get("test", phase_batch.get("valid", 8))
        test_loader = self._get_datagen(
            batch_size=test_batch_size,
            run_mode="test",
            target_gen=last_phase_info["target_info"]["gen"],
            nr_procs=8,  # Use same as valid
        )
        
        max_epoch_for_best = max_epoch

        # Load the best model from training
        def get_best_chkpt_path(phase_dir, net_name):
            """Get the best checkpoint path by valid-np_dice (optionally only epochs <= max_epoch_for_best)."""
            stat_file_path = phase_dir + "/stats.json"
            if not os.path.exists(stat_file_path):
                return get_last_chkpt_path(phase_dir, net_name)

            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)

            best_epoch = None
            best_dice = -1
            for epoch_str, epoch_data in info.items():
                epoch_int = int(epoch_str)
                if max_epoch_for_best is not None and epoch_int > max_epoch_for_best:
                    continue
                dice = None
                if isinstance(epoch_data.get("valid"), dict) and "np_dice" in epoch_data["valid"]:
                    dice = epoch_data["valid"]["np_dice"]
                if dice is None:
                    dice = epoch_data.get("valid-np_dice")
                if dice is not None and dice > best_dice:
                    best_dice = dice
                    best_epoch = epoch_int

            if best_epoch is None:
                epoch_list = [int(v) for v in info.keys()]
                if max_epoch_for_best is not None:
                    epoch_list = [e for e in epoch_list if e <= max_epoch_for_best]
                best_epoch = max(epoch_list) if epoch_list else max(int(v) for v in info.keys())

            chkpt_path = "%s/%s_epoch=%d.tar" % (phase_dir, net_name, best_epoch)
            return chkpt_path
        
        def get_last_chkpt_path(phase_dir, net_name):
            """Get the last checkpoint path."""
            stat_file_path = phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpt_path = "%s/%s_epoch=%d.tar" % (
                phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpt_path
        
        # Load network
        net_run_info = {}
        net_info_opt = last_phase_info["run_info"]
        for net_name, net_info in net_info_opt.items():
            net_desc = net_info["desc"]()
            
            # Load checkpoint
            chkpt_path = get_best_chkpt_path(log_dir, net_name)
            if not os.path.exists(chkpt_path):
                # Try last checkpoint as fallback
                chkpt_path = get_last_chkpt_path(log_dir, net_name)
            
            if os.path.exists(chkpt_path):
                net_state_dict = torch.load(chkpt_path)["desc"]
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                net_desc.load_state_dict(net_state_dict, strict=False)
                colored_word = colored(net_name, color="green", attrs=["bold"])
                print("Test Model `%s` loaded from: %s" % (colored_word, chkpt_path))
            else:
                print("Warning: No checkpoint found at %s" % chkpt_path)
            
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            net_desc.eval()
            
            net_run_info[net_name] = {
                "desc": net_desc,
            }
        
        # Get nr_types from model (need to load model first)
        nr_types = None
        if self.type_classification and net_run_info:
            first_net = list(net_run_info.values())[0]["desc"]
            if hasattr(first_net, 'module') and hasattr(first_net.module, 'nr_types'):
                nr_types = first_net.module.nr_types
        
        # Create test engine with correct nr_types
        def proc_test_output(a):
            return proc_valid_step_output(a, nr_types=nr_types)
        
        test_engine_opt = {
            "test": {
                "dataset": "",
                "nr_procs": 8,
                "run_step": valid_step,
                "reset_per_run": True,
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(proc_test_output),
                        LoggingEpochOutput(),
                    ],
                },
            }
        }
        
        runner = RunEngine(
            dataloader=test_loader,
            engine_name="test",
            run_step=test_engine_opt["test"]["run_step"],
            run_info=net_run_info,
            log_info=log_info,
        )
        
        # Attach callbacks
        callback_info = test_engine_opt["test"]["callbacks"]
        for event, callback_list in callback_info.items():
            for callback in callback_list:
                runner.add_event_handler(event, callback)
        
        runner.state.logging = self.logging
        runner.state.log_dir = test_log_dir
        
        # Run test (single epoch)
        print("Running test evaluation...")
        runner.run(1)
        
        print("\n" + "="*60)
        print("Test Evaluation Completed")
        print("Results saved to: %s" % test_log_dir)
        print("="*60 + "\n")
        return

    ####
    def run(self):
        """Define multi-stage run or cross-validation or whatever in here."""
        self.nr_gpus = torch.cuda.device_count()
        print('Detect #GPUS: %d' % self.nr_gpus)

        # When gradient CLI overrides are used, write results to a different subdir.
        base_log_dir = self.log_dir.rstrip("/")
        phase_list = self.model_config["phase_list"]

        if self.resume_path:
            base_log_dir = os.path.abspath(self.resume_path)
            print("Resume -> log dir: %s" % base_log_dir)
        else:
            grad_suffix = _grad_overrides_to_log_suffix(self.grad_overrides)
            if grad_suffix:
                base_log_dir = base_log_dir + "/" + grad_suffix
                print("Gradient overrides active -> log dir: %s" % base_log_dir)
            batch_suffix = _batch_overrides_to_log_suffix(self.batch_overrides)
            if batch_suffix:
                base_log_dir = base_log_dir + batch_suffix
                print("Batch overrides active -> log dir: %s" % base_log_dir)

            # Encode epoch schedule into log dir name, e.g. ep20_40 for
            # phase_list nr_epochs [20, 20] -> first=20, total=40.
            if phase_list:
                phase_epochs = [int(p.get("nr_epochs", 0)) for p in phase_list]
                first_ep = phase_epochs[0] if phase_epochs else 0
                total_ep = sum(phase_epochs)
                if first_ep > 0 and total_ep > 0:
                    base_log_dir = f"{base_log_dir}/ep{first_ep}_{total_ep}"
                    print("Epoch schedule -> log dir: %s" % base_log_dir)

        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = base_log_dir
            else:
                save_path = base_log_dir + "/%02d/" % (phase_idx)
            resume_log_dir = self.resume_path if (phase_idx == 0 and self.resume_path) else None
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path,
                resume_log_dir=resume_log_dir,
            )
            prev_save_path = save_path
        
        # Run test evaluation after all training phases
        final_save_path = prev_save_path if prev_save_path else base_log_dir
        self.run_test(final_save_path)


def _parse_batch_overrides(args):
    """Build batch_overrides dict from CLI (train, valid). Only includes keys that are set."""
    overrides = {}
    if args.get("--batch-train"):
        overrides["train"] = int(args["--batch-train"])
    if args.get("--batch-valid"):
        overrides["valid"] = int(args["--batch-valid"])
    return overrides


####
if __name__ == "__main__":
    args = docopt(__doc__, version="HoVer-Net v1.0")
    grad_overrides = _parse_grad_overrides(args)
    batch_overrides = _parse_batch_overrides(args)
    if grad_overrides:
        print("Gradient CLI overrides:", grad_overrides)
    if batch_overrides:
        print("Batch CLI overrides:", batch_overrides)
    resume_path = args.get("--resume") or None
    trainer = TrainManager(
        grad_overrides=grad_overrides,
        resume_path=resume_path,
        batch_overrides=batch_overrides,
    )

    if args["--view"]:
        if args["--view"] != "train" and args["--view"] != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args["--view"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
        trainer.run()
