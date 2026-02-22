"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>] [--grad-mode=<m>] [--grad-beta=<f>] [--grad-tau=<f>] [--grad-eps=<f>]
    [--grad-dir-beta=<f>] [--grad-dir-k=<f>] [--grad-dir-reject-max=<n>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help               Show this string.
  --version               Show version.
  --gpu=<id>              Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>           Visualise images after augmentation. Choose 'train' or 'valid'.
  --grad-mode=<m>         Gradient aggregation mode: sum|pcgrad|graddrop|pgrs|htdir (overrides config).
  --grad-beta=<f>         PGRS EMA beta, e.g. 0.999 (overrides config).
  --grad-tau=<f>          PGRS alignment threshold, e.g. 0.2 (overrides config).
  --grad-eps=<f>          Epsilon for gradient aggregation (overrides config).
  --grad-dir-beta=<f>     htdir concentration (>0), e.g. 5.0 (overrides config).
  --grad-dir-k=<f>        htdir tail heaviness (>0), e.g. 2.0 (overrides config).
  --grad-dir-reject-max=<n>  htdir max rejection steps, e.g. 64 (overrides config).
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
        dict: keys mode, beta, tau, eps, dir_beta, dir_k, dir_reject_max; only present if given on CLI.
    """
    overrides = {}
    if args.get("--grad-mode"):
        overrides["mode"] = args["--grad-mode"]
    if args.get("--grad-beta"):
        overrides["beta"] = float(args["--grad-beta"])
    if args.get("--grad-tau"):
        overrides["tau"] = float(args["--grad-tau"])
    if args.get("--grad-eps"):
        overrides["eps"] = float(args["--grad-eps"])
    if args.get("--grad-dir-beta"):
        overrides["dir_beta"] = float(args["--grad-dir-beta"])
    if args.get("--grad-dir-k"):
        overrides["dir_k"] = float(args["--grad-dir-k"])
    if args.get("--grad-dir-reject-max"):
        overrides["dir_reject_max"] = int(args["--grad-dir-reject-max"])
    return overrides


def _grad_overrides_to_log_suffix(grad_overrides):
    """
    Build a log subdir suffix from gradient overrides so results go to a different path.

    Args:
        grad_overrides: dict with keys mode, beta, tau, eps, dir_beta, dir_k, dir_reject_max (any subset).

    Returns:
        str: e.g. "grad_pgrs" or "grad_htdir_db5_dk2", or "" if empty.
    """
    if not grad_overrides:
        return ""
    parts = ["grad", grad_overrides.get("mode", "sum")]
    if "beta" in grad_overrides:
        parts.append("b%g" % grad_overrides["beta"])
    if "tau" in grad_overrides:
        parts.append("t%g" % grad_overrides["tau"])
    if "eps" in grad_overrides:
        parts.append("e%g" % grad_overrides["eps"])
    if "dir_beta" in grad_overrides:
        parts.append("db%g" % grad_overrides["dir_beta"])
    if "dir_k" in grad_overrides:
        parts.append("dk%g" % grad_overrides["dir_k"])
    if "dir_reject_max" in grad_overrides:
        parts.append("dr%d" % grad_overrides["dir_reject_max"])
    return "_".join(parts)


####
class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""

    def __init__(self, grad_overrides=None):
        """
        Args:
            grad_overrides: Optional dict to override GradAggConfig (mode, beta, tau, eps, dir_beta, dir_k, dir_reject_max).
        """
        super().__init__()
        self.grad_overrides = grad_overrides or {}
        return

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
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """Simply run the defined run_step of the related method once."""
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        ####
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            print(f"Runner name: {runner_name}")
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
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
            full_chkpt = None  # keep full .tar checkpoint to restore grad_agg later if present
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
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
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        full_chkpt = torch.load(pretrained_path)
                        net_state_dict = full_chkpt["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
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
            grad_cfg = {**extra_info["grad_cfg"]}
            for k in ("beta", "tau", "eps", "dir_beta", "dir_k", "dir_reject_max"):
                if k in self.grad_overrides:
                    grad_cfg[k] = self.grad_overrides[k]
            # Store effective config so logs/checkpoints record what is actually used
            extra_info["grad_mode"] = grad_mode
            extra_info["grad_cfg"] = grad_cfg
            def keeper(name, p):
                name = name[7:] if name.startswith("module.") else name
                return not name.startswith("decoder.")
            grad_agg = GradAggregator(
                net_desc,
                GradAggConfig(mode=grad_mode, **grad_cfg),
                param_filter=keeper,
                verbose=True,
            )
            if full_chkpt is not None and "grad_agg" in full_chkpt:
                grad_agg.load_state_dict(full_chkpt["grad_agg"], strict=False)
                print("[gradient] restored grad_agg state (Gpop) from checkpoint")
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
        # start the run loop
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run_test(self, log_dir):
        """
        Run test evaluation on the test dataset after training.
        
        Args:
            log_dir: Directory containing the trained model checkpoints
        """
        print("\n" + "="*60)
        print("Starting Test Evaluation")
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
        test_batch_size = last_phase_info["batch_size"].get("test", last_phase_info["batch_size"].get("valid", 8))
        test_loader = self._get_datagen(
            batch_size=test_batch_size,
            run_mode="test",
            target_gen=last_phase_info["target_info"]["gen"],
            nr_procs=8,  # Use same as valid
        )
        
        # Load the best model from training
        def get_best_chkpt_path(phase_dir, net_name):
            """Get the best checkpoint path based on validation metrics."""
            stat_file_path = phase_dir + "/stats.json"
            if not os.path.exists(stat_file_path):
                # Fallback to last checkpoint
                return get_last_chkpt_path(phase_dir, net_name)
            
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            
            # Find epoch with best validation dice
            best_epoch = None
            best_dice = -1
            for epoch_str, epoch_data in info.items():
                if "valid" in epoch_data:
                    valid_data = epoch_data["valid"]
                    if "np_dice" in valid_data:
                        dice = valid_data["np_dice"]
                        if dice > best_dice:
                            best_dice = dice
                            best_epoch = int(epoch_str)
            
            if best_epoch is None:
                # Fallback to last checkpoint
                epoch_list = [int(v) for v in info.keys()]
                best_epoch = max(epoch_list)
            
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
        grad_suffix = _grad_overrides_to_log_suffix(self.grad_overrides)
        if grad_suffix:
            base_log_dir = base_log_dir + "/" + grad_suffix
            print("Gradient overrides active -> log dir: %s" % base_log_dir)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = base_log_dir
            else:
                save_path = base_log_dir + "/%02d/" % (phase_idx)
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path
        
        # Run test evaluation after all training phases
        final_save_path = prev_save_path if prev_save_path else base_log_dir
        self.run_test(final_save_path)


####
if __name__ == "__main__":
    args = docopt(__doc__, version="HoVer-Net v1.0")
    grad_overrides = _parse_grad_overrides(args)
    if grad_overrides:
        print("Gradient CLI overrides:", grad_overrides)
    trainer = TrainManager(grad_overrides=grad_overrides)

    if args["--view"]:
        if args["--view"] != "train" and args["--view"] != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args["--view"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
        trainer.run()
