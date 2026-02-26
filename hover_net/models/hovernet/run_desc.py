import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss

from collections import OrderedDict

####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]
    grad_agg = run_info["net"]["grad_agg"]
    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()

    pred_dict = model(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.module.nr_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    # loss = 0
    loss_branch = {}
    loss_opts = run_info["net"]["extra_info"]["loss"]

    for branch_name in pred_dict.keys():
        L = 0.0
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
            term_loss = loss_func(*loss_args)
            L = L + float(loss_weight) * term_loss
        loss_branch[branch_name] = L

    # optional: log the true total weighted loss (same meaning as before)
    total_loss = sum(v for v in loss_branch.values())

    # log per-branch losses as separate training metrics
    for branch_name, branch_loss in loss_branch.items():
        track_value(f"{branch_name}_loss", branch_loss.detach().item())

    optimizer.zero_grad(set_to_none=True)
    # pgrs_lpf1 requires gpop_key (and optionally gpop_use_weight) from extra_info
    extra = run_info["net"]["extra_info"]
    gpop_key = extra.get("grad_gpop_key")
    gpop_use_weight = extra.get("grad_gpop_use_weight", True)
    stats = grad_agg.backward(
        loss_branch,
        weights=None,
        gpop_key=gpop_key,
        gpop_use_weight=gpop_use_weight,
    )
    optimizer.step()

    track_value("overall_loss", total_loss.detach().item())   # 真实总loss（全参数）

    # --- strategy metrics: op_* for HybridGradAggregator (OP-subspace), else GradAggregator (cos_raw_final, shrink_ratio) ---
    cos_raw = getattr(grad_agg, "op_cos_raw_final", None) or getattr(grad_agg, "cos_raw_final", None)
    if cos_raw is not None:
        track_value("cos_raw_final", cos_raw.item())
    shrink = getattr(grad_agg, "op_shrink_ratio", None) or getattr(grad_agg, "shrink_ratio", None)
    if shrink is not None:
        track_value("shrink_ratio", shrink.item())

    # (optional) PGRS-family stats (pgrs, pgrs_lambda, pgrs_lpf1, pgrs_stage, pgrs_common_gate)
    if hasattr(grad_agg, "last_stats") and grad_agg.last_stats:
        ls = grad_agg.last_stats
        # base pgrs metrics (if present)
        if "kept_frac" in ls:
            track_value("pgrs/kept_frac", ls["kept_frac"].item())
        if "rho_mean" in ls:
            track_value("pgrs/rho_mean",  ls["rho_mean"].item())
        if "rho_min" in ls:
            track_value("pgrs/rho_min",  ls["rho_min"].item())
        if "rho_max" in ls:
            track_value("pgrs/rho_max",  ls["rho_max"].item())
        if "drop_frac" in ls:
            track_value("pgrs/drop_frac", ls["drop_frac"].item())
        if "surgery_frac" in ls:
            track_value("pgrs/surgery_frac", ls["surgery_frac"].item())
        # fast-update stats (when use_fast_update=True)
        for key in ("fast_S", "fast_beta", "fast_e_norm", "fast_m_norm", "fast_v"):
            if key in ls:
                track_value(f"pgrs/{key}", ls[key].item())
        # pgrs_lambda specific: lambda, conf, mean_g_norm, Gpop_norm, cos_route_pc
        for key in ("lambda", "conf", "mean_g_norm", "Gpop_norm", "cos_route_pc"):
            if key in ls:
                track_value(f"pgrs/{key}", ls[key].item())
        # pgrs_lpf1 specific: gpop_key_idx, lpf_in_norm, Gpop_norm
        for key in ("gpop_key_idx", "lpf_in_norm", "Gpop_norm"):
            if key in ls:
                track_value(f"pgrs/{key}", ls[key].item())
        # pgrs_stage specific flags
        for key in ("late_stage", "loss_switch", "pgrs_stage_late_pcgrad"):
            if key in ls:
                track_value(f"pgrs_stage/{key}", ls[key].item())
        # pgrs_common_gate specific: common-subspace alignment and gating stats
        for key in (
            "rho_c_mean",
            "rho_c_min",
            "rho_c_max",
            "align_mean",
            "align_min",
            "align_neg_frac",
            "rho_c_neg_frac",
            "gpop_common_updated",
            "Gpop_common_norm",
            "common_frac_params",
            "rho_c_gate_thr",
            "cos_tt_mean",
            "cos_tt_min",
            "cos_tt_neg_frac",
            "cos_to_rawc_mean",
            "cos_to_rawc_min",
            "cos_to_rawc_neg_frac",
        ):
            if key in ls:
                track_value(f"pgrs_common_gate/{key}", ls[key].item())
        # pgrs_common_gate correlations: corr_dloss_align/<branch>, corr_dloss_rho/<branch>
        for name, val in ls.items():
            if name.startswith("corr_dloss_align/"):
                suffix = name.split("/", 1)[1].replace("/", "_")
                track_value(f"pgrs_common_gate/corr_dloss_align_{suffix}", val.item())
            if name.startswith("corr_dloss_rho/"):
                suffix = name.split("/", 1)[1].replace("/", "_")
                track_value(f"pgrs_common_gate/corr_dloss_rho_{suffix}", val.item())

    # (optional) ht_iters/ht_acc only exist for htdir mode
    if hasattr(grad_agg, "last_stats") and ("ht_iters" in grad_agg.last_stats):
        track_value("htdir/ht_iters", grad_agg.last_stats["ht_iters"].item())
        if "ht_acc" in grad_agg.last_stats:
            track_value("htdir/ht_acc", grad_agg.last_stats["ht_acc"].item())

    # torch.set_printoptions(precision=10)

    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict["np"] = pred_dict["np"][..., 1:2]  # return pos only
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict["np"] = true_np[..., None]
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        "np": (true_dict["np"], pred_dict["np"]),
        "hv": (true_dict["hv"], pred_dict["hv"]),
    }

    return result_dict


####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    # Keep batch dimension to ensure consistent accumulation across steps
    true_np = true_np.type(torch.int64)
    true_hv = true_hv.type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = true_tp.type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        if model.module.nr_types is not None:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].numpy(),
            "true_hv": true_dict["hv"].numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }
    if model.module.nr_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
    return result_dict


####
def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.cpu().numpy()


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    true_np, pred_np = raw_data["np"]
    true_hv, pred_hv = raw_data["hv"]
    if nr_types is not None:
        true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["prob_np"]
    true_np = raw_data["true_np"]
    for idx in range(len(true_np)):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value("np_acc", acc_np, "scalar")
    track_value("np_dice", dice_np, "scalar")

    # * TP statistic
    if nr_types is not None:
        pred_tp = raw_data["pred_tp"]
        true_tp = raw_data["true_tp"]
        for type_id in range(0, nr_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_np"])):
                patch_pred_tp = pred_tp[idx]
                patch_true_tp = true_tp[idx]
                inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                over_inter += inter
                over_total += total
            dice_tp = 2 * over_inter / (over_total + 1.0e-8)
            track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # * HV regression statistic
    pred_hv = raw_data["pred_hv"]
    true_hv = raw_data["true_hv"]

    over_squared_error = 0
    for idx in range(len(raw_data["true_np"])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value("hv_mse", mse, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_hv = np.array([true_hv[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    true_np = true_np[..., None] # (B, 256, 256, 1)
    prob_np = prob_np[..., None] # (B, 256, 256, 1)
    viz_raw_data = {
        "img": imgs, # (B, 256, 256, 3)
        "np": (true_np, prob_np), # (B, 256, 256, 1)
        "hv": (true_hv, pred_hv)
    }

    if nr_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["tp"] = (true_tp, pred_tp)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict