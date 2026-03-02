import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from collections import OrderedDict

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss

# ✅ add ShadowEMA
from parameter_wrapper.shadow_model import ShadowEMA


# -------------------------
# Shadow helpers
# -------------------------
def _shadow_group_fn(name: str) -> str:
    """
    Group function for ShadowEMA to split parameters into common/private.
    """
    name = name[7:] if name.startswith("module.") else name
    return "private" if name.startswith("decoder.") else "common"


def _ensure_shadow_ema_for_train(run_info, model):
    """
    TRAIN ONLY: create ShadowEMA if cfg provided.
    Valid should NOT create a new one.
    """
    extra = run_info["net"]["extra_info"]
    shadow_ema_cfg = extra.get("shadow_ema_cfg", None)
    if shadow_ema_cfg is None:
        return None

    shadow_ema = run_info["net"].get("shadow_ema", None)
    if shadow_ema is None:
        shadow_ema = ShadowEMA(model, group_fn=_shadow_group_fn, **shadow_ema_cfg)
        run_info["net"]["shadow_ema"] = shadow_ema
    return shadow_ema


def get_or_make_shadow_eval_fn(run_info, *, loss_opts, loss_func_dict):
    extra = run_info["net"]["extra_info"]
    fn = extra.get("_shadow_eval_fn_cached")
    if fn is not None:
        return fn

    def eval_fn(m, *, imgs, true_dict, true_np_onehot):
        was_training = bool(m.training)
        m.eval()
        with torch.no_grad():
            pred = m(imgs)
            pred = OrderedDict((k, v.permute(0,2,3,1).contiguous()) for k,v in pred.items())
            pred["np"] = F.softmax(pred["np"], dim=-1)
            if m.module.nr_types is not None and "tp" in pred:
                pred["tp"] = F.softmax(pred["tp"], dim=-1)

            loss_branch = {}
            for branch_name in pred.keys():
                L = 0.0
                for loss_name, loss_weight in loss_opts[branch_name].items():
                    loss_func = loss_func_dict[loss_name]
                    loss_args = [true_dict[branch_name], pred[branch_name]]
                    if loss_name == "msge":
                        loss_args.append(true_np_onehot[..., 1])
                    L = L + float(loss_weight) * loss_func(*loss_args)
                loss_branch[branch_name] = L
            total = sum(loss_branch.values())

        if was_training:
            m.train()

        out = {"overall_loss": float(total.detach().item())}
        for k, v in loss_branch.items():
            out[f"{k}_loss"] = float(v.detach().item())
        return out

    extra["_shadow_eval_fn_cached"] = eval_fn
    return eval_fn

# -------------------------
# Train
# -------------------------
def train_step(batch_data, run_info):
    """
    Original semantics unchanged.
    Adds:
      - ShadowEMA update AFTER optimizer.step()
      - logs shadow_* stats into result_dict["EMA"]
    """
    run_info, state_info = run_info

    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }

    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]
    grad_agg = run_info["net"]["grad_agg"]

    # ✅ ShadowEMA: TRAIN creates/owns it
    shadow_ema = _ensure_shadow_ema_for_train(run_info, model)
    prev_raw_snap = prev_shadow_snap = None
    if shadow_ema is not None:
        prev_raw_snap = shadow_ema.snapshot_raw(model)
        prev_shadow_snap = shadow_ema.snapshot_shadow()

    # ---- data to GPU (original)
    imgs = batch_data["img"].to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()
    true_np = batch_data["np_map"].to("cuda").type(torch.int64)
    true_hv = batch_data["hv_map"].to("cuda").type(torch.float32)

    true_np_onehot = F.one_hot(true_np, num_classes=2).type(torch.float32)
    true_dict = {"np": true_np_onehot, "hv": true_hv}

    if model.module.nr_types is not None:
        true_tp = torch.squeeze(batch_data["tp_map"]).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types).type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    model.train()

    pred_dict = model(imgs)
    pred_dict = OrderedDict((k, v.permute(0, 2, 3, 1).contiguous()) for k, v in pred_dict.items())
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.module.nr_types is not None and "tp" in pred_dict:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    # ---- loss (original)
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

    total_loss = sum(v for v in loss_branch.values())

    for branch_name, branch_loss in loss_branch.items():
        track_value(f"{branch_name}_loss", branch_loss.detach().item())

    optimizer.zero_grad(set_to_none=True)
    stats = grad_agg.backward(loss_branch, weights=None)
    optimizer.step()

    # ✅ ShadowEMA update AFTER step
    if shadow_ema is not None:
        shadow_ema.update(model)

    track_value("overall_loss", total_loss.detach().item())

    for key, val in stats.items():
        if torch.is_tensor(val) and val.numel() == 1:
            track_value("grad_" + key.replace("/", "_"), val.detach().item())
        elif isinstance(val, (int, float)):
            track_value("grad_" + key.replace("/", "_"), float(val))

    # ✅ ShadowEMA stats (optional eval)
    if shadow_ema is not None:
        step = state_info.get("step", None)
        eval_fn = get_or_make_shadow_eval_fn(
            run_info, loss_opts=loss_opts, loss_func_dict=loss_func_dict
        )
        eval_every = run_info["net"]["extra_info"].get("shadow_eval_every", 1)
        if eval_every < 1:
            raise ValueError("shadow_eval_every must be >= 1")
        shadow_stats = shadow_ema.on_step_end(
            model,
            prev_raw_snap=prev_raw_snap,
            prev_shadow_snap=prev_shadow_snap,
            do_gap=True,
            do_raw_step=True,
            do_shadow_step=True,
            eval_fn=eval_fn,
            eval_every=eval_every,
            step=step,
            eval_kwargs={"imgs": imgs, "true_dict": true_dict, "true_np_onehot": true_np_onehot}
        )

        for key, val in shadow_stats.items():
            if isinstance(val, (int, float)):
                track_value("train_shadow_" + key.replace("/", "_"), float(val))

    # ---- visualization protocol (original)
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs_viz = imgs[sample_indices].byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict_viz = dict(pred_dict)
    pred_dict_viz["np"] = pred_dict_viz["np"][..., 1:2]  # pos only
    pred_dict_viz = {k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict_viz.items()}

    true_dict_viz = dict(true_dict)
    true_dict_viz["np"] = true_np[..., None]
    true_dict_viz = {k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict_viz.items()}

    result_dict["raw"] = {
        "img": imgs_viz,
        "np": (true_dict_viz["np"], pred_dict_viz["np"]),
        "hv": (true_dict_viz["hv"], pred_dict_viz["hv"]),
    }
    return result_dict


# -------------------------
# Valid
# -------------------------
def valid_step(batch_data, run_info):
    """
    Original semantics unchanged.
    Adds:
      - optionally logs shadow_gap + eval/raw vs eval/shadow into result_dict["EMA"]
    """
    run_info, state_info = run_info
    model = run_info["net"]["desc"]
    model.eval()

    imgs = batch_data["img"]
    true_np = batch_data["np_map"].type(torch.int64)
    true_hv = batch_data["hv_map"].type(torch.float32)

    imgs_gpu = imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()

    with torch.no_grad():
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict((k, v.permute(0, 2, 3, 1).contiguous()) for k, v in pred_dict.items())
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        if model.module.nr_types is not None and "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False).type(torch.float32)
            pred_dict["tp"] = type_map

    result_dict = {
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_np.numpy(),
            "true_hv": true_hv.numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        },
        "EMA": {},  # ✅ for shadow stats
    }
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"].type(torch.int64)
        result_dict["raw"]["true_tp"] = true_tp.numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()

    # ✅ reuse TRAIN shadow_ema only; do not create in valid
    shadow_ema = run_info["net"].get("shadow_ema", None)
    if shadow_ema is not None:
        step = state_info.get("step", None)
        true_np_cuda = true_np.to("cuda")
        true_hv_cuda = true_hv.to("cuda")
        true_np_onehot = F.one_hot(true_np_cuda, num_classes=2).type(torch.float32)
        true_dict = {"np": true_np_onehot, "hv": true_hv_cuda}
        if model.module.nr_types is not None:
            true_tp_cuda = batch_data["tp_map"].to("cuda").type(torch.int64)
            true_tp_onehot = F.one_hot(true_tp_cuda, num_classes=model.module.nr_types).type(torch.float32)
            true_dict["tp"] = true_tp_onehot

        loss_opts = run_info["net"]["extra_info"]["loss"]
        loss_func_dict = {
            "bce": xentropy_loss,
            "dice": dice_loss,
            "mse": mse_loss,
            "msge": msge_loss,
        }

        eval_fn = get_or_make_shadow_eval_fn(
            run_info, loss_opts=loss_opts, loss_func_dict=loss_func_dict
        )
        eval_every = run_info["net"]["extra_info"].get("shadow_eval_every", 1)
        if eval_every < 1:
            raise ValueError("shadow_eval_every must be >= 1")

        shadow_stats = shadow_ema.on_step_end(
            model,
            prev_raw_snap=None,
            prev_shadow_snap=None,
            do_gap=True,
            do_raw_step=False,
            do_shadow_step=False,
            eval_fn=eval_fn,
            eval_every=eval_every,
            step=step,
            eval_kwargs={"imgs": imgs_gpu, "true_dict": true_dict, "true_np_onehot": true_np_onehot}
        )

        for key, val in shadow_stats.items():
            if isinstance(val, (int, float)):
                track_value("valid_shadow_" + key.replace("/", "_"), float(val))

    return result_dict

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