from collections import deque
import os
import random

import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np

from bioplnn.models import (
    SpatiallyEmbeddedClassifier,
    SpatiallyEmbeddedRNN,
    SpatiallyEmbeddedAreaConfig,
)
from hyperparameters import get_default_hp
from task import WCST, get_default_hp_wcst


# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


# Get hyperparameters
hp, optimizer_name, loss_fnc = get_default_hp()


# Define the model  
area_configs_feedback_model = [
    SpatiallyEmbeddedAreaConfig(  # Sensory module — choice output (first n_output channels)
        num_neuron_types=5,
        num_neuron_subtypes=np.array([70, 140, 10, 10, 10]),  # soma, dendrite, PV, SST, VIP
        neuron_type_class=np.array(['excitatory', 'excitatory',
                                    'inhibitory', 'inhibitory', 'inhibitory']),
        neuron_type_nonlinearity=["relu", "tanh", "relu", "relu", "relu"],
        tau_mode='subtype',
        # state_clip=20.0,
        default_neuron_state_init_fn='rand',
        inter_neuron_type_connectivity=np.array([
            [0, 1, 1, 0, 0, 0],  # input
            [0, 1, 1, 1, 1, 0],  # feedback (from PFC)
            [1, 0, 1, 1, 0, 1],  # soma
            [1, 0, 0, 0, 0, 0],  # dendrite
            [1, 0, 1, 0, 0, 0],  # PV
            [0, 1, 1, 0, 1, 0],  # SST
            [0, 0, 0, 1, 0, 0],  # VIP
        ]),  # columns: soma, dendrite, PV, SST, VIP, output
        feedback_channels=70,
        in_size=[1, 1],
        in_channels=hp['n_input'],  # sensory-only channels
        out_channels=70,  # rich repr; first n_output channels read as choice
        inter_neuron_type_spatial_extents=(1, 1),
    ),
    SpatiallyEmbeddedAreaConfig(  # PFC module — rule output
        num_neuron_types=5,
        num_neuron_subtypes=np.array([70, 140, 10, 10, 10]),
        neuron_type_class=np.array(['excitatory', 'excitatory',
                                    'inhibitory', 'inhibitory', 'inhibitory']),
        neuron_type_nonlinearity=["relu", "tanh", "relu", "relu", "relu"],
        tau_mode='subtype',
        # state_clip=20.0,
        default_neuron_state_init_fn='rand',
        inter_neuron_type_connectivity=np.array([
            [0, 1, 1, 0, 0, 0],  # input (= sensory output)
            [1, 0, 1, 1, 0, 1],  # soma
            [1, 0, 0, 0, 0, 0],  # dendrite
            [1, 0, 1, 0, 0, 0],  # PV
            [0, 1, 1, 0, 1, 0],  # SST
            [0, 0, 0, 1, 0, 0],  # VIP
        ]),
        in_size=[1, 1],
        in_channels=70 + hp['n_input'] + 2 + 3,  # receives sensory output (+ history features now routed via this stream)
        out_channels=hp['n_output_rule'],  # 2 (rule only)
        inter_neuron_type_spatial_extents=(1, 1),
    )
]

model = SpatiallyEmbeddedRNN(
    num_areas=2,
    area_configs=area_configs_feedback_model,
    batch_first=False,
    inter_area_feedback_connectivity=np.array(
        [
            [0, 0],
            [1, 0],
        ]
    ),
    # inter_area_feedback_connectivity = np.array([[0, 1],[1, 0]]
)


def _build_spatial_wcst_model(
    first_in_channels: int, split_output_areas: bool = True
) -> SpatiallyEmbeddedRNN:
    """Build SpatiallyEmbeddedRNN with given packed input channels (37 or 39).

    If split_output_areas is False, the last area's out_channels is expanded to
    n_output + n_output_rule so both readouts come from the same area.
    """
    c0 = area_configs_feedback_model[0]
    c1 = area_configs_feedback_model[1]
    first_config = SpatiallyEmbeddedAreaConfig(
        num_neuron_types=c0.num_neuron_types,
        num_neuron_subtypes=c0.num_neuron_subtypes,
        neuron_type_class=c0.neuron_type_class,
        neuron_type_nonlinearity=c0.neuron_type_nonlinearity,
        tau_mode=c0.tau_mode,
        default_neuron_state_init_fn=c0.default_neuron_state_init_fn,
        inter_neuron_type_connectivity=c0.inter_neuron_type_connectivity,
        feedback_channels=c0.feedback_channels,
        in_size=c0.in_size,
        in_channels=c0.in_channels,
        out_channels=c0.out_channels,
        inter_neuron_type_spatial_extents=c0.inter_neuron_type_spatial_extents,
    )
    base_total_in_channels = hp["n_input"] * 2 + 2 + 3
    extra_rule_channels = max(0, int(first_in_channels) - int(base_total_in_channels))
    last_out_channels = (
        c1.out_channels
        if split_output_areas
        else hp["n_output"] + hp["n_output_rule"]
    )
    last_config = SpatiallyEmbeddedAreaConfig(
        num_neuron_types=c1.num_neuron_types,
        num_neuron_subtypes=c1.num_neuron_subtypes,
        neuron_type_class=c1.neuron_type_class,
        neuron_type_nonlinearity=c1.neuron_type_nonlinearity,
        tau_mode=c1.tau_mode,
        default_neuron_state_init_fn=c1.default_neuron_state_init_fn,
        inter_neuron_type_connectivity=c1.inter_neuron_type_connectivity,
        in_size=c1.in_size,
        in_channels=int(c1.in_channels) + extra_rule_channels,
        out_channels=last_out_channels,
        inter_neuron_type_spatial_extents=c1.inter_neuron_type_spatial_extents,
    )
    return SpatiallyEmbeddedRNN(
        num_areas=2,
        area_configs=[first_config, last_config],
        batch_first=False,
        inter_area_feedback_connectivity=np.array([[0, 0], [1, 0]]),
    )


def _pack_inputs_spatial(
    x_curr: torch.Tensor,
    I_prev_stim: torch.Tensor,
    I_prev_rew: torch.Tensor,
    I_prev_choice: torch.Tensor,
    rule_cue: torch.Tensor | None,
    n_channels: int,
    spatial_size: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Build packed external input for SpatiallyEmbeddedRNN.

    Channel layout: [sensory input | prev_stim | prev_rew | prev_choice | optional rule cue].
    The flat input vector is tiled across the spatial dimensions.
    """
    t, b, n_in = x_curr.shape
    h, w = spatial_size
    n_base = n_in * 2 + 2 + 3
    x_full = torch.zeros(
        t, b, n_channels, h, w, device=x_curr.device, dtype=x_curr.dtype
    )
    x_full[:, :, :n_in, :, :] = x_curr.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
    if n_channels >= n_base:
        x_full[:, :, n_in : 2 * n_in, :, :] = I_prev_stim.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
        x_full[:, :, 2 * n_in : 2 * n_in + 2, :, :] = I_prev_rew.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
        x_full[:, :, 2 * n_in + 2 : n_base, :, :] = I_prev_choice.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
    if rule_cue is not None and n_channels >= 2:
        rc = rule_cue.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(t, -1, -1, h, w)
        x_full[:, :, -rule_cue.shape[-1] :, :, :] = rc
    return x_full


class SimpleWCSTRNN(torch.nn.Module):
    """Simple RNN for testing the WCST training pipeline.
    Input: (T, B, input_size), Output: (T, B, n_output + n_output_rule).

    Args:
        num_layers: number of stacked RNN layers (default 1). When > 1, a
            dropout of 0.1 is applied between layers during training.
        split_layers: if True and num_layers==2, treat as sensory (layer 0) + PFC (layer 1).
            Layer 0 receives only sensory input (input_size = n_input); layer 1 receives
            hidden from layer 0 plus trial history (prev_stim, prev_reward, prev_choice, rule_cue).
            Requires history_input_size = n_input + 2 + 3 + (0 or 2 for rule).
        history_input_size: used only when split_layers=True; size of the trial-history
            vector (prev_stim + prev_rew + prev_choice + optional rule_cue) fed to PFC layer.
        split_output: used only when split_layers=True. If True (default), choice logits
            come from the sensory layer and rule logits from the PFC layer. If False,
            both choice and rule come from a single readout on the PFC layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        n_output: int = 3,
        n_output_rule: int = 2,
        num_layers: int = 1,
        split_layers: bool = False,
        history_input_size: int | None = None,
        split_output: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_output = n_output
        self.n_output_rule = n_output_rule
        self.num_layers = num_layers
        self.split_layers = bool(split_layers)
        self.history_input_size = history_input_size  # only used when split_layers
        self.split_output = bool(split_output) if split_layers else False

        if self.split_layers:
            if num_layers != 2:
                raise ValueError("split_layers=True requires num_layers=2")
            if history_input_size is None:
                raise ValueError("split_layers=True requires history_input_size")
            # Sensory layer: only current stimulus
            self.rnn_sensory = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=False,
                nonlinearity="tanh",
            )
            # PFC layer: sensory hidden + prev_stim, prev_rew, prev_choice, rule_cue
            self.rnn_pfc = torch.nn.RNN(
                input_size=hidden_size + history_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=False,
                nonlinearity="tanh",
            )
            if self.split_output:
                self.readout_choice = torch.nn.Linear(hidden_size, n_output)
                self.readout_rule = torch.nn.Linear(hidden_size, n_output_rule)
                self.readout = None
            else:
                self.readout_choice = None
                self.readout_rule = None
                self.readout = torch.nn.Linear(hidden_size, n_output + n_output_rule)
        else:
            self.rnn_sensory = None
            self.rnn_pfc = None
            self.rnn = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,
                nonlinearity="tanh",
                dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.readout = torch.nn.Linear(hidden_size, n_output + n_output_rule)

    def forward(
        self,
        x: torch.Tensor,
        x_history: torch.Tensor | None = None,
        h0: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        if self.split_layers:
            if x_history is None:
                raise ValueError("split_layers=True requires x_history (T, B, history_input_size)")
            # x: (T, B, input_size), x_history: (T, B, history_input_size)
            h_sensory_prev, h_pfc_prev = (h0 if h0 is not None else (None, None))
            out_sensory, h_sensory = self.rnn_sensory(x, h_sensory_prev)
            x_pfc = torch.cat([out_sensory, x_history], dim=-1)
            out_pfc, h_pfc = self.rnn_pfc(x_pfc, h_pfc_prev)
            if self.split_output:
                y = self.readout_choice(out_sensory)
                y_rule = self.readout_rule(out_pfc)
            else:
                logits = self.readout(out_pfc)
                y = logits[..., : self.n_output]
                y_rule = logits[..., self.n_output :]
            return y, y_rule, (h_sensory, h_pfc)
        else:
            # x: (T, B, input_size), h0: (num_layers, B, hidden_size) or None
            out, hn = self.rnn(x, h0)
            logits = self.readout(out)
            y = logits[..., : self.n_output]
            y_rule = logits[..., self.n_output :]
            return y, y_rule, hn


def _build_optimizer(model: torch.nn.Module, hp: dict) -> torch.optim.Optimizer:
    """Instantiate the optimizer based on hyperparameters."""
    opt_name = str(hp.get("optimizer", "adam")).lower()
    lr = hp.get("learning_rate", 1e-3)

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {opt_name}")


def train_wcst(
    num_blocks: int = 5,
    num_batches_per_block: int = 2000,
    print_every: int = 50,
    trials_per_rollout: int | None = None,
    rule_loss_weight: float = 0.1,
    loss_window: int = 100,
    return_loss_history: bool = False,
    fixed_rule: bool = False,
    use_rule_cue: bool = False,
    spatial_model: SpatiallyEmbeddedRNN | None = None,
    curriculum: bool = False,
    curriculum_phase1_batches: int = 2000,
    curriculum_phase2_lr_scale: float = 0.7,
    curriculum_fade_batches: int | None = None,
    curriculum_fade_per_trial: bool = False,
    curriculum_fade_max_batches: int | None = None,
    curriculum_fade_perf_threshold: float = 0.8,
    curriculum_fade_rollback_threshold: float | None = None,
    curriculum_fade_min_consecutive_good: int = 1,
    curriculum_fade_start: float = 0.0,
    postfade_lr_scale: float = 0.5,
    postfade_reset_optimizer: bool = True,
    save_path: str | None = None,
    curriculum_pretrain_save_path: str | None = None,
    curriculum_pretrain_load_path: str | None = None,
    rule_cue_noise: float = 0.0,
    split_output_areas: bool = True,
    plot_every: int | None = None,
    save_every: int | None = 100,
    debug_nonfinite: bool = False,
) -> SpatiallyEmbeddedRNN:
    """Train the spatially embedded RNN on the WCST task with the same procedure and options as train_wcst_simple_rnn.

    Uses trial-history rollout, block structure with rule switches, optional curriculum
    (phase 1 with rule cue, fade, phase 2 cue off), performance-based fade advance/rollback,
    pretrain save/load, and rule_cue_noise. See train_wcst_simple_rnn docstring for details.

    curriculum_fade_start: initial fade progress when entering the fade phase (default 0.0).
    E.g. 0.8 means start the fade with 80% cue-drop probability; useful to resume or shorten the fade.

    split_output_areas: if True (default), read choice from area 0 and rule from area 1.
    If False, read both choice and rule from the last area (requires last area out_channels >= n_output + n_output_rule).

    plot_every: when not None and save_path is set, periodically re-save the loss figure
    (total/resp/rule and grad_norm) to the same file as the final loss plot, every
    this many global batches.

    debug_nonfinite: if True, raise immediately on the first non-finite loss or gradient
    norm so torch.autograd anomaly detection can provide a useful traceback.
    """
    base_in_channels = hp["n_input"] * 2 + 2 + 3
    need_rule_channels = curriculum or use_rule_cue
    first_in_channels = base_in_channels + 2 if need_rule_channels else base_in_channels

    curriculum_skip_phase1 = False
    _custom_model = spatial_model is not None
    if _custom_model:
        spatial_model = spatial_model.to(device)  # type: ignore[union-attr]
        assert spatial_model is not None
        first_in_channels = int(
            getattr(spatial_model, "input_channels", spatial_model.areas[0].in_channels)
        )
        if need_rule_channels and first_in_channels < base_in_channels + 2:
            raise ValueError(
                f"Provided model has in_channels={first_in_channels}, but "
                f"curriculum=True or use_rule_cue=True requires at least "
                f"{base_in_channels + 2} (base {base_in_channels} + 2 rule cue channels). "
                f"Rebuild the model with in_channels={base_in_channels + 2}."
            )
        print(f"[SpatialRNN] Using provided model (in_channels={first_in_channels})")
    elif (
        curriculum
        and curriculum_pretrain_load_path
        and os.path.isfile(curriculum_pretrain_load_path)
    ):
        spatial_model = load_wcst_spatial(curriculum_pretrain_load_path, device)
        curriculum_skip_phase1 = True
        print(
            f"[SpatialRNN] Loaded phase-1 pretrain from {curriculum_pretrain_load_path}, skipping phase 1"
        )
    elif need_rule_channels:
        spatial_model = _build_spatial_wcst_model(first_in_channels, split_output_areas)
        spatial_model.to(device)
    else:
        spatial_model = model
        spatial_model.to(device)

    assert spatial_model is not None, "No model provided or built"
    _spatial_size = tuple(spatial_model.areas[0].in_size)
    _sensory_in = int(getattr(spatial_model.areas[0], "in_channels"))
    _area_external = getattr(spatial_model, "area_external_in_channels", None)
    _pfc_extra = int(_area_external[1]) if _area_external is not None and len(_area_external) > 1 else max(0, int(getattr(spatial_model.areas[1], "in_channels")) - int(getattr(spatial_model.areas[0], "out_channels")))
    _total_in = int(getattr(spatial_model, "input_channels", _sensory_in + _pfc_extra))
    print(
        f"[SpatialRNN] Channel routing: sensory_in={_sensory_in}, "
        f"pfc_extra={_pfc_extra}, total_input_channels={_total_in}"
    )
    optimizer = _build_optimizer(spatial_model, hp)
    criterion = loss_fnc

    hp_wcst = get_default_hp_wcst()
    rule_list = ["color", "shape"]
    n_features_per_rule = 2
    n_test_cards = 3
    n_out = hp["n_output"]
    n_out_rule = hp["n_output_rule"]
    dt = hp["dt"]
    hist_start_ts = int(hp_wcst["trial_history_start"] // dt)
    hist_end_ts = int(hp_wcst["trial_history_end"] // dt)
    resp_start_ts = int(hp_wcst["resp_start"] // dt)
    resp_end_ts = int(hp_wcst["resp_end"] // dt)
    rule_start_ts = int(hp_wcst["trial_start"] // dt)
    rule_end_ts = int(hp_wcst["trial_end"] // dt)

    def _compute_trial_history(
        last_rew: torch.Tensor | None,
        prev_stim_full: torch.Tensor | None,
        prev_choice_vec: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_steps = int((hp_wcst["trial_end"] - hp_wcst["trial_start"]) // dt)
        I_prev_rew = torch.zeros(n_steps, batch_size, 2, device=device)
        I_prev_stim = torch.zeros(n_steps, batch_size, hp["n_input"], device=device)
        I_prev_choice = torch.zeros(n_steps, batch_size, hp["n_output"], device=device)
        input_start = int(hp_wcst["trial_history_start"] // dt)
        input_end = int(hp_wcst["trial_history_end"] // dt)
        stim_start_ts = int(hp_wcst["test_cards_on"] // dt)
        stim_end_ts = int(hp_wcst["test_cards_off"] // dt)
        if last_rew is not None:
            correct = last_rew.float()
            incorrect = 1.0 - correct
            I_prev_rew[input_start:input_end, :, 0] = correct.unsqueeze(0)
            I_prev_rew[input_start:input_end, :, 1] = incorrect.unsqueeze(0)
        if prev_stim_full is not None:
            stim_mean = prev_stim_full[stim_start_ts:stim_end_ts].mean(dim=0)
            I_prev_stim[input_start:input_end, :, :] = stim_mean.unsqueeze(0)
        if prev_choice_vec is not None:
            I_prev_choice[input_start:input_end, :, :] = prev_choice_vec.unsqueeze(0)
        return I_prev_rew, I_prev_stim, I_prev_choice

    recent_total = deque(maxlen=max(1, int(loss_window)))
    recent_resp = deque(maxlen=max(1, int(loss_window)))
    recent_rule = deque(maxlen=max(1, int(loss_window)))
    loss_hist: list[dict[str, float]] = []
    loss_hist_saved_len = 0
    phase2_lr_scaled = False
    curriculum_fade_progress = max(0.0, min(1.0, curriculum_fade_start))
    curriculum_fade_batch_count = 0
    curriculum_fade_consecutive_good = 0
    curriculum_postfade_saved = False
    last_total_batch = -1

    def _derive_postfade_path(base_path: str) -> str:
        base, ext = os.path.splitext(base_path)
        return f"{base}_postfade{ext}" if ext else f"{base_path}_postfade"

    def _derive_loss_history_path(base_path: str) -> str:
        base, ext = os.path.splitext(base_path)
        return f"{base}_loss_history.json" if ext else f"{base_path}_loss_history.json"

    def _derive_loss_plot_path(base_path: str) -> str:
        base, ext = os.path.splitext(base_path)
        return f"{base}_loss.png" if ext else f"{base_path}_loss.png"

    def _append_loss_history(base_path: str) -> None:
        """Append only new loss_hist entries from this run to existing history on disk."""
        nonlocal loss_hist_saved_len
        import json
        import os as _os

        loss_history_path = _derive_loss_history_path(base_path)
        existing: list[dict] = []
        if _os.path.isfile(loss_history_path):
            try:
                with open(loss_history_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = loaded
            except Exception:
                existing = []
        new_entries = loss_hist[loss_hist_saved_len:]
        if not new_entries:
            return
        combined = existing + new_entries
        with open(loss_history_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        loss_hist_saved_len = len(loss_hist)

    # Periodic loss plot saving: overwrite the same loss.png during training.
    _plot_every_eff: int | None = None
    if plot_every is not None and save_path is not None:
        try:
            _plot_every_eff = max(1, int(plot_every))
        except Exception:
            _plot_every_eff = None

    # Periodic checkpoint saving during training (default every 100 batches).
    _save_every_eff: int | None = None
    if save_every is not None:
        try:
            _save_every_eff = max(1, int(save_every))
        except Exception:
            _save_every_eff = None

    def _save_loss_plot_live(base_path: str) -> None:
        """Save or update the loss plot alongside the checkpoint during training."""
        if _plot_every_eff is None or not loss_hist:
            return
        try:
            import importlib
            plt = importlib.import_module("matplotlib.pyplot")
        except Exception:
            return
        loss_plot_path = _derive_loss_plot_path(base_path)
        x_axis = list(range(len(loss_hist)))
        total = [d.get("total", float("nan")) for d in loss_hist]
        resp = [d.get("resp", float("nan")) for d in loss_hist]
        rule = [d.get("rule", float("nan")) for d in loss_hist]
        grad = [d.get("grad_norm", float("nan")) for d in loss_hist]

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, total, label="total", linewidth=1.5)
        plt.plot(x_axis, resp, label="resp", linewidth=1.0, alpha=0.9)
        plt.plot(x_axis, rule, label="rule", linewidth=1.0, alpha=0.9)
        plt.xlabel("update")
        plt.ylabel("loss")
        plt.title("WCST SpatialRNN training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=200)
        plt.close()

    for block in range(num_blocks):
        for batch_idx in range(num_batches_per_block):
            bsz = int(hp["batch_size"])
            total_batch = block * num_batches_per_block + batch_idx
            if curriculum_skip_phase1 and total_batch < curriculum_phase1_batches:
                continue
            last_total_batch = total_batch

            fade_batches = curriculum_fade_batches if curriculum_fade_batches is not None else 0
            in_phase1 = (
                curriculum
                and total_batch < curriculum_phase1_batches
                and not curriculum_skip_phase1
            )
            in_fade = (
                curriculum
                and fade_batches > 0
                and total_batch >= curriculum_phase1_batches
                and curriculum_fade_progress < 1.0
            )
            fade_fraction = curriculum_fade_progress if in_fade else 0.0
            in_phase2_cue_off = (
                curriculum
                and total_batch >= curriculum_phase1_batches
                and (fade_batches == 0 or curriculum_fade_progress >= 1.0)
            )
            if curriculum and in_phase1:
                use_cue_this_batch = True
            elif curriculum and in_fade and not curriculum_fade_per_trial:
                use_cue_this_batch = random.random() > fade_fraction
            elif curriculum and in_phase2_cue_off:
                use_cue_this_batch = False
            else:
                use_cue_this_batch = use_rule_cue

            if (
                curriculum
                and not phase2_lr_scaled
                and total_batch >= curriculum_phase1_batches
            ):
                phase2_lr_scaled = True
                for g in optimizer.param_groups:
                    g["lr"] *= curriculum_phase2_lr_scale
                msg = (
                    f"[SpatialRNN] Curriculum: phase 2 (rule cue off), LR scaled by {curriculum_phase2_lr_scale}"
                    if fade_batches <= 0
                    else (
                        f"[SpatialRNN] Curriculum: cue fade over {fade_batches} batches"
                        + (" (per-trial)" if curriculum_fade_per_trial else "")
                        + f", LR scaled by {curriculum_phase2_lr_scale}"
                    )
                )
                print(msg)

            block_len = int(hp.get("block_len", 20))
            n_switches = int(hp.get("n_switches", 3))
            if fixed_rule:
                switches = set()
                current_rule_idx = 0
            else:
                if block_len > 1 and n_switches > 0:
                    n_switches_eff = min(max(n_switches, 0), block_len - 1)
                    switches = set(random.sample(range(block_len - 1), k=n_switches_eff))
                else:
                    switches = set()
                current_rule_idx = 0

            last_rew_vec: torch.Tensor | None = None
            prev_stim_full: torch.Tensor | None = None
            prev_choice_vec: torch.Tensor | None = None

            # Carry SpatiallyEmbeddedRNN hidden states across trials in rollout
            carry_output_state: list | None = None
            carry_neuron_state: list | None = None
            carry_feedback_state: list | None = None

            total_loss = torch.tensor(0.0, device=device)
            total_loss_resp = torch.tensor(0.0, device=device)
            total_loss_rule = torch.tensor(0.0, device=device)
            last_choice_acc = float("nan")
            last_rule_acc = float("nan")
            sum_choice_acc = 0.0
            sum_rule_acc = 0.0
            n_acc = 0

            rollout_len = (
                int(trials_per_rollout) if trials_per_rollout is not None else block_len
            )

            for tr in range(rollout_len):
                I_prev_rew, I_prev_stim, I_prev_choice = _compute_trial_history(
                    last_rew=last_rew_vec,
                    prev_stim_full=prev_stim_full,
                    prev_choice_vec=prev_choice_vec,
                    batch_size=bsz,
                    device=device,
                )
                rule = rule_list[current_rule_idx]
                wcst = WCST(
                    hp=hp,
                    hp_wcst=hp_wcst,
                    rule=rule,
                    rule_list=rule_list,
                    n_features_per_rule=n_features_per_rule,
                    n_test_cards=n_test_cards,
                )
                x, x_rule, yhat, yhat_rule, task_data = wcst.make_task_batch(batch_size=bsz)
                x = x.to(device)
                yhat = yhat.to(device)
                yhat_rule = yhat_rule.to(device)

                if curriculum_fade_per_trial and curriculum:
                    if in_phase1:
                        use_cue_this_trial = True
                    elif in_fade:
                        use_cue_this_trial = random.random() > fade_fraction
                    else:
                        use_cue_this_trial = False
                else:
                    use_cue_this_trial = use_cue_this_batch

                rule_cue = None
                if use_cue_this_trial:
                    drop_cue = rule_cue_noise > 0 and random.random() < rule_cue_noise
                    if not drop_cue:
                        rule_idx_tensor = torch.full(
                            (bsz,), current_rule_idx, device=device, dtype=torch.long
                        )
                        rule_cue = torch.nn.functional.one_hot(
                            rule_idx_tensor, num_classes=len(rule_list)
                        ).float()

                x_full = _pack_inputs_spatial(
                    x, I_prev_stim, I_prev_rew, I_prev_choice, rule_cue, first_in_channels,
                    spatial_size=_spatial_size,
                )
                output_states, neuron_states, feedback_states = spatial_model(
                    x_full,
                    output_state0=carry_output_state,
                    neuron_state0=carry_neuron_state,
                    feedback_state0=carry_feedback_state,
                )

                # Extract final-timestep states to carry into next trial
                with torch.no_grad():
                    carry_output_state = [
                        out[-1].detach() for out in output_states
                    ]
                    carry_neuron_state = [
                        [ct[-1].detach() for ct in area_ns]
                        for area_ns in neuron_states
                    ]
                    carry_feedback_state = [
                        fb[-1].detach() if fb is not None else None
                        for fb in feedback_states
                    ]

                if split_output_areas:
                    sensory_out = output_states[0].mean(dim=(-2, -1))
                    pfc_out = output_states[-1].mean(dim=(-2, -1))
                    y = sensory_out[:, :, :n_out]
                    y_rule = pfc_out[:, :, :n_out_rule]
                else:
                    last_out = output_states[-1].mean(dim=(-2, -1))
                    y = last_out[:, :, :n_out]
                    y_rule = last_out[:, :, n_out:]

                loss_resp = criterion(y, yhat)
                loss_rule = (
                    criterion(y_rule, yhat_rule) if hp.get("train_rule", True) else None
                )
                loss = loss_resp + (
                    rule_loss_weight * loss_rule if loss_rule is not None else 0.0
                )

                # L2 weight regularization on parameters
                reg_lambda = float(hp.get("l2_weight", 0.0))
                if reg_lambda > 0.0:
                    reg = torch.tensor(0.0, device=device)
                    for p in spatial_model.parameters():
                        reg = reg + p.pow(2).mean()
                    loss = loss + reg_lambda * reg

                # L2 regularization on hidden activity (l2_h), across all areas / neuron types.
                act_lambda = float(hp.get("l2_h", 0.0))
                if act_lambda > 0.0:
                    act_reg = torch.tensor(0.0, device=device)
                    for area_ns in neuron_states:
                        for ct_state in area_ns:
                            act_reg = act_reg + ct_state.pow(2).mean()
                    loss = loss + act_lambda * act_reg

                total_loss = total_loss + loss
                total_loss_resp = total_loss_resp + loss_resp
                if loss_rule is not None:
                    total_loss_rule = total_loss_rule + loss_rule

                with torch.no_grad():
                    prev_stim_full = x.detach()
                    choice_prob = y[resp_start_ts:resp_end_ts].mean(dim=0)
                    choice_idx = choice_prob.argmax(dim=-1)
                    prev_choice_vec = torch.nn.functional.one_hot(
                        choice_idx, num_classes=hp["n_output"]
                    ).float()
                    target_prob = yhat[resp_start_ts:resp_end_ts].mean(dim=0)
                    target_idx = target_prob.argmax(dim=-1)
                    correct = choice_idx == target_idx
                    last_rew_vec = correct
                    last_choice_acc = correct.float().mean().item()
                    rule_pred = y_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(dim=-1)
                    rule_tgt = yhat_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(dim=-1)
                    last_rule_acc = (rule_pred == rule_tgt).float().mean().item()
                    sum_choice_acc += last_choice_acc
                    sum_rule_acc += last_rule_acc
                    n_acc += 1

                if tr in switches:
                    current_rule_idx = 1 - current_rule_idx

            roll = float(max(1, rollout_len))
            loss = total_loss / roll
            loss_resp = total_loss_resp / roll
            loss_rule = (
                (total_loss_rule / roll) if hp.get("train_rule", True) else None
            )

            # Optional detailed debug on the first non-finite batch to localize the source.
            if debug_nonfinite and not torch.isfinite(loss):
                print(
                    f"[SpatialRNN][Debug] Inspecting tensors at block={block}, "
                    f"batch_idx={batch_idx}, total_batch={total_batch}, "
                    f"fade={curriculum_fade_progress:.3f}"
                )
                try:
                    print(
                        "  y finite:", bool(torch.isfinite(y).all()),
                        "max|y|:", float(y.abs().max().detach().cpu())
                    )
                    print(
                        "  y_rule finite:", bool(torch.isfinite(y_rule).all()),
                        "max|y_rule|:", float(y_rule.abs().max().detach().cpu())
                    )
                    for ai, area_out in enumerate(output_states):
                        print(
                            f"  output_states[{ai}] finite:",
                            bool(torch.isfinite(area_out).all()),
                            "max|out|:",
                            float(area_out.abs().max().detach().cpu()),
                        )
                    for ai, area_ns in enumerate(neuron_states):
                        for ti, ct_state in enumerate(area_ns):
                            finite = bool(torch.isfinite(ct_state).all())
                            max_abs = float(ct_state.abs().max().detach().cpu())
                            print(
                                f"  neuron_states[{ai}][{ti}] "
                                f"finite={finite}, max|state|={max_abs}"
                            )
                except Exception as e:
                    print(f"[SpatialRNN][Debug] tensor inspection failed: {e}")

            # Robust NaN/Inf guard: skip optimizer step if loss or gradients are non-finite.
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                msg = (
                    f"[SpatialRNN] Non-finite loss at block {block}, batch {batch_idx}, "
                    f"total_batch={total_batch}, fade={curriculum_fade_progress:.3f}."
                )
                print(msg)
                if debug_nonfinite:
                    raise RuntimeError(msg)
                # Do not backprop or step; keep model parameters unchanged for this batch.
                grad_norm = float("nan")
            else:
                loss.backward()
                max_norm = hp.get("grad_clip_max_norm", 1.0)
                grad_norm_tensor = clip_grad_norm_(
                    spatial_model.parameters(),
                    max_norm=float("inf") if max_norm is None else max_norm,
                )
                if not torch.isfinite(grad_norm_tensor):
                    msg = (
                        f"[SpatialRNN] Non-finite gradients at block {block}, batch {batch_idx}, "
                        f"total_batch={total_batch}, fade={curriculum_fade_progress:.3f}; "
                        "skipping optimizer step."
                    )
                    print(msg)
                    if debug_nonfinite:
                        raise RuntimeError(msg)
                    optimizer.zero_grad(set_to_none=True)
                    grad_norm = float("nan")
                else:
                    grad_norm = float(grad_norm_tensor.detach().cpu())
                    optimizer.step()

            total_val = float(loss.detach().cpu().item())
            resp_val = float(loss_resp.detach().cpu().item())
            rule_val = (
                float(loss_rule.detach().cpu().item()) if loss_rule is not None else float("nan")
            )
            recent_total.append(total_val)
            recent_resp.append(resp_val)
            if loss_rule is not None:
                recent_rule.append(rule_val)
            loss_hist.append(
                dict(
                    total=total_val,
                    resp=resp_val,
                    rule=rule_val,
                    block=float(block),
                    batch=float(batch_idx),
                    grad_norm=grad_norm,
                )
            )

            if save_path and _plot_every_eff is not None and (total_batch + 1) % _plot_every_eff == 0:
                try:
                    _save_loss_plot_live(save_path)
                except Exception as e:
                    print(f"[SpatialRNN] Warning: failed to update loss plot during training ({e})")

            if (batch_idx + 1) % print_every == 0:
                block_choice_acc = (
                    sum_choice_acc / max(1.0, float(n_acc)) if n_acc > 0 else float("nan")
                )
                block_rule_acc = (
                    sum_rule_acc / max(1.0, float(n_acc)) if n_acc > 0 else float("nan")
                )
                total_ma = sum(recent_total) / max(1, len(recent_total))
                resp_ma = sum(recent_resp) / max(1, len(recent_resp))
                rule_ma = (
                    sum(recent_rule) / max(1, len(recent_rule))
                    if len(recent_rule) > 0
                    else float("nan")
                )
                fade_suffix = f" fade={curriculum_fade_progress:.3f}" if in_fade else ""
                print(
                    f"[SpatialRNN] Block {block + 1}/{num_blocks}, "
                    f"Batch {batch_idx + 1}/{num_batches_per_block}, "
                    f"Loss {total_val:.4f} (ma{len(recent_total)} {total_ma:.4f}), "
                    f"Resp {resp_val:.4f} (ma{len(recent_resp)} {resp_ma:.4f}), "
                    f"Rule {rule_val:.4f} (ma{len(recent_rule)} {rule_ma:.4f}), "
                    f"Choice acc last {last_choice_acc:.3f} (block {block_choice_acc:.3f}), "
                    f"Rule acc last {last_rule_acc:.3f} (block {block_rule_acc:.3f}), "
                    f"grad_norm {grad_norm:.3f}{fade_suffix}"
                )

            prev_fade_progress = curriculum_fade_progress
            if (
                curriculum
                and fade_batches > 0
                and total_batch >= curriculum_phase1_batches
                and curriculum_fade_progress < 1.0
            ):
                curriculum_fade_batch_count += 1
                if (
                    curriculum_fade_max_batches is not None
                    and curriculum_fade_batch_count >= curriculum_fade_max_batches
                ):
                    curriculum_fade_progress = 1.0
                else:
                    block_choice_acc = (
                        sum_choice_acc / max(1.0, float(n_acc)) if n_acc > 0 else 0.0
                    )
                    block_rule_acc = (
                        sum_rule_acc / max(1.0, float(n_acc)) if n_acc > 0 else 0.0
                    )
                    perf_choice_ok = block_choice_acc >= curriculum_fade_perf_threshold
                    perf_rule_ok = block_rule_acc >= curriculum_fade_perf_threshold
                    perf_both_low = (
                        curriculum_fade_rollback_threshold is not None
                        and block_choice_acc < curriculum_fade_rollback_threshold
                        and block_rule_acc < curriculum_fade_rollback_threshold
                    )
                    if perf_choice_ok and perf_rule_ok:
                        curriculum_fade_consecutive_good += 1
                        if (
                            curriculum_fade_consecutive_good
                            >= curriculum_fade_min_consecutive_good
                        ):
                            curriculum_fade_progress = min(
                                1.0,
                                curriculum_fade_progress + 1.0 / fade_batches,
                            )
                            curriculum_fade_consecutive_good = 0
                    else:
                        curriculum_fade_consecutive_good = 0
                    if perf_both_low:
                        curriculum_fade_progress = max(
                            0.0,
                            curriculum_fade_progress - 1.0 / fade_batches,
                        )

            # Periodic checkpoint every `save_every` batches when enabled
            if save_path and _save_every_eff is not None and (total_batch + 1) % _save_every_eff == 0:
                base, ext = os.path.splitext(save_path)
                step_path = (
                    f"{base}_step{total_batch + 1}{ext}"
                    if ext
                    else f"{save_path}_step{total_batch + 1}"
                )
                _save_wcst_spatial(
                    spatial_model, step_path, first_in_channels, split_output_areas
                )
                # Append only new loss entries for this run alongside periodic checkpoints
                try:
                    _append_loss_history(save_path)
                except Exception as e:
                    print(f"[SpatialRNN] Warning: failed to update loss history ({e})")

            # Save checkpoint once the fade completes (before further training continues)
            if (
                curriculum
                and fade_batches > 0
                and total_batch >= curriculum_phase1_batches
                and (not curriculum_postfade_saved)
                and prev_fade_progress < 1.0
                and curriculum_fade_progress >= 1.0
            ):
                base_path = save_path or curriculum_pretrain_save_path
                if base_path:
                    postfade_path = _derive_postfade_path(base_path)
                    _save_wcst_spatial(spatial_model, postfade_path, first_in_channels, split_output_areas)
                    print(f"[SpatialRNN] Saved post-fade checkpoint to {postfade_path}")
                curriculum_postfade_saved = True

                if postfade_reset_optimizer:
                    new_lr = optimizer.param_groups[0]["lr"] * postfade_lr_scale
                    optimizer = _build_optimizer(spatial_model, hp)
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    print(
                        f"[SpatialRNN] Post-fade: reset optimizer, LR -> {new_lr:.2e}"
                    )
                elif postfade_lr_scale != 1.0:
                    for g in optimizer.param_groups:
                        g["lr"] *= postfade_lr_scale
                    print(
                        f"[SpatialRNN] Post-fade: LR scaled by {postfade_lr_scale} "
                        f"-> {optimizer.param_groups[0]['lr']:.2e}"
                    )

            if (
                curriculum
                and curriculum_pretrain_save_path
                and total_batch == curriculum_phase1_batches - 1
            ):
                _save_wcst_spatial(
                    spatial_model,
                    curriculum_pretrain_save_path,
                    first_in_channels,
                    split_output_areas,
                    optimizer=optimizer,
                    fade_progress=curriculum_fade_progress,
                    total_batch=total_batch,
                )
                print(
                    f"[SpatialRNN] Saved phase-1 pretrain to {curriculum_pretrain_save_path}"
                )

        # End-of-block checkpoint once fade is complete
        if curriculum_postfade_saved and save_path:
            base, ext = os.path.splitext(save_path)
            block_path = f"{base}_block{block}{ext}" if ext else f"{save_path}_block{block}"
            _save_wcst_spatial(
                spatial_model,
                block_path,
                first_in_channels,
                split_output_areas,
                optimizer=optimizer,
                fade_progress=curriculum_fade_progress,
                total_batch=last_total_batch,
            )
            print(f"[SpatialRNN] Saved end-of-block checkpoint to {block_path}")

    if return_loss_history:
        setattr(spatial_model, "loss_history", loss_hist)
    if save_path:
        try:
            _append_loss_history(save_path)
            print(
                f"[SpatialRNN] Saved loss history to "
                f"{_derive_loss_history_path(save_path)}"
            )
        except Exception as e:
            print(f"[SpatialRNN] Warning: failed to save loss history ({e})")

        _save_wcst_spatial(
            spatial_model,
            save_path,
            first_in_channels,
            split_output_areas,
            optimizer=optimizer,
            fade_progress=curriculum_fade_progress,
            total_batch=last_total_batch,
        )

        try:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")

            loss_plot_path = _derive_loss_plot_path(save_path)
            x_axis = list(range(len(loss_hist)))
            total = [d.get("total", float("nan")) for d in loss_hist]
            resp = [d.get("resp", float("nan")) for d in loss_hist]
            rule = [d.get("rule", float("nan")) for d in loss_hist]

            plt.figure(figsize=(10, 4))
            plt.plot(x_axis, total, label="total", linewidth=1.5)
            plt.plot(x_axis, resp, label="resp", linewidth=1.0, alpha=0.9)
            plt.plot(x_axis, rule, label="rule", linewidth=1.0, alpha=0.9)
            plt.xlabel("update")
            plt.ylabel("loss")
            plt.title("WCST SpatialRNN training loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_plot_path, dpi=200)
            plt.close()
            print(f"[SpatialRNN] Saved loss plot to {loss_plot_path}")
        except Exception as e:
            print(f"[SpatialRNN] Warning: failed to save loss plot ({e})")

        _save_wcst_spatial(spatial_model, save_path, first_in_channels, split_output_areas)
    return spatial_model


def train_wcst_simple_rnn(num_blocks: int = 5,
    num_batches_per_block: int = 2000,
    print_every: int = 50,
    hidden_size: int = 64,
    num_layers: int = 1,
    split_layers: bool = False,
    split_output: bool = True,
    trials_per_rollout: int | None = None,
    rule_loss_weight: float = 2.0,
    loss_window: int = 100,
    return_loss_history: bool = False,
    fixed_rule: bool = False,
    use_rule_cue: bool = False,
    rule_only: bool = False,
    curriculum: bool = False,
    curriculum_phase1_batches: int = 2000,
    curriculum_phase2_lr_scale: float = 0.5,
    curriculum_fade_batches: int | None = None,
    curriculum_fade_per_trial: bool = False,
    curriculum_fade_max_batches: int | None = None,
    curriculum_fade_perf_threshold: float = 0.8,
    curriculum_fade_rollback_threshold: float | None = None,
    curriculum_fade_min_consecutive_good: int = 1,
    curriculum_fade_start: float = 0.0,
    postfade_lr_scale: float = 0.5,
    postfade_reset_optimizer: bool = True,
    save_path: str | None = None,
    curriculum_pretrain_save_path: str | None = None,
    curriculum_pretrain_load_path: str | None = None,
    rule_cue_noise: float = 0.0,
    plot_every: int | None = None,
    save_every: int | None = 100,
) -> SimpleWCSTRNN:
    """Train a simple RNN on the WCST task to verify the training pipeline.

    Monitors training loss (instant + moving average over `loss_window`).

    Curriculum: when curriculum=True, phase 1 uses rule cue until
    curriculum_phase1_batches; then phase 2 continues with rule cue off (zeros).
    If curriculum_fade_batches is set, the cue is not switched off abruptly:
    the fade fraction (drop probability) now increases based on overall
    performance rather than loss. Specifically, after phase 1, if on a given
    batch BOTH the block-averaged choice and rule accuracy exceed
    curriculum_fade_perf_threshold, we advance the fade by 1/fade_batches.
    Optionally set curriculum_fade_rollback_threshold (e.g. 0.6): when both
    accuracies drop below it, we decrease fade by one step so the model gets
    the cue more often and can stabilize. Optionally set
    curriculum_fade_min_consecutive_good > 1 to require that many consecutive
    good batches before advancing (slows fade and can improve stability).
    curriculum_fade_max_batches caps total batches in fade (phase 2 forced after). When curriculum_pretrain_save_path
    is set, the model is saved automatically at the end of phase 1. When
    curriculum_pretrain_load_path is set and the file exists, that checkpoint
    is loaded and phase 1 is skipped (training starts at the fade). By default
    the mask is drawn per batch; if curriculum_fade_per_trial is True, per
    trial. Optionally scales LR by curriculum_phase2_lr_scale at the start of
    phase 2.

    curriculum_fade_start: initial fade progress when entering the fade phase (default 0.0).
    E.g. 0.8 means start the fade with 80% cue-drop probability instead of 0%; useful to
    resume or shorten the fade (e.g. after loading a pretrain that was saved mid-fade).

    postfade_lr_scale: when the fade completes, multiply the learning rate by this factor
    (default 0.5). Helps stabilize training at the cue-off transition.

    postfade_reset_optimizer: if True (default), the optimizer is re-created at fade
    completion (clearing Adam momentum/variance), then the new LR is set to
    old_lr * postfade_lr_scale. If False, only the LR is scaled (momentum kept).

    rule_cue_noise: when > 0 and the trial would show a rule cue, with this
    probability (per trial) the cue is dropped (no cue, zeros). Otherwise
    correct cue. Encourages the model to use trial history when the cue is absent.

    split_layers: if True, use a two-layer RNN as sensory + PFC: layer 0 (sensory)
    receives only current stimulus; layer 1 (PFC) receives sensory hidden state
    plus prev_stim, prev_reward, prev_choice, and optional rule_cue. Forces
    num_layers=2. Use load_wcst_simple_rnn(save_path) to load; checkpoint stores
    split_layers, history_input_size, and split_output.
    split_output: when split_layers=True, if True (default) choice is read from
    sensory layer and rule from PFC; if False, both are read from PFC.

    If save_path is set, the model state and config are saved there after
    training; use load_wcst_simple_rnn(save_path) to load for evaluation.

    Usage examples:

        # Curriculum with loss-conditioned fade (cue drop only when loss decreases)
        model = train_wcst_simple_rnn(
            num_blocks=5,
            num_batches_per_block=2000,
            print_every=50,
            curriculum=True,
            curriculum_phase1_batches=2000,
            curriculum_fade_batches=1500,
            curriculum_fade_per_trial=True,
            curriculum_phase2_lr_scale=0.7,
            curriculum_fade_max_batches=5000,  # optional: force phase 2 after this many batches in fade
            curriculum_fade_perf_threshold=0.8,  # advance when block acc >= this
            curriculum_fade_rollback_threshold=0.6,  # optional: roll back fade when both acc < this (stabilize)
            curriculum_fade_min_consecutive_good=3,  # optional: require 3 good batches before advancing
            curriculum_pretrain_save_path="wcst_pretrain.pt",  # auto-save after phase 1
            curriculum_pretrain_load_path="wcst_pretrain.pt",  # optional: load and skip phase 1 next run
            save_path="wcst_simple_rnn.pt",
        )

        # Cue on with occasional drop (no curriculum)
        model = train_wcst_simple_rnn(
            use_rule_cue=True,
            rule_cue_noise=0.2,
            num_blocks=5,
            num_batches_per_block=2000,
        )

        # Load and evaluate
        model = load_wcst_simple_rnn("wcst_simple_rnn.pt")
        choice_acc, rule_acc = eval_wcst_simple_rnn(model, num_trials=200, n_switches=3)
    """
    # Option B: provide trial history signals (prev stim / reward / choice)
    base_input_size = hp["n_input"] * 2 + 2 + 3
    if curriculum:
        # Always 39-dim so we can feed rule cue in phase 1 and zeros in phase 2
        input_size = base_input_size + 2
    else:
        extra_rule_cue = 2 if use_rule_cue else 0
        input_size = base_input_size + extra_rule_cue
    n_out = hp["n_output"]
    n_out_rule = hp["n_output_rule"]
    n_input = hp["n_input"]
    need_rule_channels = curriculum or use_rule_cue
    history_input_size = n_input + 2 + 3 + (2 if need_rule_channels else 0)

    if split_layers:
        num_layers = 2

    curriculum_skip_phase1 = False
    if curriculum and curriculum_pretrain_load_path and os.path.isfile(curriculum_pretrain_load_path):
        simple_model = load_wcst_simple_rnn(curriculum_pretrain_load_path, device)
        curriculum_skip_phase1 = True
        print(f"[SimpleRNN] Loaded phase-1 pretrain from {curriculum_pretrain_load_path}, skipping phase 1")
    else:
        simple_model = SimpleWCSTRNN(
            input_size=n_input if split_layers else input_size,
            hidden_size=hidden_size,
            n_output=n_out,
            n_output_rule=n_out_rule,
            num_layers=num_layers,
            split_layers=split_layers,
            history_input_size=history_input_size if split_layers else None,
            split_output=split_output,
        )
        simple_model.to(device)
    optimizer = _build_optimizer(simple_model, hp)
    criterion = loss_fnc

    hp_wcst = get_default_hp_wcst()
    # Match original WCST naming
    rule_list = ["color", "shape"]
    n_features_per_rule = 2
    n_test_cards = 3

    dt = hp["dt"]
    hist_start_ts = int(hp_wcst["trial_history_start"] // dt)
    hist_end_ts = int(hp_wcst["trial_history_end"] // dt)
    resp_start_ts = int(hp_wcst["resp_start"] // dt)
    resp_end_ts = int(hp_wcst["resp_end"] // dt)
    rule_start_ts = int(hp_wcst["trial_start"] // dt)
    rule_end_ts = int(hp_wcst["trial_end"] // dt)

    loss_hist: list[dict[str, float]] = []
    recent_total = deque(maxlen=max(1, int(loss_window)))
    recent_resp = deque(maxlen=max(1, int(loss_window)))
    recent_rule = deque(maxlen=max(1, int(loss_window)))

    # Progressive removal of trial-history inputs (prev stimulus, prev choice)
    # Only active when there is no explicit rule cue (training must rely on history).
    use_prev_stim = True
    use_prev_choice = True
    history_phase = 0  # 0: full history, 1: no prev_stim, 2: no prev_choice (and no prev_stim)
    eval_success_prev_stim = 0
    eval_success_prev_choice = 0
    eval_success_early_stop = 0
    early_stop = False

    def _compute_trial_history(
        last_rew: torch.Tensor | None,
        prev_stim_full: torch.Tensor | None,
        prev_choice_vec: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute trial-history currents similar to the original WCST training."""
        n_steps = int((hp_wcst["trial_end"] - hp_wcst["trial_start"]) // dt)

        I_prev_rew = torch.zeros(n_steps, batch_size, 2, device=device)
        I_prev_stim = torch.zeros(
            n_steps, batch_size, hp["n_input"], device=device
        )
        I_prev_choice = torch.zeros(
            n_steps, batch_size, hp["n_output"], device=device
        )

        input_start = int(hp_wcst["trial_history_start"] // dt)
        input_end = int(hp_wcst["trial_history_end"] // dt)

        stim_start_ts = int(hp_wcst["test_cards_on"] // dt)
        stim_end_ts = int(hp_wcst["test_cards_off"] // dt)

        if last_rew is not None:
            correct = last_rew.float()
            incorrect = 1.0 - correct
            I_prev_rew[input_start:input_end, :, 0] = correct.unsqueeze(0)
            I_prev_rew[input_start:input_end, :, 1] = incorrect.unsqueeze(0)

        if prev_stim_full is not None and use_prev_stim:
            stim_mean = prev_stim_full[stim_start_ts:stim_end_ts].mean(dim=0)
            I_prev_stim[input_start:input_end, :, :] = stim_mean.unsqueeze(0)

        if prev_choice_vec is not None and use_prev_choice:
            I_prev_choice[input_start:input_end, :, :] = prev_choice_vec.unsqueeze(0)

        return I_prev_rew, I_prev_stim, I_prev_choice

    def _pack_inputs(
        x_curr: torch.Tensor,
        I_prev_stim: torch.Tensor,
        I_prev_rew: torch.Tensor,
        I_prev_choice: torch.Tensor,
        rule_cue: torch.Tensor | None,
        n_channels: int,
    ) -> torch.Tensor:
        """
        Build model inputs with trial history. n_channels is the model input size
        (37 or 39). When 39, last 2 are rule cue if rule_cue is not None, else zeros.
        """
        t, b, n_in = x_curr.shape
        n_base = n_in * 2 + 2 + 3
        x_in = torch.zeros(
            t, b, n_channels, device=x_curr.device, dtype=x_curr.dtype
        )
        x_in[:, :, :n_in] = x_curr
        x_in[:, :, n_in : 2 * n_in] = I_prev_stim
        x_in[:, :, 2 * n_in : 2 * n_in + 2] = I_prev_rew
        x_in[:, :, 2 * n_in + 2 : n_base] = I_prev_choice

        if rule_cue is not None and n_channels >= n_base + 2:
            # rule_cue: (B, 2) -> broadcast to (T, B, 2)
            x_in[:, :, -rule_cue.shape[-1] :] = rule_cue.unsqueeze(0).expand(
                t, -1, -1
            )

        return x_in

    def _pack_history(
        I_prev_stim: torch.Tensor,
        I_prev_rew: torch.Tensor,
        I_prev_choice: torch.Tensor,
        rule_cue: torch.Tensor | None,
        history_size: int,
    ) -> torch.Tensor:
        """Build (T, B, history_size) for PFC layer: prev_stim | prev_rew | prev_choice | rule_cue."""
        t, b, _ = I_prev_stim.shape
        n_in = I_prev_stim.shape[-1]
        x_hist = torch.zeros(t, b, history_size, device=I_prev_stim.device, dtype=I_prev_stim.dtype)
        x_hist[:, :, :n_in] = I_prev_stim
        x_hist[:, :, n_in : n_in + 2] = I_prev_rew
        x_hist[:, :, n_in + 2 : n_in + 2 + 3] = I_prev_choice
        if rule_cue is not None and history_size >= n_in + 2 + 3 + 2:
            x_hist[:, :, -2:] = rule_cue.unsqueeze(0).expand(t, -1, -1)
        return x_hist

    phase2_lr_scaled = False
    # 0..1; curriculum_fade_start sets where fade begins when entering fade phase
    curriculum_fade_progress = max(0.0, min(1.0, curriculum_fade_start))
    curriculum_fade_batch_count = 0  # batches spent in fade (for max_batches cap)
    curriculum_fade_consecutive_good = 0  # for min_consecutive_good
    curriculum_postfade_saved = False
    loss_hist_saved_len = 0
    last_total_batch = -1
    loss_hist_saved_len = 0

    def _derive_postfade_path(base_path: str) -> str:
        base, ext = os.path.splitext(base_path)
        if ext:
            return f"{base}_postfade{ext}"
        return f"{base_path}_postfade"

    def _derive_loss_history_path(base_path: str) -> str:
        base, ext = os.path.splitext(base_path)
        # Keep extension even if non-.pt, for symmetry with existing conventions
        if ext:
            return f"{base}_loss_history.json"
        return f"{base_path}_loss_history.json"

    def _derive_loss_plot_path(base_path: str) -> str:
        base, ext = os.path.splitext(base_path)
        if ext:
            return f"{base}_loss.png"
        return f"{base_path}_loss.png"

    def _append_loss_history(base_path: str) -> None:
        """Append only new loss_hist entries from this run to existing history on disk."""
        nonlocal loss_hist_saved_len
        import json
        import os as _os

        loss_history_path = _derive_loss_history_path(base_path)
        existing: list[dict] = []
        if _os.path.isfile(loss_history_path):
            try:
                with open(loss_history_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = loaded
            except Exception:
                existing = []
        new_entries = loss_hist[loss_hist_saved_len:]
        if not new_entries:
            return
        combined = existing + new_entries
        with open(loss_history_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        loss_hist_saved_len = len(loss_hist)

    # Periodic loss plot saving: overwrite the same loss.png during training.
    _plot_every_eff: int | None = None
    if plot_every is not None and save_path is not None:
        try:
            _plot_every_eff = max(1, int(plot_every))
        except Exception:
            _plot_every_eff = None

    # Periodic checkpoint saving during training (default every 100 batches).
    _save_every_eff: int | None = None
    if save_every is not None:
        try:
            _save_every_eff = max(1, int(save_every))
        except Exception:
            _save_every_eff = None

    def _save_loss_plot_live(base_path: str) -> None:
        """Save or update the loss plot alongside the checkpoint during training."""
        if _plot_every_eff is None or not loss_hist:
            return
        try:
            import importlib
            plt = importlib.import_module("matplotlib.pyplot")
        except Exception:
            return
        loss_plot_path = _derive_loss_plot_path(base_path)
        x = list(range(len(loss_hist)))
        total = [d.get("total", float("nan")) for d in loss_hist]
        resp = [d.get("resp", float("nan")) for d in loss_hist]
        rule = [d.get("rule", float("nan")) for d in loss_hist]
        grad_norm_vals = [d.get("grad_norm", float("nan")) for d in loss_hist]

        plt.figure(figsize=(10, 4))
        plt.plot(x, total, label="total", linewidth=1.5)
        plt.plot(x, resp, label="resp", linewidth=1.0, alpha=0.9)
        plt.plot(x, rule, label="rule", linewidth=1.0, alpha=0.9)
        plt.xlabel("update")
        plt.ylabel("loss")
        plt.title("WCST SimpleRNN training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=200)
        plt.close()

    for block in range(num_blocks):
        for batch_idx in range(num_batches_per_block):
            if early_stop:
                break
            bsz = int(hp["batch_size"])
            total_batch = block * num_batches_per_block + batch_idx
            last_total_batch = total_batch
            if curriculum_skip_phase1 and total_batch < curriculum_phase1_batches:
                continue
            # Curriculum: phase 1 = rule cue on; then optional fade (loss-conditioned); then phase 2 = cue off
            fade_batches = curriculum_fade_batches if curriculum_fade_batches is not None else 0
            in_phase1 = curriculum and total_batch < curriculum_phase1_batches and not curriculum_skip_phase1
            in_fade = (
                curriculum
                and fade_batches > 0
                and total_batch >= curriculum_phase1_batches
                and curriculum_fade_progress < 1.0
            )
            # Fade fraction = progress (increases only when loss has decreased)
            fade_fraction = curriculum_fade_progress if in_fade else 0.0
            in_phase2_cue_off = (
                curriculum
                and total_batch >= curriculum_phase1_batches
                and (fade_batches == 0 or curriculum_fade_progress >= 1.0)
            )
            # Per-batch cue decision (used when not fading per trial)
            if curriculum and in_phase1:
                use_cue_this_batch = True
            elif curriculum and in_fade and not curriculum_fade_per_trial:
                use_cue_this_batch = random.random() > fade_fraction
            elif curriculum and in_phase2_cue_off:
                use_cue_this_batch = False
            else:
                use_cue_this_batch = use_rule_cue

            if (
                curriculum
                and not phase2_lr_scaled
                and total_batch >= curriculum_phase1_batches
            ):
                phase2_lr_scaled = True
                for g in optimizer.param_groups:
                    g["lr"] *= curriculum_phase2_lr_scale
                msg = (
                    f"[SimpleRNN] Curriculum: phase 2 (rule cue off), LR scaled by {curriculum_phase2_lr_scale}"
                    if fade_batches <= 0
                    else (
                        f"[SimpleRNN] Curriculum: cue fade over {fade_batches} batches"
                        + (" (per-trial)" if curriculum_fade_per_trial else "")
                        + f", LR scaled by {curriculum_phase2_lr_scale}"
                    )
                )
                print(msg)

            # Block structure and rule switches, mirroring the original WCST training
            block_len = int(hp.get("block_len", 20))
            n_switches = int(hp.get("n_switches", 3))
            if fixed_rule:
                switches = set()
                current_rule_idx = 0
            else:
                if block_len > 1 and n_switches > 0:
                    n_switches_eff = min(max(n_switches, 0), block_len - 1)
                    switches = set(
                        random.sample(range(block_len - 1), k=n_switches_eff)
                    )
                else:
                    switches = set()
                current_rule_idx = 0

            # Trial-history state across trials in this block
            last_rew_vec: torch.Tensor | None = None
            prev_stim_full: torch.Tensor | None = None
            prev_choice_vec: torch.Tensor | None = None

            # Carry RNN hidden state across trials inside this rollout
            if split_layers:
                h = (
                    torch.zeros(1, bsz, hidden_size, device=device),
                    torch.zeros(1, bsz, hidden_size, device=device),
                )
            else:
                h = torch.zeros(num_layers, bsz, hidden_size, device=device)

            total_loss = torch.tensor(0.0, device=device)
            total_loss_resp = torch.tensor(0.0, device=device)
            total_loss_rule = torch.tensor(0.0, device=device)

            last_choice_acc = float("nan")
            last_rule_acc = float("nan")

            # Block-level accuracy aggregates
            sum_choice_acc = 0.0
            sum_rule_acc = 0.0
            n_acc = 0

            # Last-trial loss diagnostics
            last_trial_total = float("nan")
            last_trial_resp = float("nan")
            last_trial_rule = float("nan")

            # Determine how many trials are in this rollout (default to block length)
            rollout_len = int(trials_per_rollout) if trials_per_rollout is not None else block_len

            for tr in range(rollout_len):
                # Trial-history currents from the previous trial
                I_prev_rew, I_prev_stim, I_prev_choice = _compute_trial_history(
                    last_rew=last_rew_vec,
                    prev_stim_full=prev_stim_full,
                    prev_choice_vec=prev_choice_vec,
                    batch_size=bsz,
                    device=device,
                )

                rule = rule_list[current_rule_idx]
                wcst = WCST(
                    hp=hp,
                    hp_wcst=hp_wcst,
                    rule=rule,
                    rule_list=rule_list,
                    n_features_per_rule=n_features_per_rule,
                    n_test_cards=n_test_cards,
                )

                x, x_rule, yhat, yhat_rule, task_data = wcst.make_task_batch(
                    batch_size=bsz
                )
                x = x.to(device)  # (T, B, n_input)
                yhat = yhat.to(device)  # (T, B, 3)
                yhat_rule = yhat_rule.to(device)  # (T, B, 2)

                # Optional explicit rule cue (phase 1 of curriculum or use_rule_cue)
                # Per-trial fade: mix cued and uncued trials in same rollout so RNN must use history
                if curriculum_fade_per_trial and curriculum:
                    if in_phase1:
                        use_cue_this_trial = True
                    elif in_fade:
                        use_cue_this_trial = random.random() > fade_fraction
                    else:
                        use_cue_this_trial = False
                else:
                    use_cue_this_trial = use_cue_this_batch

                rule_cue = None
                if use_cue_this_trial:
                    # With rule_cue_noise, per-trial drop cue with that probability (correct or no cue only)
                    drop_cue = rule_cue_noise > 0 and random.random() < rule_cue_noise
                    if not drop_cue:
                        rule_idx_tensor = torch.full(
                            (bsz,),
                            current_rule_idx,
                            device=device,
                            dtype=torch.long,
                        )
                        rule_cue = torch.nn.functional.one_hot(
                            rule_idx_tensor, num_classes=len(rule_list)
                        ).float()

                if split_layers:
                    x_sensory = x
                    x_hist = _pack_history(
                        I_prev_stim, I_prev_rew, I_prev_choice, rule_cue, history_input_size
                    )
                    y, y_rule, h = simple_model(x_sensory, x_history=x_hist, h0=h)
                else:
                    x_in = _pack_inputs(
                        x, I_prev_stim, I_prev_rew, I_prev_choice, rule_cue, input_size
                    )
                    y, y_rule, h = simple_model(x_in, h0=h)

                loss_resp = criterion(y, yhat)
                loss_rule = (
                    criterion(y_rule, yhat_rule) if hp.get("train_rule", True) else None
                )
                if rule_only and loss_rule is not None:
                    loss = rule_loss_weight * loss_rule
                else:
                    loss = loss_resp + (
                        rule_loss_weight * loss_rule if loss_rule is not None else 0.0
                    )

                # Optional simple L2 weight regularization (reuses hp['l2_weight'])
                reg_lambda = float(hp.get("l2_weight", 0.0))
                if reg_lambda > 0.0:
                    reg = torch.tensor(0.0, device=device)
                    for p in simple_model.parameters():
                        reg = reg + p.pow(2).mean()
                    loss = loss + reg_lambda * reg

                # Record last-trial losses for diagnostics
                last_trial_total = float(loss.detach().cpu().item())
                last_trial_resp = float(loss_resp.detach().cpu().item())
                last_trial_rule = (
                    float(loss_rule.detach().cpu().item())
                    if loss_rule is not None
                    else float("nan")
                )

                total_loss = total_loss + loss
                total_loss_resp = total_loss_resp + loss_resp
                if loss_rule is not None:
                    total_loss_rule = total_loss_rule + loss_rule

                # Update trial-history signals for next trial (no gradient through these)
                with torch.no_grad():
                    # Previous stimulus: full time series of last trial
                    prev_stim_full = x.detach()

                    # Previous choice: winner-take-all on mean response-period output
                    choice_prob = y[resp_start_ts:resp_end_ts].mean(dim=0)  # (B, 3)
                    choice_idx = choice_prob.argmax(dim=-1)  # (B,)
                    prev_choice_vec = torch.nn.functional.one_hot(
                        choice_idx, num_classes=hp["n_output"]
                    ).float()

                    # Reward: compare choice to target choice for this trial
                    target_prob = yhat[resp_start_ts:resp_end_ts].mean(dim=0)  # (B, 3)
                    target_idx = target_prob.argmax(dim=-1)
                    correct = choice_idx == target_idx
                    last_rew_vec = correct

                    last_choice_acc = correct.float().mean().item()

                    # Rule accuracy (mean over entire trial)
                    rule_pred = y_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(
                        dim=-1
                    )
                    rule_tgt = yhat_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(
                        dim=-1
                    )
                    last_rule_acc = (rule_pred == rule_tgt).float().mean().item()

                    sum_choice_acc += last_choice_acc
                    sum_rule_acc += last_rule_acc
                    n_acc += 1

                # Switch environment rule at designated trials within the block
                if tr in switches:
                    current_rule_idx = 1 - current_rule_idx

            # Average loss across trials in rollout
            roll = float(max(1, rollout_len))
            loss = total_loss / roll
            loss_resp = total_loss_resp / roll
            loss_rule = (
                (total_loss_rule / roll) if hp.get("train_rule", True) else None
            )

            optimizer.zero_grad()
            loss.backward()
            max_norm = hp.get("grad_clip_max_norm", 1.0)
            grad_norm = float(
                clip_grad_norm_(
                    simple_model.parameters(),
                    max_norm=float("inf") if max_norm is None else max_norm,
                ).detach().cpu()
            )
            optimizer.step()

            # Track losses
            total_val = float(loss.detach().cpu().item())
            resp_val = float(loss_resp.detach().cpu().item())
            rule_val = (
                float(loss_rule.detach().cpu().item()) if loss_rule is not None
                else float("nan")
            )
            recent_total.append(total_val)
            recent_resp.append(resp_val)
            if loss_rule is not None:
                recent_rule.append(rule_val)
            loss_hist.append(
                dict(
                    total=total_val,
                    resp=resp_val,
                    rule=rule_val,
                    block=float(block),
                    batch=float(batch_idx),
                    last_trial_total=last_trial_total,
                    last_trial_resp=last_trial_resp,
                    last_trial_rule=last_trial_rule,
                    grad_norm=grad_norm,
                )
            )

            if save_path and _plot_every_eff is not None and (total_batch + 1) % _plot_every_eff == 0:
                try:
                    _save_loss_plot_live(save_path)
                except Exception as e:
                    print(f"[SimpleRNN] Warning: failed to update loss plot during training ({e})")

            if (batch_idx + 1) % print_every == 0:
                # Block-level average accuracies
                block_choice_acc = (
                    sum_choice_acc / max(1.0, float(n_acc))
                    if n_acc > 0
                    else float("nan")
                )
                block_rule_acc = (
                    sum_rule_acc / max(1.0, float(n_acc))
                    if n_acc > 0
                    else float("nan")
                )

                total_ma = sum(recent_total) / max(1, len(recent_total))
                resp_ma = sum(recent_resp) / max(1, len(recent_resp))
                rule_ma = (
                    (sum(recent_rule) / max(1, len(recent_rule)))
                    if len(recent_rule) > 0
                    else float("nan")
                )
                fade_suffix = (
                    f" fade={curriculum_fade_progress:.3f}" if in_fade else ""
                )
                print(
                    f"[SimpleRNN] Block {block + 1}/{num_blocks}, "
                    f"Batch {batch_idx + 1}/{num_batches_per_block}, "
                    f"Loss {total_val:.4f} (ma{len(recent_total)} {total_ma:.4f}), "
                    f"Resp {resp_val:.4f} (ma{len(recent_resp)} {resp_ma:.4f}), "
                    f"Rule {rule_val:.4f} (ma{len(recent_rule)} {rule_ma:.4f}), "
                    f"Last-trial loss {last_trial_total:.4f} "
                    f"(resp {last_trial_resp:.4f}, rule {last_trial_rule:.4f}), "
                    f"Choice acc last {last_choice_acc:.3f} "
                    f"(block {block_choice_acc:.3f}), "
                    f"Rule acc last {last_rule_acc:.3f} "
                    f"(block {block_rule_acc:.3f}), "
                    f"grad_norm {grad_norm:.3f}{fade_suffix}"
                )

                # When training without an effective rule cue (either no-curriculum
                # or curriculum phase 2 where the cue is fully dropped) and performance
                # is high, periodically evaluate on a longer WCST sequence with
                # history-based inputs but no cue. If the model maintains high
                # performance for 5 such tests, first drop the prev stimulus input
                # from training, then (after another 5 tests) drop the prev choice
                # input as well.
                if (
                    (
                        (not curriculum and not use_rule_cue)
                        or (curriculum and in_phase2_cue_off)
                    )
                    and not rule_only
                    and block_choice_acc >= 0.8
                    and block_rule_acc >= 0.8
                ):
                    # Choose which phase we are in and track consecutive successes separately
                    target_phase = history_phase
                    try:
                        choice_acc_long, rule_acc_long = eval_wcst_simple_rnn(
                            simple_model,
                            num_trials=200,
                            n_switches=10,
                            fixed_rule=False,
                            use_history=True,
                            use_rule_cue=False,
                        )
                    finally:
                        # Ensure we return to training mode after eval.
                        simple_model.train()

                    print(
                        f"[SimpleRNN] Long-sequence eval result (phase {target_phase}): "
                        f"choice_acc={choice_acc_long:.3f}, rule_acc={rule_acc_long:.3f}"
                    )

                    if choice_acc_long >= 0.9 and rule_acc_long >= 0.9:
                        if target_phase == 0:
                            eval_success_prev_stim += 1
                            print(
                                f"[SimpleRNN] Long-sequence eval (phase 0) success "
                                f"{eval_success_prev_stim}/5: "
                                f"choice_acc={choice_acc_long:.3f}, rule_acc={rule_acc_long:.3f}"
                            )
                            if eval_success_prev_stim >= 5:
                                use_prev_stim = False
                                history_phase = 1
                                eval_success_early_stop = 0
                                print(
                                    "[SimpleRNN] Reached stable high performance on long sequence "
                                    "without rule cue; disabling prev stimulus input for further training."
                                )
                        elif target_phase == 1:
                            eval_success_prev_choice += 1
                            print(
                                f"[SimpleRNN] Long-sequence eval (phase 1) success "
                                f"{eval_success_prev_choice}/5: "
                                f"choice_acc={choice_acc_long:.3f}, rule_acc={rule_acc_long:.3f}"
                            )
                            if eval_success_prev_choice >= 5:
                                use_prev_choice = False
                                history_phase = 2
                                eval_success_early_stop = 0
                                print(
                                    "[SimpleRNN] Reached stable high performance on long sequence "
                                    "without rule cue; disabling prev choice input for remaining training."
                                )
                        elif target_phase == 2:
                            eval_success_early_stop += 1
                            print(
                                f"[SimpleRNN] Long-sequence eval (phase 2: no prev_stim/choice) success "
                                f"{eval_success_early_stop}/5: "
                                f"choice_acc={choice_acc_long:.3f}, rule_acc={rule_acc_long:.3f}"
                            )
                            if eval_success_early_stop >= 5:
                                early_stop = True
                                print(
                                    "[SimpleRNN] Early stop: stable high performance on long sequence "
                                    "after dropping prev stimulus and prev choice."
                                )
                    else:
                        if target_phase == 0 and eval_success_prev_stim > 0:
                            print(
                                f"[SimpleRNN] Long-sequence eval (phase 0) failed, "
                                f"resetting success counter (choice_acc={choice_acc_long:.3f}, "
                                f"rule_acc={rule_acc_long:.3f})."
                            )
                            eval_success_prev_stim = 0
                        elif target_phase == 1 and eval_success_prev_choice > 0:
                            print(
                                f"[SimpleRNN] Long-sequence eval (phase 1) failed, "
                                f"resetting success counter (choice_acc={choice_acc_long:.3f}, "
                                f"rule_acc={rule_acc_long:.3f})."
                            )
                            eval_success_prev_choice = 0
                        elif target_phase == 2 and eval_success_early_stop > 0:
                            print(
                                f"[SimpleRNN] Long-sequence eval (phase 2) failed, "
                                f"resetting early-stop counter (choice_acc={choice_acc_long:.3f}, "
                                f"rule_acc={rule_acc_long:.3f})."
                            )
                            eval_success_early_stop = 0

            # Performance-conditioned fade: advance drop probability when overall performance is high
            prev_fade_progress = curriculum_fade_progress
            if (
                curriculum
                and fade_batches > 0
                and total_batch >= curriculum_phase1_batches
                and curriculum_fade_progress < 1.0
            ):
                curriculum_fade_batch_count += 1
                # Cap: force phase 2 after max_batches in fade if set
                if (
                    curriculum_fade_max_batches is not None
                    and curriculum_fade_batch_count >= curriculum_fade_max_batches
                ):
                    curriculum_fade_progress = 1.0
                else:
                    # Use block-averaged accuracies as the performance signal
                    block_choice_acc = (
                        sum_choice_acc / max(1.0, float(n_acc)) if n_acc > 0 else 0.0
                    )
                    block_rule_acc = (
                        sum_rule_acc / max(1.0, float(n_acc)) if n_acc > 0 else 0.0
                    )
                    perf_choice_ok = block_choice_acc >= curriculum_fade_perf_threshold
                    perf_rule_ok = block_rule_acc >= curriculum_fade_perf_threshold
                    perf_both_low = (
                        curriculum_fade_rollback_threshold is not None
                        and block_choice_acc < curriculum_fade_rollback_threshold
                        and block_rule_acc < curriculum_fade_rollback_threshold
                    )
                    if perf_choice_ok and perf_rule_ok:
                        curriculum_fade_consecutive_good += 1
                        if (
                            curriculum_fade_consecutive_good
                            >= curriculum_fade_min_consecutive_good
                        ):
                            curriculum_fade_progress = min(
                                1.0,
                                curriculum_fade_progress + 1.0 / fade_batches,
                            )
                            curriculum_fade_consecutive_good = 0
                    else:
                        curriculum_fade_consecutive_good = 0
                    if perf_both_low:
                        curriculum_fade_progress = max(
                            0.0,
                            curriculum_fade_progress - 1.0 / fade_batches,
                        )

            # Periodic checkpoint every `save_every` batches when enabled
            if save_path and _save_every_eff is not None and (total_batch + 1) % _save_every_eff == 0:
                base, ext = os.path.splitext(save_path)
                step_path = (
                    f"{base}_step{total_batch + 1}{ext}"
                    if ext
                    else f"{save_path}_step{total_batch + 1}"
                )
                _save_wcst_simple_rnn(
                    simple_model,
                    step_path,
                    optimizer=optimizer,
                    fade_progress=curriculum_fade_progress,
                    total_batch=total_batch,
                )
                # Append only new loss entries for this run alongside periodic checkpoints
                try:
                    _append_loss_history(save_path)
                except Exception as e:
                    print(f"[SimpleRNN] Warning: failed to update loss history ({e})")

            # Save checkpoint once the fade completes (before further training continues)
            if (
                curriculum
                and fade_batches > 0
                and total_batch >= curriculum_phase1_batches
                and (not curriculum_postfade_saved)
                and prev_fade_progress < 1.0
                and curriculum_fade_progress >= 1.0
            ):
                base_path = save_path or curriculum_pretrain_save_path
                if base_path:
                    postfade_path = _derive_postfade_path(base_path)
                    _save_wcst_simple_rnn(simple_model, postfade_path)
                    print(f"[SimpleRNN] Saved post-fade checkpoint to {postfade_path}")
                curriculum_postfade_saved = True

                if postfade_reset_optimizer:
                    new_lr = optimizer.param_groups[0]["lr"] * postfade_lr_scale
                    optimizer = _build_optimizer(simple_model, hp)
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    print(
                        f"[SimpleRNN] Post-fade: reset optimizer, LR -> {new_lr:.2e}"
                    )
                elif postfade_lr_scale != 1.0:
                    for g in optimizer.param_groups:
                        g["lr"] *= postfade_lr_scale
                    print(
                        f"[SimpleRNN] Post-fade: LR scaled by {postfade_lr_scale} "
                        f"-> {optimizer.param_groups[0]['lr']:.2e}"
                    )

            # Save pretrain checkpoint at end of phase 1
            if (
                curriculum
                and curriculum_pretrain_save_path
                and total_batch == curriculum_phase1_batches - 1
            ):
                _save_wcst_simple_rnn(simple_model, curriculum_pretrain_save_path)
                print(f"[SimpleRNN] Saved phase-1 pretrain to {curriculum_pretrain_save_path}")

        # End-of-block checkpoint once fade is complete
        if curriculum_postfade_saved and save_path:
            base, ext = os.path.splitext(save_path)
            block_path = f"{base}_block{block}{ext}" if ext else f"{save_path}_block{block}"
            _save_wcst_simple_rnn(simple_model, block_path)
            print(f"[SimpleRNN] Saved end-of-block checkpoint to {block_path}")
        if early_stop:
            break

    if return_loss_history:
        setattr(simple_model, "loss_history", loss_hist)
    if save_path:
        # Save loss history + plot alongside the checkpoint
        try:
            _append_loss_history(save_path)
            print(
                f"[SimpleRNN] Saved loss history to "
                f"{_derive_loss_history_path(save_path)}"
            )
        except Exception as e:
            print(f"[SimpleRNN] Warning: failed to save loss history ({e})")

        _save_wcst_simple_rnn(
            simple_model,
            save_path,
            optimizer=optimizer,
            fade_progress=curriculum_fade_progress,
            total_batch=total_batch,
        )

        try:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")

            loss_plot_path = _derive_loss_plot_path(save_path)
            x = list(range(len(loss_hist)))
            total = [d.get("total", float("nan")) for d in loss_hist]
            resp = [d.get("resp", float("nan")) for d in loss_hist]
            rule = [d.get("rule", float("nan")) for d in loss_hist]

            plt.figure(figsize=(10, 4))
            plt.plot(x, total, label="total", linewidth=1.5)
            plt.plot(x, resp, label="resp", linewidth=1.0, alpha=0.9)
            plt.plot(x, rule, label="rule", linewidth=1.0, alpha=0.9)
            plt.xlabel("update")
            plt.ylabel("loss")
            plt.title("WCST SimpleRNN training loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_plot_path, dpi=200)
            plt.close()
            print(f"[SimpleRNN] Saved loss plot to {loss_plot_path}")
        except Exception as e:
            print(f"[SimpleRNN] Warning: failed to save loss plot ({e})")

        _save_wcst_simple_rnn(simple_model, save_path)
    return simple_model


def _save_wcst_simple_rnn(
    model: SimpleWCSTRNN,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    fade_progress: float | None = None,
    total_batch: int | None = None,
) -> None:
    """Save model state and config so it can be loaded with load_wcst_simple_rnn."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "n_output": model.n_output,
        "n_output_rule": model.n_output_rule,
        "num_layers": model.num_layers,
        "split_layers": getattr(model, "split_layers", False),
        "history_input_size": getattr(model, "history_input_size", None),
        "split_output": getattr(model, "split_output", True),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "fade_progress": fade_progress,
        "total_batch": total_batch,
    }
    torch.save(checkpoint, path)


def load_wcst_simple_rnn(
    path: str,
    device: torch.device | None = None,
) -> SimpleWCSTRNN:
    """Load a SimpleWCSTRNN saved with train_wcst_simple_rnn(..., save_path=...)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", {})
    # Old split_layers checkpoints had a single "readout"; infer split_output from state_dict
    if checkpoint.get("split_layers", False) and "split_output" not in checkpoint:
        split_output = "readout_choice" in state
    else:
        split_output = checkpoint.get("split_output", True)
    model = SimpleWCSTRNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        n_output=checkpoint["n_output"],
        n_output_rule=checkpoint["n_output_rule"],
        num_layers=checkpoint.get("num_layers", 1),
        split_layers=checkpoint.get("split_layers", False),
        history_input_size=checkpoint.get("history_input_size"),
        split_output=split_output,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def _save_wcst_spatial(
    model: SpatiallyEmbeddedRNN,
    path: str,
    first_in_channels: int,
    split_output_areas: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    fade_progress: float | None = None,
    total_batch: int | None = None,
) -> None:
    """Save spatial WCST model state and channel-routing metadata."""
    first_area_in = int(getattr(model.areas[0], "in_channels"))
    total_input_channels = int(
        getattr(model, "input_channels", first_area_in)
    )
    area_external = getattr(model, "area_external_in_channels", None)
    pfc_external_in = (
        int(area_external[1])
        if area_external is not None and len(area_external) > 1
        else max(
            0,
            int(getattr(model.areas[1], "in_channels"))
            - int(getattr(model.areas[0], "out_channels")),
        )
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            # Legacy key kept for backward compatibility with existing loaders.
            "in_channels": first_in_channels,
            "first_area_in_channels": first_area_in,
            "pfc_external_in_channels": pfc_external_in,
            "total_input_channels": total_input_channels,
            "split_output_areas": split_output_areas,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "fade_progress": fade_progress,
            "total_batch": total_batch,
        },
        path,
    )


def load_wcst_spatial(
    path: str, device: torch.device | None = None
) -> SpatiallyEmbeddedRNN:
    """Load a SpatiallyEmbeddedRNN saved with train_wcst(..., save_path=...) or pretrain save."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # New checkpoints store total_input_channels explicitly. Older checkpoints only
    # store "in_channels", which could mean either total packed channels (newer
    # pipeline) or first-area channels (older pipeline). Infer robustly.
    base_sensory_in = int(hp["n_input"])
    base_total_in = int(hp["n_input"] * 2 + 2 + 3)
    if "total_input_channels" in checkpoint:
        in_channels = int(checkpoint["total_input_channels"])
    else:
        legacy_in = int(checkpoint.get("in_channels", base_total_in))
        # Legacy first-area format: n_input (+ optional 2 rule-cue channels).
        if legacy_in <= base_sensory_in + 2:
            in_channels = base_total_in + max(0, legacy_in - base_sensory_in)
        else:
            # Already in packed-total format.
            in_channels = legacy_in
    split_output = checkpoint.get("split_output_areas", True)
    spatial_model = _build_spatial_wcst_model(in_channels, split_output)
    spatial_model.load_state_dict(checkpoint["state_dict"])
    spatial_model.to(device)
    return spatial_model


def eval_wcst_simple_rnn(
    model: SimpleWCSTRNN,
    num_trials: int = 200,
    n_switches: int = 3,
    fixed_rule: bool = False,
    *,
    use_history: bool = False,
    use_rule_cue: bool = False,
) -> tuple[float, float]:
    """Frozen-weights evaluation over a WCST sequence for the SimpleWCSTRNN.

    - If `use_history=False` (default), behaviour matches the original simple eval:
      only current stimulus is provided (trial-history and rule cue channels are zero).
    - If `use_history=True`, behaviour matches the training-time inputs:
      prev stimulus summary, prev reward, prev choice, and optional rule cue
      (`use_rule_cue=True`) are fed via the history channels.
    """
    model.eval()
    hp_wcst = get_default_hp_wcst()
    rule_list = ["color", "shape"]

    dt = hp["dt"]
    resp_start_ts = int(hp_wcst["resp_start"] // dt)
    resp_end_ts = int(hp_wcst["resp_end"] // dt)
    rule_start_ts = int(hp_wcst["trial_start"] // dt)
    rule_end_ts = int(hp_wcst["trial_end"] // dt)

    bsz = int(hp["batch_size"])
    device = next(model.parameters()).device

    correct_choice: list[float] = []
    correct_rule: list[float] = []

    def _make_switches(num_trials_local: int, n_switches_local: int) -> set[int]:
        """Sample switch indices in [0, num_trials-2], matching training's convention."""
        if fixed_rule:
            return set()
        if num_trials_local <= 1 or n_switches_local <= 0:
            return set()
        n_switches_eff = min(max(int(n_switches_local), 0), int(num_trials_local) - 1)
        return set(random.sample(range(int(num_trials_local) - 1), k=n_switches_eff))

    current_rule_idx = 0
    switches = _make_switches(int(num_trials), int(n_switches))

    # Setup RNN hidden state
    if getattr(model, "split_layers", False):
        h: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros(1, bsz, model.hidden_size, device=device),
            torch.zeros(1, bsz, model.hidden_size, device=device),
        )
    else:
        h = torch.zeros(model.num_layers, bsz, model.hidden_size, device=device)

    # When using history-based eval, we need extra metadata and state
    if use_history:
        input_start = int(hp_wcst["trial_history_start"] // dt)
        input_end = int(hp_wcst["trial_history_end"] // dt)
        stim_start_ts = int(hp_wcst["test_cards_on"] // dt)
        stim_end_ts = int(hp_wcst["test_cards_off"] // dt)
        n_in = int(hp["n_input"])
        n_out = int(hp["n_output"])

        last_rew_vec: torch.Tensor | None = None  # (B,)
        prev_stim_full: torch.Tensor | None = None  # (T,B,n_in)
        prev_choice_vec: torch.Tensor | None = None  # (B,n_out)

        def _trial_history_currents() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Build (T,B,*) trial-history currents for the current trial."""
            n_steps = int((hp_wcst["trial_end"] - hp_wcst["trial_start"]) // dt)
            I_prev_rew = torch.zeros(n_steps, bsz, 2, device=device)
            I_prev_stim = torch.zeros(n_steps, bsz, n_in, device=device)
            I_prev_choice = torch.zeros(n_steps, bsz, n_out, device=device)

            if last_rew_vec is not None:
                correct = last_rew_vec.float()
                incorrect = 1.0 - correct
                I_prev_rew[input_start:input_end, :, 0] = correct.unsqueeze(0)
                I_prev_rew[input_start:input_end, :, 1] = incorrect.unsqueeze(0)

            if prev_stim_full is not None:
                stim_mean = prev_stim_full[stim_start_ts:stim_end_ts].mean(dim=0)
                I_prev_stim[input_start:input_end, :, :] = stim_mean.unsqueeze(0)

            if prev_choice_vec is not None:
                I_prev_choice[input_start:input_end, :, :] = prev_choice_vec.unsqueeze(0)

            return I_prev_rew, I_prev_stim, I_prev_choice

        def _pack_inputs_with_history(
            x_curr: torch.Tensor,
            I_prev_stim: torch.Tensor,
            I_prev_rew: torch.Tensor,
            I_prev_choice: torch.Tensor,
            rule_cue: torch.Tensor | None,
            n_channels: int,
        ) -> torch.Tensor:
            t, b, n_in_local = x_curr.shape
            n_base = n_in_local * 2 + 2 + 3
            x_in = torch.zeros(t, b, n_channels, device=device, dtype=x_curr.dtype)
            x_in[:, :, :n_in_local] = x_curr
            x_in[:, :, n_in_local : 2 * n_in_local] = I_prev_stim
            x_in[:, :, 2 * n_in_local : 2 * n_in_local + 2] = I_prev_rew
            x_in[:, :, 2 * n_in_local + 2 : n_base] = I_prev_choice
            if rule_cue is not None and n_channels >= n_base + 2:
                x_in[:, :, -rule_cue.shape[-1] :] = rule_cue.unsqueeze(0).expand(t, -1, -1)
            return x_in

        def _pack_history_with_cue(
            I_prev_stim: torch.Tensor,
            I_prev_rew: torch.Tensor,
            I_prev_choice: torch.Tensor,
            rule_cue: torch.Tensor | None,
            history_size: int,
        ) -> torch.Tensor:
            t, b, _ = I_prev_stim.shape
            x_hist = torch.zeros(t, b, history_size, device=device, dtype=I_prev_stim.dtype)
            x_hist[:, :, :n_in] = I_prev_stim
            x_hist[:, :, n_in : n_in + 2] = I_prev_rew
            x_hist[:, :, n_in + 2 : n_in + 2 + 3] = I_prev_choice
            if rule_cue is not None and history_size >= n_in + 2 + 3 + 2:
                x_hist[:, :, -2:] = rule_cue.unsqueeze(0).expand(t, -1, -1)
            return x_hist

    with torch.no_grad():
        for tr in range(int(num_trials)):
            rule = rule_list[current_rule_idx]
            wcst = WCST(
                hp=hp,
                hp_wcst=hp_wcst,
                rule=rule,
                rule_list=rule_list,
                n_features_per_rule=2,
                n_test_cards=3,
            )
            x, _, yhat, yhat_rule, _ = wcst.make_task_batch(batch_size=bsz)
            x = x.to(device)
            yhat = yhat.to(device)
            yhat_rule = yhat_rule.to(device)

            t, b, n_in_local = x.shape
            if use_history:
                I_prev_rew, I_prev_stim, I_prev_choice = _trial_history_currents()

                rule_cue = None
                if use_rule_cue:
                    rule_idx_tensor = torch.full(
                        (bsz,), current_rule_idx, device=device, dtype=torch.long
                    )
                    rule_cue = torch.nn.functional.one_hot(
                        rule_idx_tensor, num_classes=len(rule_list)
                    ).float()

                if getattr(model, "split_layers", False):
                    hist_size = getattr(model, "history_input_size", None)
                    if hist_size is None:
                        raise ValueError(
                            "Model has split_layers=True but history_input_size is None; "
                            "cannot run history-based evaluation."
                        )
                    x_hist = _pack_history_with_cue(
                        I_prev_stim,
                        I_prev_rew,
                        I_prev_choice,
                        rule_cue,
                        int(hist_size),
                    )
                    y, y_rule, h = model(x, x_history=x_hist, h0=h)
                else:
                    x_in = _pack_inputs_with_history(
                        x,
                        I_prev_stim,
                        I_prev_rew,
                        I_prev_choice,
                        rule_cue,
                        int(model.input_size),
                    )
                    y, y_rule, h = model(x_in, h0=h)
            else:
                # Simple eval: only current stimulus, history channels zero
                if getattr(model, "split_layers", False):
                    x_sensory = x
                    x_history = torch.zeros(
                        t, b, model.history_input_size, device=device, dtype=x.dtype
                    )
                    y, y_rule, h = model(x_sensory, x_history=x_history, h0=h)
                else:
                    n_channels = model.input_size
                    x_in = torch.zeros(t, b, n_channels, device=device, dtype=x.dtype)
                    x_in[:, :, :n_in_local] = x
                    y, y_rule, h = model(x_in, h0=h)

            choice_idx = y[resp_start_ts:resp_end_ts].mean(dim=0).argmax(dim=-1)
            target_idx = yhat[resp_start_ts:resp_end_ts].mean(dim=0).argmax(dim=-1)
            correct = (choice_idx == target_idx)

            rule_pred = y_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(dim=-1)
            rule_tgt = (
                yhat_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(dim=-1)
            )
            correct_r = (rule_pred == rule_tgt)

            correct_choice.append(float(correct.float().mean().item()))
            correct_rule.append(float(correct_r.float().mean().item()))

            # Update trial-history state for the next trial (only needed when using history)
            if use_history:
                prev_stim_full = x.detach()
                prev_choice_vec = torch.nn.functional.one_hot(
                    choice_idx, num_classes=n_out
                ).float()
                last_rew_vec = correct.detach()

            if tr in switches:
                current_rule_idx = 1 - current_rule_idx

    return float(sum(correct_choice) / len(correct_choice)), float(sum(correct_rule) / len(correct_rule))


def eval_wcst_spatial(
    model: SpatiallyEmbeddedRNN,
    num_trials: int = 200,
    batch_size: int | None = None,
    split_output_areas: bool | None = None,
    *,
    n_switches: int = 3,
    fixed_rule: bool = False,
    use_history: bool = False,
    use_rule_cue: bool = False,
) -> tuple[float, float]:
    """Frozen-weights evaluation over a WCST sequence for SpatiallyEmbeddedRNN.

    Mirrors eval_wcst_simple_rnn switching (`n_switches` over `num_trials`).

    - If `use_history=False` (default), trial-history channels are zero.
    - If `use_history=True`, feed prev stimulus summary, prev reward, prev choice,
      and optional explicit rule cue (`use_rule_cue=True`) through the input channels,
      matching the training-time input packing.

    If split_output_areas is None, it is inferred from the last area's out_channels:
    - True when last out_channels == n_output_rule (rule-only PFC output)
    - False when last out_channels >= n_output + n_output_rule (combined output)
    """
    model.eval()
    hp_wcst = get_default_hp_wcst()
    rule_list = ["color", "shape"]

    dt = hp["dt"]
    resp_start_ts = int(hp_wcst["resp_start"] // dt)
    resp_end_ts = int(hp_wcst["resp_end"] // dt)
    rule_start_ts = int(hp_wcst["trial_start"] // dt)
    rule_end_ts = int(hp_wcst["trial_end"] // dt)

    bsz = int(hp["batch_size"]) if batch_size is None else int(batch_size)
    device = next(model.parameters()).device
    n_out = int(hp["n_output"])
    n_out_rule = int(hp["n_output_rule"])

    if split_output_areas is None:
        last_out = int(getattr(model.areas[-1], "out_channels", n_out_rule))
        split_output_areas = last_out == n_out_rule

    n_channels = int(getattr(model, "input_channels", getattr(model.areas[0], "in_channels")))
    _sensory_in = int(getattr(model.areas[0], "in_channels"))
    _area_external = getattr(model, "area_external_in_channels", None)
    _pfc_extra = int(_area_external[1]) if _area_external is not None and len(_area_external) > 1 else max(0, int(getattr(model.areas[1], "in_channels")) - int(getattr(model.areas[0], "out_channels")))
    print(
        f"[SpatialRNN][eval] Channel routing: sensory_in={_sensory_in}, "
        f"pfc_extra={_pfc_extra}, total_input_channels={n_channels}"
    )
    in_size = tuple(getattr(model.areas[0], "in_size"))
    spatial_size = (int(in_size[0]), int(in_size[1]))

    correct_choice: list[float] = []
    correct_rule: list[float] = []

    def _make_switches(num_trials_local: int, n_switches_local: int) -> set[int]:
        """Sample switch indices in [0, num_trials-2], matching training's convention."""
        if fixed_rule:
            return set()
        if num_trials_local <= 1 or n_switches_local <= 0:
            return set()
        n_switches_eff = min(max(int(n_switches_local), 0), int(num_trials_local) - 1)
        return set(random.sample(range(int(num_trials_local) - 1), k=n_switches_eff))

    current_rule_idx = 0
    switches = _make_switches(int(num_trials), int(n_switches))

    # Trial-history state (only used when use_history=True)
    last_rew_vec: torch.Tensor | None = None  # (B,)
    prev_stim_full: torch.Tensor | None = None  # (T,B,n_in)
    prev_choice_vec: torch.Tensor | None = None  # (B,n_out)

    # Carry spatial model states across trials (mirrors training)
    carry_output_state = None
    carry_neuron_state = None
    carry_feedback_state = None

    # Trial-history injection window (matches training)
    input_start = int(hp_wcst["trial_history_start"] // dt)
    input_end = int(hp_wcst["trial_history_end"] // dt)
    stim_start_ts = int(hp_wcst["test_cards_on"] // dt)
    stim_end_ts = int(hp_wcst["test_cards_off"] // dt)

    with torch.no_grad():
        for tr in range(int(num_trials)):
            rule = rule_list[current_rule_idx]
            wcst = WCST(
                hp=hp,
                hp_wcst=hp_wcst,
                rule=rule,
                rule_list=rule_list,
                n_features_per_rule=2,
                n_test_cards=3,
            )
            x, _, yhat, yhat_rule, _ = wcst.make_task_batch(batch_size=bsz)
            x = x.to(device)
            yhat = yhat.to(device)
            yhat_rule = yhat_rule.to(device)

            t, b, n_in = x.shape

            # Build trial-history currents (zeros if not using history)
            n_steps = int((hp_wcst["trial_end"] - hp_wcst["trial_start"]) // dt)
            I_prev_rew = torch.zeros(n_steps, bsz, 2, device=device)
            I_prev_stim = torch.zeros(n_steps, bsz, n_in, device=device)
            I_prev_choice = torch.zeros(n_steps, bsz, n_out, device=device)

            if use_history:
                if last_rew_vec is not None:
                    correct_prev = last_rew_vec.float()
                    incorrect_prev = 1.0 - correct_prev
                    I_prev_rew[input_start:input_end, :, 0] = correct_prev.unsqueeze(0)
                    I_prev_rew[input_start:input_end, :, 1] = incorrect_prev.unsqueeze(0)
                if prev_stim_full is not None:
                    stim_mean = prev_stim_full[stim_start_ts:stim_end_ts].mean(dim=0)
                    I_prev_stim[input_start:input_end, :, :] = stim_mean.unsqueeze(0)
                if prev_choice_vec is not None:
                    I_prev_choice[input_start:input_end, :, :] = prev_choice_vec.unsqueeze(0)

            rule_cue = None
            if use_rule_cue:
                rule_idx_tensor = torch.full(
                    (bsz,), current_rule_idx, device=device, dtype=torch.long
                )
                rule_cue = torch.nn.functional.one_hot(
                    rule_idx_tensor, num_classes=len(rule_list)
                ).float()

            x_full = _pack_inputs_spatial(
                x,
                I_prev_stim,
                I_prev_rew,
                I_prev_choice,
                rule_cue,
                n_channels,
                spatial_size=spatial_size,
            )

            output_states, neuron_states, feedback_states = model(
                x_full,
                output_state0=carry_output_state,
                neuron_state0=carry_neuron_state,
                feedback_state0=carry_feedback_state,
            )

            # Carry final-timestep states into next trial (mirrors training)
            carry_output_state = [out[-1].detach() for out in output_states]
            carry_neuron_state = [
                [ct[-1].detach() for ct in area_ns] for area_ns in neuron_states
            ]
            carry_feedback_state = [
                fb[-1].detach() if fb is not None else None for fb in feedback_states
            ]

            if split_output_areas:
                sensory_out = output_states[0].mean(dim=(-2, -1))
                pfc_out = output_states[-1].mean(dim=(-2, -1))
                y = sensory_out[:, :, :n_out]
                y_rule = pfc_out[:, :, :n_out_rule]
            else:
                last_out = output_states[-1].mean(dim=(-2, -1))
                y = last_out[:, :, :n_out]
                y_rule = last_out[:, :, n_out : n_out + n_out_rule]

            choice_prob = y[resp_start_ts:resp_end_ts].mean(dim=0)
            choice_idx = choice_prob.argmax(dim=-1)
            target_prob = yhat[resp_start_ts:resp_end_ts].mean(dim=0)
            target_idx = target_prob.argmax(dim=-1)
            correct = (choice_idx == target_idx).float()

            rule_pred = (
                y_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(dim=-1)
            )
            rule_tgt = (
                yhat_rule[rule_start_ts:rule_end_ts].mean(dim=0).argmax(dim=-1)
            )
            correct_r = (rule_pred == rule_tgt).float()

            correct_choice.append(float(correct.mean().item()))
            correct_rule.append(float(correct_r.mean().item()))

            if use_history:
                prev_stim_full = x.detach()
                prev_choice_vec = torch.nn.functional.one_hot(
                    choice_idx, num_classes=n_out
                ).float()
                last_rew_vec = (choice_idx == target_idx).detach()

            if tr in switches:
                current_rule_idx = 1 - current_rule_idx

    return float(sum(correct_choice) / len(correct_choice)), float(
        sum(correct_rule) / len(correct_rule)
    )


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1 and sys.argv[1] == "--simple":
#         train_wcst_simple_rnn(num_blocks=1, num_batches_per_block=50, print_every=10)
#     else:
#         train_wcst()