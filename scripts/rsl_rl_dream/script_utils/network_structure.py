"""Network-structure export helpers for RSL-RL runner scripts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch


SUPPORTED_GRAPH_FORMATS = ("png", "pdf", "svg")
SUPPORTED_GRAPH_DIRECTIONS = ("TB", "LR", "BT", "RL")


class ForwardAllGraphWrapper(torch.nn.Module):
    """Wrap actor models so torchview traces policy and auxiliary loss outputs in one graph."""

    output_keys = (
        "action",
        "est_explicit",
        "est_implicit",
        "implicit_latent",
        "mean_explicit",
        "logvar_explicit",
        "mean_latent",
        "logvar_latent",
    )

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, obs: Any) -> tuple[torch.Tensor, ...]:
        outputs = self.model.forward_all(obs)
        if not isinstance(outputs, dict):
            raise TypeError("forward_all() must return a dict for network-structure export.")
        return tuple(outputs[key] for key in self.output_keys if key in outputs and outputs[key] is not None)


def export_torchview_graph(
    model: torch.nn.Module,
    output_dir: str | Path,
    filename: str = "network_structure",
    *,
    output_format: str = "png",
    input_data: Any | None = None,
    input_size: Any | None = None,
    graph_name: str | None = None,
    device: str | torch.device | None = "cpu",
    depth: int | float = 6,
    expand_nested: bool = True,
    graph_dir: str = "TB",
    dpi: int | None = None,
    hide_module_functions: bool = True,
    hide_inner_tensors: bool = True,
    strict: bool = False,
    clone_model: bool = True,
    **draw_graph_kwargs: Any,
) -> Path:
    """Export a torchview graph for a PyTorch module."""
    if input_data is None and input_size is None:
        raise ValueError("Either input_data or input_size must be provided for torchview.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}.")
    output_format = output_format.lower().lstrip(".")
    if output_format not in SUPPORTED_GRAPH_FORMATS:
        raise ValueError(f"output_format must be one of {SUPPORTED_GRAPH_FORMATS}, got {output_format!r}.")
    graph_dir = graph_dir.upper()
    if graph_dir not in SUPPORTED_GRAPH_DIRECTIONS:
        raise ValueError(f"graph_dir must be one of {SUPPORTED_GRAPH_DIRECTIONS}, got {graph_dir!r}.")

    try:
        from torchview import draw_graph
    except ImportError as exc:
        raise ImportError("torchview is required to export a network graph. Install it with `pip install torchview`.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename_path = Path(filename)
    file_stem = filename_path.stem if filename_path.suffix else filename_path.name
    if not file_stem:
        raise ValueError("filename must not be empty.")

    graph_model = copy.deepcopy(model) if clone_model else model
    # Trace a disposable CPU copy; never change the live training actor's mode
    # or recurrent state. The runner exporter supplies a one-item observation.
    graph_model = graph_model.to(device) if device is not None else graph_model
    if clone_model and hasattr(graph_model, "reset"):
        graph_model.reset()
    graph_name = graph_name or file_stem

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model_graph = draw_graph(
            graph_model,
            input_data=input_data,
            input_size=input_size,
            graph_name=graph_name,
            depth=depth,
            device=device,
            mode="eval",
            strict=strict,
            expand_nested=expand_nested,
            graph_dir=graph_dir,
            hide_module_functions=hide_module_functions,
            hide_inner_tensors=hide_inner_tensors,
            save_graph=False,
            filename=file_stem,
            directory=str(output_path),
            **draw_graph_kwargs,
        )
        if dpi is not None:
            model_graph.visual_graph.graph_attr["dpi"] = str(dpi)
        rendered_path = model_graph.visual_graph.render(format=output_format, cleanup=True)

    return Path(rendered_path)


def export_torchview_png(
    model: torch.nn.Module,
    output_dir: str | Path,
    filename: str = "network_structure",
    **kwargs: Any,
) -> Path:
    """Export a torchview PNG graph for a PyTorch module."""
    kwargs.pop("output_format", None)
    return export_torchview_graph(model, output_dir, filename, output_format="png", **kwargs)


def get_runner_policy_module(runner: Any) -> torch.nn.Module:
    """Return the loaded actor/policy module from different RSL-RL runner variants."""
    alg = getattr(runner, "alg", None)
    if alg is not None and hasattr(alg, "get_policy"):
        return alg.get_policy()
    if alg is not None and hasattr(alg, "policy"):
        return alg.policy
    if alg is not None and hasattr(alg, "actor_critic"):
        return alg.actor_critic
    raise AttributeError("Could not locate a policy module on runner.alg.")


def get_actor_class_name(agent_cfg: Any, policy_module: torch.nn.Module) -> str:
    """Read the configured actor class name from the current runner cfg."""
    try:
        runner_cfg = agent_cfg.to_dict()
    except Exception:
        runner_cfg = {}
    actor_cfg = runner_cfg.get("actor") or runner_cfg.get("policy") or {}
    if isinstance(actor_cfg, dict):
        class_name = actor_cfg.get("class_name")
        if isinstance(class_name, str) and class_name:
            return class_name
    return type(policy_module).__name__


def safe_filename_stem(value: str) -> str:
    """Return a filesystem-friendly filename stem."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_") or "policy"


def export_runner_network_structure(
    runner: Any,
    agent_cfg: Any,
    obs: Any,
    export_model_dir: str | Path,
    *,
    device: str | torch.device | None = "cpu",
    depth: int | float = 8,
    expand_nested: bool = True,
    output_format: str = "png",
    graph_dir: str = "TB",
    dpi: int | None = None,
) -> Path:
    """Export the runner policy network structure graph with the active runner cfg name."""
    policy_nn = get_runner_policy_module(runner)
    # RSL-RL observations are TensorDicts. Avoid tracing thousands of envs and
    # move them explicitly because torchview's generic recursion is not aware
    # of every TensorDict implementation.
    obs = obs[:1].clone().to(device) if hasattr(obs, "batch_size") else obs
    actor_class_name = get_actor_class_name(agent_cfg, policy_nn)
    graph_model = ForwardAllGraphWrapper(policy_nn) if hasattr(policy_nn, "forward_all") else policy_nn
    graph_name_suffix = " Full Network Structure" if hasattr(policy_nn, "forward_all") else " Network Structure"
    return export_torchview_graph(
        graph_model,
        export_model_dir,
        filename=f"{safe_filename_stem(actor_class_name)}_network_structure",
        output_format=output_format,
        input_data=(obs,),
        graph_name=f"{actor_class_name}{graph_name_suffix}",
        device=device,
        depth=depth,
        expand_nested=expand_nested,
        graph_dir=graph_dir,
        dpi=dpi,
    )
