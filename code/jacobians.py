"""
Jacobians of neural nets, with a flexible choice of which neurons count as
inputs and which as outputs, and whether to differentiate w.r.t. inputs or
parameters -- the input Jacobian J_{y<-x} and the parameter Jacobian J_{y<-W}.

A "system" here is any subgraph of a computation with chosen input and
output nodes: whatever tensors you hand to `jacobian` -- the raw model
input, its parameters, or any intermediate activation grabbed with
`capture`. Concatenating several nodes into one list of outputs (or
inputs) builds a "combined" system, e.g. two nets sharing an input, or a
layer's output plus a loss.

Core primitive
---------------
    jacobian(outputs, inputs) -> (n, d) tensor

    outputs, inputs: a tensor, or a list of tensors, that are part of the
    same autograd graph (`inputs` need not be leaves -- an intermediate
    activation works fine, as does a `nn.Parameter`). Each is flattened and
    concatenated, so the result is the n x d matrix d(outputs)/d(inputs)
    regardless of whether "inputs" means the raw input x, a hidden layer's
    activation, or a set of parameters.

Convenience wrappers
---------------------
    capture(model, names)        -- context manager grabbing named
                                     submodules' forward outputs, so they
                                     can be used as Jacobian inputs/outputs.
    select_params(model, names)  -- select nn.Parameters by dotted name or
                                     prefix, for parameter Jacobians.
    input_jacobian(model, x, ...)      -- J_{y<-x}
    parameter_jacobian(model, x, ...)  -- J_{y<-W}
"""

import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Sequence, Union

TensorOrList = Union[Tensor, Sequence[Tensor]]


def _as_list(t: TensorOrList) -> List[Tensor]:
    return [t] if isinstance(t, Tensor) else list(t)


def _flatten_cat(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def jacobian(outputs: TensorOrList, inputs: TensorOrList,
             create_graph: bool = False) -> Tensor:
    """The n x d Jacobian d(outputs)/d(inputs).

    `outputs` and `inputs` are each a tensor or list of tensors, all part of
    one autograd graph (a single forward pass). They are flattened and
    concatenated in the given order to form the n output coordinates and d
    input coordinates. `inputs` need not be leaves: an intermediate
    activation, or a parameter, both work.

    One batched backward pass computes the whole matrix (grad_outputs is the
    n x n identity), with a per-row fallback for ops that don't support
    vmapped backward (e.g. some RNN/cuDNN kernels).
    """
    outs = _as_list(outputs)
    ins = _as_list(inputs)
    y = _flatten_cat(outs)
    n = y.numel()
    if n == 0:
        return y.new_zeros(0, sum(i.numel() for i in ins))
    eye = torch.eye(n, dtype=y.dtype, device=y.device)
    try:
        grads = torch.autograd.grad(
            y, ins, grad_outputs=eye, is_grads_batched=True,
            retain_graph=True, create_graph=create_graph, allow_unused=True,
        )
    except RuntimeError:
        grads = _jacobian_rowwise(y, ins, n, create_graph)
    cols = []
    for g, inp in zip(grads, ins):
        if g is None:
            g = y.new_zeros(n, inp.numel())
        else:
            g = g.reshape(n, -1)
        cols.append(g)
    return torch.cat(cols, dim=1)


def _jacobian_rowwise(y: Tensor, ins: List[Tensor], n: int,
                       create_graph: bool) -> List[Optional[Tensor]]:
    """Fallback: one backward pass per output row, stacked into the batch
    dimension `is_grads_batched` would otherwise have produced."""
    rows: List[List[Optional[Tensor]]] = [[] for _ in ins]
    for i in range(n):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1.0
        grads = torch.autograd.grad(
            y, ins, grad_outputs=grad_outputs,
            retain_graph=True, create_graph=create_graph, allow_unused=True,
        )
        for k, g in enumerate(grads):
            rows[k].append(g)
    out = []
    for k, inp in enumerate(ins):
        if all(g is None for g in rows[k]):
            out.append(None)
        else:
            filled = [g if g is not None else torch.zeros_like(inp) for g in rows[k]]
            out.append(torch.stack(filled, dim=0))
    return out


class capture:
    """Context manager that hooks named submodules of `model`, recording
    their forward-pass output tensors (still attached to the autograd
    graph) in `self.acts`, keyed by dotted name as in
    `model.named_modules()`.

    Use to pick out an arbitrary neuron/layer as a Jacobian input or output.

        with capture(model, ["layer2", "layer4"]) as cap:
            model(x)
        J = jacobian(cap["layer4"], cap["layer2"])
    """

    def __init__(self, model: nn.Module, names: Sequence[str]):
        self.model = model
        self.names = list(dict.fromkeys(names))  # dedupe, keep order
        self.acts: Dict[str, Tensor] = {}
        self._handles = []

    def __getitem__(self, name: str) -> Tensor:
        return self.acts[name]

    def __enter__(self) -> "capture":
        modules = dict(self.model.named_modules())
        for name in self.names:
            if name not in modules:
                raise KeyError(
                    f"no submodule named {name!r}; "
                    f"available: {list(modules)}"
                )

            def hook(_module, _inp, out, name=name):
                self.acts[name] = out

            self._handles.append(modules[name].register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def select_params(model: nn.Module, names: Sequence[str]) -> List[nn.Parameter]:
    """Select parameters of `model` by dotted name (as in
    `model.named_parameters()`), e.g. "layer3.weight", or by prefix, e.g.
    "layer3" for all of that submodule's parameters. Order follows `names`,
    then `named_parameters()` order within a prefix match.
    """
    params = dict(model.named_parameters())
    out: List[nn.Parameter] = []
    for name in names:
        if name in params:
            out.append(params[name])
            continue
        prefix = name + "."
        matched = [p for k, p in params.items() if k.startswith(prefix)]
        if not matched:
            raise KeyError(f"no parameter matches {name!r}; available: {list(params)}")
        out.extend(matched)
    return out


def input_jacobian(model: nn.Module, x: Tensor,
                    output_names: Optional[Sequence[str]] = None,
                    input_names: Optional[Sequence[str]] = None,
                    **forward_kwargs) -> Tensor:
    """The input Jacobian J_{y<-x} = grad_x y(x, W).

    output_names / input_names: dotted submodule names (as in
    `model.named_modules()`) whose forward output to treat as the system's
    outputs / inputs; several names are concatenated. `None` means the raw
    model output / raw model input `x` respectively.

    Runs one forward pass through `model` and returns the resulting
    n x d matrix.
    """
    x = x.detach().clone().requires_grad_(True)
    watch = list(dict.fromkeys((output_names or []) + (input_names or [])))
    with capture(model, watch) as cap:
        y = model(x, **forward_kwargs)
    outs = [cap[n] for n in output_names] if output_names else [y]
    ins = [cap[n] for n in input_names] if input_names else [x]
    return jacobian(outs, ins)


def parameter_jacobian(model: nn.Module, x: Tensor,
                        output_names: Optional[Sequence[str]] = None,
                        param_names: Optional[Sequence[str]] = None,
                        **forward_kwargs) -> Tensor:
    """The parameter Jacobian J_{y<-W} = grad_W y(x, W).

    output_names: as in `input_jacobian`; `None` means the raw model output.
    param_names: dotted parameter or submodule names (as in
    `model.named_parameters()`); `None` means every parameter in `model`.
    """
    with capture(model, output_names or []) as cap:
        y = model(x, **forward_kwargs)
    outs = [cap[n] for n in output_names] if output_names else [y]
    params = select_params(model, param_names) if param_names else list(model.parameters())
    return jacobian(outs, params)


# ---------------------------------------------------------------------------
# Smoke tests / worked examples from the paper.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # --- (1) input_jacobian matches finite differences on a small relu MLP.
    mlp = nn.Sequential(
        nn.Linear(4, 6), nn.ReLU(),
        nn.Linear(6, 5), nn.ReLU(),
        nn.Linear(5, 3),
    ).double()
    x0 = torch.randn(4, dtype=torch.float64)

    J = input_jacobian(mlp, x0)
    eps = 1e-6
    J_fd = torch.zeros(3, 4, dtype=torch.float64)
    with torch.no_grad():
        y0 = mlp(x0)
        for j in range(4):
            dx = x0.clone()
            dx[j] += eps
            J_fd[:, j] = (mlp(dx) - y0) / eps
    err = (J - J_fd).abs().max().item()
    print(f"input_jacobian vs finite differences: max abs err = {err:.2e}")
    assert err < 1e-5

    # --- (2) parameter_jacobian: check shape against total parameter count,
    # and cross-check one column against a manual backward pass.
    n_params = sum(p.numel() for p in mlp.parameters())
    Jw = parameter_jacobian(mlp, x0)
    print(f"parameter_jacobian shape = {tuple(Jw.shape)} (n_params={n_params})")
    assert Jw.shape == (3, n_params)

    mlp.zero_grad()
    y = mlp(x0)
    y[1].backward()
    manual = torch.cat([p.grad.reshape(-1) for p in mlp.parameters()])
    err = (Jw[1] - manual).abs().max().item()
    print(f"row 1 of parameter_jacobian vs manual backward: max abs err = {err:.2e}")
    assert err < 1e-10

    # --- (3) Jacobian to/from a named intermediate layer, via `capture`.
    with capture(mlp, ["2"]) as cap:  # index "2" = second nn.Linear in the Sequential
        y = mlp(x0)
    J_layer_to_out = jacobian(y, cap["2"])
    print(f"J_{{y<-a_l}} shape = {tuple(J_layer_to_out.shape)}")
    assert J_layer_to_out.shape == (3, 5)  # module "2" is Linear(6, 5)

    # --- (4) Two nets sharing an input: input Jacobian is dense (cohesive);
    # parameter Jacobian is block diagonal (cohesion zero), since the nets
    # don't share parameters. Kept small (plain linear layers, no hidden
    # layer) just so the printed matrices stay legible.
    net1 = nn.Linear(3, 1).double()
    net2 = nn.Linear(3, 1).double()
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    y1, y2 = net1(x), net2(x)

    J_input = jacobian([y1, y2], x)
    params = list(net1.parameters()) + list(net2.parameters())
    J_param = jacobian([y1, y2], params)

    from jacobian_measures import cohesion
    coh_input = cohesion(J_input).item()
    coh_param = cohesion(J_param).item()
    print(f"cohesion(J_{{y<-x}}) = {coh_input:.4f} (dense: nets share the input)")
    print(f"cohesion(J_{{y<-W}}) = {coh_param:.4f} (=0: independent parameters)")
    assert coh_input > 0.05
    assert coh_param < 1e-6

    print("\nAll jacobians.py smoke tests passed.")
