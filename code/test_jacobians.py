"""
Tests for jacobians.py: the input/parameter Jacobian utilities.
"""
import pytest
import torch
from torch import nn

from jacobians import (
    capture,
    input_jacobian,
    jacobian,
    parameter_jacobian,
    select_params,
)
from jacobian_measures import cohesion

DTYPE = torch.float64


def gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def relu_mlp(dims, generator=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers).to(DTYPE)
    if generator is not None:
        with torch.no_grad():
            for p in net.parameters():
                p.copy_(torch.randn(p.shape, generator=generator, dtype=DTYPE))
    return net


def finite_diff_jacobian(f, x, eps=1e-6):
    with torch.no_grad():
        y0 = f(x)
        d = x.numel()
        n = y0.numel()
        J = torch.zeros(n, d, dtype=x.dtype)
        for j in range(d):
            dx = x.clone()
            dx.view(-1)[j] += eps
            J[:, j] = (f(dx) - y0).reshape(-1) / eps
        return J


# ---------------------------------------------------------------------------
# Core primitive: jacobian(outputs, inputs)
# ---------------------------------------------------------------------------

class TestJacobianPrimitive:

    @pytest.mark.parametrize("n,d", [(1, 1), (3, 5), (5, 3), (4, 4)])
    def test_linear_map_recovers_matrix(self, n, d):
        g = gen(0)
        W = torch.randn(n, d, generator=g, dtype=DTYPE)
        x = torch.randn(d, dtype=DTYPE, requires_grad=True)
        y = W @ x
        J = jacobian(y, x)
        assert torch.allclose(J, W, atol=1e-10)

    def test_matches_finite_differences_on_relu_mlp(self):
        g = gen(1)
        net = relu_mlp([4, 6, 5, 3], g)
        x = torch.randn(4, dtype=DTYPE, requires_grad=True)
        J = jacobian(net(x), x)
        J_fd = finite_diff_jacobian(lambda z: net(z.detach()), x, eps=1e-6)
        assert torch.allclose(J, J_fd, atol=1e-5)

    def test_list_of_outputs_and_inputs_concatenates(self):
        g = gen(2)
        x1 = torch.randn(3, dtype=DTYPE, requires_grad=True)
        x2 = torch.randn(2, dtype=DTYPE, requires_grad=True)
        A = torch.randn(4, 3, generator=g, dtype=DTYPE)
        B = torch.randn(4, 2, generator=g, dtype=DTYPE)
        y1 = A @ x1
        y2 = B @ x2 + (x1 * 0).sum()  # y2 independent of x1
        J = jacobian([y1, y2], [x1, x2])
        expected = torch.zeros(8, 5, dtype=DTYPE)
        expected[:4, :3] = A
        expected[4:, 3:] = B
        assert torch.allclose(J, expected, atol=1e-10)

    def test_parameters_as_inputs_matches_manual_backward(self):
        g = gen(3)
        net = relu_mlp([3, 5, 2], g)
        x = torch.randn(3, dtype=DTYPE)
        params = list(net.parameters())
        y = net(x)
        Jw = jacobian(y, params)
        # cross-check row 0 against a fresh manual backward pass
        net.zero_grad()
        y2 = net(x)
        y2[0].backward()
        manual = torch.cat([p.grad.reshape(-1) for p in net.parameters()])
        assert torch.allclose(Jw[0], manual, atol=1e-10)

    def test_unused_input_gives_zero_columns(self):
        x = torch.randn(3, dtype=DTYPE, requires_grad=True)
        unused = torch.randn(2, dtype=DTYPE, requires_grad=True)
        y = (x ** 2).sum().reshape(1)
        J = jacobian(y, [x, unused])
        assert torch.allclose(J[:, 3:], torch.zeros(1, 2, dtype=DTYPE))

    def test_rnn_input_jacobian_is_lower_triangular(self):
        """y_t depends only on x_1..x_t."""
        g = gen(4)
        cell = nn.RNNCell(1, 4).to(DTYPE)
        with torch.no_grad():
            for p in cell.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=DTYPE))
        readout = nn.Linear(4, 1).to(DTYPE)

        T = 6
        xs = torch.randn(T, dtype=DTYPE, requires_grad=True)
        h = torch.zeros(1, 4, dtype=DTYPE)
        ys = []
        for t in range(T):
            h = cell(xs[t].reshape(1, 1), h)
            ys.append(readout(h).reshape(()))
        y = torch.stack(ys)

        J = jacobian(y, xs)
        assert torch.allclose(J.triu(1), torch.zeros_like(J))
        assert (J.diagonal().abs() > 1e-8).all()

    def test_mlp_over_time_input_jacobian_is_diagonal(self):
        """No interaction across time -> diagonal Jacobian."""
        g = gen(5)
        net = relu_mlp([1, 4, 1], g)
        T = 5
        xs = torch.randn(T, dtype=DTYPE, requires_grad=True)
        ys = torch.stack([net(xs[t].reshape(1)).reshape(()) for t in range(T)])
        J = jacobian(ys, xs)
        off_diag = J - torch.diag(J.diagonal())
        assert torch.allclose(off_diag, torch.zeros_like(J), atol=1e-12)


# ---------------------------------------------------------------------------
# capture / select_params
# ---------------------------------------------------------------------------

class TestCaptureAndSelectParams:

    def test_capture_grabs_named_submodule_output(self):
        net = relu_mlp([3, 4, 2], gen(6))
        x = torch.randn(3, dtype=DTYPE)
        with capture(net, ["0"]) as cap:  # first Linear(3, 4)
            net(x)
        assert cap["0"].shape == (4,)

    def test_capture_unknown_name_raises(self):
        net = relu_mlp([3, 4, 2], gen(7))
        with pytest.raises(KeyError):
            with capture(net, ["not_a_module"]):
                net(torch.randn(3, dtype=DTYPE))

    def test_capture_hooks_removed_after_context(self):
        net = relu_mlp([3, 4, 2], gen(8))
        with capture(net, ["0"]) as cap:
            net(torch.randn(3, dtype=DTYPE))
        for m in net.modules():
            assert len(m._forward_hooks) == 0

    def test_select_params_exact_name(self):
        net = relu_mlp([3, 4, 2], gen(9))
        [w] = select_params(net, ["0.weight"])
        assert w is dict(net.named_parameters())["0.weight"]

    def test_select_params_prefix_matches_submodule(self):
        net = relu_mlp([3, 4, 2], gen(10))
        ps = select_params(net, ["0"])
        names = [n for n, _ in net.named_parameters() if n.startswith("0.")]
        assert len(ps) == len(names) == 2

    def test_select_params_unknown_raises(self):
        net = relu_mlp([3, 4, 2], gen(11))
        with pytest.raises(KeyError):
            select_params(net, ["nope"])


# ---------------------------------------------------------------------------
# input_jacobian / parameter_jacobian convenience wrappers
# ---------------------------------------------------------------------------

class TestConvenienceWrappers:

    def test_input_jacobian_default_matches_core_primitive(self):
        net = relu_mlp([4, 5, 3], gen(12))
        x = torch.randn(4, dtype=DTYPE)
        J1 = input_jacobian(net, x)
        x2 = x.clone().requires_grad_(True)
        J2 = jacobian(net(x2), x2)
        assert torch.allclose(J1, J2, atol=1e-10)

    def test_input_jacobian_intermediate_output_node(self):
        net = relu_mlp([4, 5, 3], gen(13))
        x = torch.randn(4, dtype=DTYPE)
        J = input_jacobian(net, x, output_names=["0"])  # first Linear(4,5)
        assert J.shape == (5, 4)
        w0 = dict(net.named_parameters())["0.weight"]
        assert torch.allclose(J, w0, atol=1e-10)  # pre-activation Jacobian == weight matrix

    def test_input_jacobian_intermediate_input_node(self):
        """J_{y<-a_l}: Jacobian from a hidden layer's activation to the
        output."""
        net = relu_mlp([4, 5, 3], gen(14))
        x = torch.randn(4, dtype=DTYPE)
        J = input_jacobian(net, x, input_names=["1"])  # ReLU output, 5-dim
        assert J.shape == (3, 5)

    def test_parameter_jacobian_default_covers_all_params(self):
        net = relu_mlp([4, 5, 3], gen(15))
        x = torch.randn(4, dtype=DTYPE)
        n_params = sum(p.numel() for p in net.parameters())
        J = parameter_jacobian(net, x)
        assert J.shape == (3, n_params)

    def test_parameter_jacobian_restricted_to_one_layer(self):
        net = relu_mlp([4, 5, 3], gen(16))
        x = torch.randn(4, dtype=DTYPE)
        J = parameter_jacobian(net, x, param_names=["0"])  # first Linear only
        n0 = sum(p.numel() for p in select_params(net, ["0"]))
        assert J.shape == (3, n0)

    def test_two_nets_shared_input_vs_independent_params(self):
        """Input Jacobian is dense (shared input), parameter Jacobian is
        block diagonal (cohesion 0, independent params)."""
        g = gen(17)
        net1 = nn.Linear(3, 1).to(DTYPE)
        net2 = nn.Linear(3, 1).to(DTYPE)
        with torch.no_grad():
            for net in (net1, net2):
                for p in net.parameters():
                    p.copy_(torch.randn(p.shape, generator=g, dtype=DTYPE))
        x = torch.randn(3, dtype=DTYPE, requires_grad=True)
        y1, y2 = net1(x), net2(x)

        J_in = jacobian([y1, y2], x)
        assert cohesion(J_in).item() > 0.05

        params = list(net1.parameters()) + list(net2.parameters())
        J_w = jacobian([y1, y2], params)
        assert cohesion(J_w).item() < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
