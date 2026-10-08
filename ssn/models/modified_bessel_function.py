import torch
from torch.autograd import Function
from scipy.special import iv


class ModifiedBesselSecondKindFunction(Function):
    @staticmethod
    def forward(ctx, x, v):
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        y_np = iv(v_np, x_np, is_scaled=False)
        y = torch.from_numpy(y_np).to(x.device)
        ctx.save_for_backward(x, v)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, v = ctx.saved_tensors
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        vjp = lambda v_: iv(v_, x_np, is_scaled=True)
        grad_x_np = grad_output_np * vjp(v_np)
        grad_v_np = grad_output_np * iv(v_np, x_np, derivative=1, is_scaled=True)
        grad_x = torch.from_numpy(grad_x_np).to(x.device)
        grad_v = torch.from_numpy(grad_v_np).to(v.device)
        return grad_x, grad_v


def modified_bessel_second_kind(x, v):
    """
    Modified Bessel function of the first kind of real order.

    Parameters
    ----------
    x : tensor of float or complex
        Argument.
    v : tensor
        Order. If `x` is of real type and negative, `v` must be integer
        valued.

    Returns
    -------
    out : tensor
        Values of the modified Bessel function.

    Notes
    -----
    For real `x` and :math:`v \in [-50, 50]`, the evaluation is carried out
    using Temme's method [1]_.  For larger orders, uniform asymptotic
    expansions are applied.

    For complex `x` and positive `v`, the AMOS [2]_ `zbesi` routine is
    called. It uses a power series for small `x`, the asymptotic expansion
    for large `abs(x)`, the Miller algorithm normalized by the Wronskian
    and a Neumann series for intermediate magnitudes, and the uniform
    asymptotic expansions for :math:`I_v(x)` and :math:`J_v(x)` for large
    orders. Backward recurrence is used to generate sequences or reduce
    orders when necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(x)

    (valid when the real part of `x` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(x) = I_v(x) + \frac{2}{\pi} \sin(\pi v) K_v(x)

    is used, where :math:`K_v(x)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.


    References
    ----------
    .. [1] Temme, Journal of Computational Physics, vol 21, 343 (1976)
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
        of a Complex Argument and Nonnegative Order",
        http://netlib.org/amos/
    """
    return ModifiedBesselSecondKindFunction.apply(x, v)
