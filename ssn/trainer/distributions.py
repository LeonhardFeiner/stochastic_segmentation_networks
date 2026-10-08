import torch.distributions as td
from torch.distributions import Transform, constraints
import torch
import torch.nn.functional as F
from typing import Tuple


def pad(x, axis):
    shape = x.shape[:axis] + (1,) + x.shape[axis + 1 :]
    zeros = x.new_zeros(shape)
    return torch.concat((zeros, x), dim=axis)


def pad_epsilon(x, axis, epsilon=1e-6):
    shape = x.shape[:axis] + (1,) + x.shape[axis + 1 :]
    zeros = x.new_full(shape, epsilon)
    return torch.concat((zeros, x), dim=axis)


def split(x, axis):
    sections = [1, x.shape[axis] - 1]
    log_normalization, x = torch.split(x, split_size_or_sections=sections, dim=axis)
    return x, log_normalization


class SoftmaxTransform(Transform):
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, axis=-4, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.axis = axis

    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(self.axis, True)[0]).exp()
        return probs / probs.sum(self.axis, True)

    def _inverse(self, y):
        probs = y
        return probs.log()

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape


class LogSoftmaxTransform(Transform):
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, axis=-4, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.axis = axis

    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        return F.log_softmax(x, axis=self.axis)
        # logprobs = x
        # probs = (logprobs - logprobs.max(self.axis, True)[0]).exp()
        # return probs / probs.sum(self.axis, True)

    def _inverse(self, y):
        return y
        # probs = y
        # return probs.log()

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape


class AxisSimplex(constraints.Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    def __init__(self, axis) -> None:
        super().__init__()
        self.axis

    event_dim = 1

    def check(self, value):
        return torch.all(value >= 0, dim=-1) & ((value.sum(self.axis) - 1).abs() < 1e-6)


class CenteredSoftmaxTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """
    domain = constraints.real
    # codomain = constraints.simplex
    bijective = True
    sign = +1

    def __init__(self, axis=-4, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.axis = axis
        self.codomain = AxisSimplex(axis)

    def __eq__(self, other):
        return isinstance(other, CenteredSoftmaxTransform)

    def _pad(self, x):
        shape = x.shape[: self.axis] + (1,) + x.shape[self.axis + 1 :]
        zeros = x.new_zeros(shape)
        return torch.concat((zeros, x), dim=self.axis)

    def _split(self, x):
        sections = [1, x.shape[self.axis] - 1]
        log_normalization, x = torch.split(
            x, split_size_or_sections=sections, dim=self.axis
        )
        return x, log_normalization

    def _call(self, x):
        logprobs = self._pad(x)
        return torch.softmax(logprobs, axis=self.axis)
        # probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        # return probs / probs.sum(-1, True)

    def _inverse(self, y):
        x = torch.log(y)
        x, log_normalization = self._split(x)
        return x - log_normalization

        # finfo = torch.finfo(y.dtype)
        # y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        # return y.log() - (-y).log1p()

    # def log_abs_det_jacobian(self, x, y):
    #     return -F.softplus(-x) - F.softplus(x)

    def log_abs_det_jacobian(self, x, y):
        np1 = torch.tensor(1 + x.shape[self.axis], dtype=x.dtype)

        ret_val = (
            0.5 * torch.log(np1)
            + torch.sum(x, axis=self.axis)
            - np1 * F.softplus(torch.logsumexp(x, axis=self.axis))
        )
        return ret_val

        # return -(0.5 * torch.log(np1) + torch.sum(torch.log(y), axis=1))

    # def _inverse_log_det_jacobian(self, y):
    #     np1 = y.shape[-1].type(y.dtype)
    #     return -(0.5 * torch.log(np1) + torch.sum(torch.log(y), axis=-1))

    # def _forward_log_det_jacobian(self, x):
    #     np1 = (1 + x.shape[-1]).type(x.dtype)

    #     return (
    #         0.5 * torch.log(np1)
    #         + torch.sum(x, axis=-1)
    #         - np1 * F.softplus(torch.logsumexp(x, axis=-1))
    #     )

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        batch, features, *other = shape
        return (batch, features + 1, *other)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        batch, features, *other = shape
        return (batch, features - 1, *other)


class CenteredLogSoftmaxTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """
    domain = constraints.real
    codomain = constraints.less_than(0)
    bijective = True
    sign = +1

    def __init__(self, axis=-4, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.axis = axis

    def __eq__(self, other):
        return isinstance(other, CenteredSoftmaxTransform)

    def _pad(self, x):
        shape = x.shape[: self.axis] + (1,) + x.shape[self.axis + 1 :]
        zeros = x.new_zeros(shape)
        return torch.concat((zeros, x), dim=self.axis)

    def _split(self, x):
        sections = [1, x.shape[self.axis] - 1]
        log_normalization, x = torch.split(
            x, split_size_or_sections=sections, dim=self.axis
        )
        return x, log_normalization

    def _call(self, x):
        logprobs = self._pad(x)
        return torch.log_softmax(logprobs, axis=self.axis)
        # probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        # return probs / probs.sum(-1, True)

    def _inverse(self, y):
        # x = torch.log(y)
        x, log_normalization = self._split(y)
        return x - log_normalization

        # finfo = torch.finfo(y.dtype)
        # y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        # return y.log() - (-y).log1p()

    # def log_abs_det_jacobian(self, x, y):
    #     return -F.softplus(-x) - F.softplus(x)

    def log_abs_det_jacobian(self, x, y):
        np1 = torch.tensor(1 + x.shape[self.axis], dtype=x.dtype)

        ret_val = (
            0.5 * torch.log(np1)
            + torch.sum(x, axis=self.axis)
            - np1 * F.softplus(torch.logsumexp(x, axis=self.axis))
        )
        return ret_val

        # return -(0.5 * torch.log(np1) + torch.sum(torch.log(y), axis=1))

    # def _inverse_log_det_jacobian(self, y):
    #     np1 = y.shape[-1].type(y.dtype)
    #     return -(0.5 * torch.log(np1) + torch.sum(torch.log(y), axis=-1))

    # def _forward_log_det_jacobian(self, x):
    #     np1 = (1 + x.shape[-1]).type(x.dtype)

    #     return (
    #         0.5 * torch.log(np1)
    #         + torch.sum(x, axis=-1)
    #         - np1 * F.softplus(torch.logsumexp(x, axis=-1))
    #     )

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        batch, features, *other = shape
        return (batch, features + 1, *other)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        batch, features, *other = shape
        return (batch, features - 1, *other)


class PaddingTransform(Transform):

    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = False
    sign = +1

    def __init__(self, axis=-4, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.axis = axis

    def __eq__(self, other):
        return isinstance(other, CenteredSoftmaxTransform)

    def _pad(self, x):
        shape = x.shape[: self.axis] + (1,) + x.shape[self.axis + 1 :]
        zeros = x.new_zeros(shape)
        return torch.concat((zeros, x), dim=self.axis)

    def _split(self, x):
        # print(x.shape[self.axis], x.shape)
        sections = [1, x.shape[self.axis] - 1]
        log_normalization, x = torch.split(
            x, split_size_or_sections=sections, dim=self.axis
        )
        return x, log_normalization

    def _call(self, x):
        # print(x.shape[self.axis], x.shape)
        logprobs = self._pad(x)
        return logprobs

    def _inverse(self, y):
        x, log_normalization = self._split(y)
        assert torch.all(log_normalization == 0)
        return x

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        batch, features, *other = shape
        return (batch, features + 1, *other)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        batch, features, *other = shape
        return (batch, features - 1, *other)


# from torch.distributions import log_normal


# def _clipped_sigmoid(x):
#     finfo = torch.finfo(x.dtype)
#     return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


# class SoftmaxCentered(bijector.AutoCompositeTensorBijector):
#   """Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.
#   To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
#   bijection, the forward transformation appends a value to the input and the
#   inverse removes this coordinate. The appended coordinate represents a pivot,
#   e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
#   coordinate.
#   Example Use:
#   ```python
#   bijector.SoftmaxCentered().forward(tf.log([2, 3, 4]))
#   # Result: [0.2, 0.3, 0.4, 0.1]
#   # Extra result: 0.1
#   bijector.SoftmaxCentered().inverse([0.2, 0.3, 0.4, 0.1])
#   # Result: tf.log([2, 3, 4])
#   # Extra coordinate removed.
#   ```
#   At first blush it may seem like the [Invariance of domain](
#   https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
#   implementation is not a bijection. However, the appended dimension
#   makes the (forward) image non-open and the theorem does not directly apply.
#   """

#   def __init__(self,
#                validate_args=False,
#                name='softmax_centered'):
#     parameters = dict(locals())
#     with tf.name_scope(name) as name:
#       self._pad = pad_lib.Pad( v=validate_args)
#       super(SoftmaxCentered, self).__init__(
#           forward_min_event_ndims=1,
#           validate_args=validate_args,
#           parameters=parameters,
#           name=name)

#   @classmethod
#   def _parameter_properties(cls, dtype):
#     return dict()

#   def _forward_event_shape(self, input_shape):
#     return self._pad.forward_event_shape(input_shape)

#   def _forward_event_shape_tensor(self, input_shape):
#     return self._pad.forward_event_shape_tensor(input_shape)

#   def _inverse_event_shape(self, output_shape):
#     return self._pad.inverse_event_shape(output_shape)

#   def _inverse_event_shape_tensor(self, output_shape):
#     return self._pad.inverse_event_shape_tensor(output_shape)

#   def _forward(self, x):
#     return tf.math.softmax(self._pad.forward(x))

#   def _inverse(self, y):
#     # To derive the inverse mapping note that:
#     #   y[i] = exp(x[i]) / normalization
#     # and
#     #   y[end] = 1 / normalization.
#     # Thus:
#     # x[i] = log(exp(x[i])) - log(y[end]) - log(normalization)
#     #      = log(exp(x[i])/normalization) - log(y[end])
#     #      = log(y[i]) - log(y[end])

#     # Do this first to make sure CSE catches that it'll happen again in
#     # _inverse_log_det_jacobian.

# assertions = []
# if self.validate_args:
#   assertions.append(assert_util.assert_near(
#       tf.reduce_sum(y, axis=-1),
#       tf.ones([], y.dtype),
#       2. * np.finfo(dtype_util.as_numpy_dtype(y.dtype)).eps,
#       message='Last dimension of `y` must sum to `1`.'))
#   assertions.append(assert_util.assert_less_equal(
#       y, tf.ones([], y.dtype),
#       message='Elements of `y` must be less than or equal to `1`.'))
#   assertions.append(assert_util.assert_non_negative(
#       y, message='Elements of `y` must be non-negative.'))

#     with tf.control_dependencies(assertions):
#       x = tf.math.log(y)
#       x, log_normalization = tf.split(x, num_or_size_splits=[-1, 1], axis=-1)
#     return x - log_normalization

#   def _inverse_log_det_jacobian(self, y):
#     # Let B be the forward map defined by the bijector. Consider the map
#     # F : R^n -> R^n where the image of B in R^{n+1} is restricted to the first
#     # n coordinates.
#     #
#     # Claim: det{ dF(X)/dX } = prod(Y) where Y = B(X).
#     # Proof: WLOG, in vector notation:
#     #     X = log(Y[:-1]) - log(Y[-1])
#     #   where,
#     #     Y[-1] = 1 - sum(Y[:-1]).
#     #   We have:
#     #     det{dF} = 1 / det{ dX/dF(X} }                                      (1)
#     #             = 1 / det{ diag(1 / Y[:-1]) + 1 / Y[-1] }
#     #             = 1 / det{ inv{ diag(Y[:-1]) - Y[:-1]' Y[:-1] } }
#     #             = det{ diag(Y[:-1]) - Y[:-1]' Y[:-1] }
#     #             = (1 + Y[:-1]' inv{diag(Y[:-1])} Y[:-1]) det{diag(Y[:-1])} (2)
#     #             = Y[-1] prod(Y[:-1])
#     #             = prod(Y)
#     #
#     # Let P be the image of R^n under F. Define the lift G, from P to R^{n+1},
#     # which appends the last coordinate, Y[-1] := 1 - \sum_k Y_k. G is linear,
#     # so its Jacobian is constant.
#     #
#     # The differential of G, DG, is eye(n) with a row of -1s appended to the
#     # bottom. To compute the Jacobian sqrt{det{(DG)^T(DG)}}, one can see that
#     # (DG)^T(DG) = A + eye(n), where A is the n x n matrix of 1s. This has
#     # eigenvalues (n + 1, 1,...,1), so the determinant is (n + 1). Hence, the
#     # Jacobian of G is sqrt{n + 1} everywhere.
#     #
#     # Putting it all together, the forward bijective map B can be written as
#     # B(X) = G(F(X)) and has Jacobian sqrt{n + 1} * prod(F(X)).
#     #
#     # (1) - https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
#     #       or by noting that det{ dX/dY } = 1 / det{ dY/dX } from Bijector
#     #       docstring "Tip".
#     # (2) - https://en.wikipedia.org/wiki/Matrix_determinant_lemma
#     np1 = ps.cast(ps.shape(y)[-1], dtype=y.dtype)
#     return -(0.5 * ps.log(np1) +
#              tf.reduce_sum(tf.math.log(y), axis=-1))

#   def _forward_log_det_jacobian(self, x):
#     # This code is similar to tf.math.log_softmax but different because we have
#     # an implicit zero column to handle. I.e., instead of:
#     #   reduce_sum(logits - reduce_sum(exp(logits), dim))
#     # we must do:
#     #   log_normalization = 1 + reduce_sum(exp(logits))
#     #   -log_normalization + reduce_sum(logits - log_normalization)
#     np1 = ps.cast(1 + ps.shape(x)[-1], dtype=x.dtype)
#     return (0.5 * ps.log(np1) +
#             tf.reduce_sum(x, axis=-1) -
#             np1 * tf.math.softplus(tf.reduce_logsumexp(x, axis=-1)))
