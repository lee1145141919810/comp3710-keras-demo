import torch

from part4_recognition.gan.train import spatial_moment_loss


def test_collapsed_mean_is_penalised_and_receives_finite_gradients():
    # Two different structures with the same batch mean as a collapsed generator.
    real = torch.zeros(4, 1, 32, 32)
    real[:2, :, :16] = 1
    real[2:, :, 16:] = 1
    collapsed = real.mean(0, keepdim=True).repeat(4, 1, 1, 1).requires_grad_()
    loss = spatial_moment_loss(collapsed, real)
    assert loss > spatial_moment_loss(real.flip(0), real)
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_real_reference_is_not_differentiated():
    real = torch.randn(4, 1, 32, 32, requires_grad=True)
    fake = torch.randn_like(real, requires_grad=True)
    spatial_moment_loss(fake, real).backward()
    assert real.grad is None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0
