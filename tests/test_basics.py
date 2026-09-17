"""Fast unit tests (no downloads, CPU only):  python -m pytest tests -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from part1_dft.dft_torch import naive_dft_torch, square_wave_fourier_torch, to_numpy_complex  # noqa: E402
from part1_dft.square_wave_numpy import naive_dft, square_wave_fourier  # noqa: E402
from part3_cnn.dawnbench.data import augment  # noqa: E402
from part3_cnn.dawnbench.resnet import resnet18  # noqa: E402
from part4_recognition.gan.model import Discriminator, Generator  # noqa: E402
from part4_recognition.oasis_data import grey_levels_to_labels, labels_to_one_hot  # noqa: E402
from part4_recognition.unet.metrics import DiceAccumulator, soft_dice_loss  # noqa: E402
from part4_recognition.unet.model import UNet  # noqa: E402
from part4_recognition.vae.model import ConvVAE, vae_loss  # noqa: E402


# -- Part 1 ------------------------------------------------------------------------------------
def test_naive_dft_matches_fft():
    x = np.random.default_rng(0).standard_normal(64)
    assert np.allclose(naive_dft(x), np.fft.fft(x))


def test_torch_dft_matches_fft_in_chunks():
    x = torch.randn(96, dtype=torch.float64)
    X = to_numpy_complex(naive_dft_torch(x, chunk_elements=96 * 10))  # forces several row chunks
    assert np.allclose(X, np.fft.fft(x.numpy()))


def test_torch_fourier_series_matches_numpy():
    t = np.linspace(0, 1, 512, endpoint=False)
    expected = square_wave_fourier(t, 1.0, 7)
    got = square_wave_fourier_torch(torch.as_tensor(t), 1.0, 7).numpy()
    assert np.allclose(expected, got)


# -- Part 3.2 ----------------------------------------------------------------------------------
def test_resnet18_shape_and_size():
    model = resnet18()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 11_173_962


def test_augment_crops_flips_and_cuts():
    padded = torch.rand(8, 3, 40, 40)
    out = augment(padded, crop=32, cutout=8)
    assert out.shape == (8, 3, 32, 32)
    assert (out == 0).any(), "cutout should zero some pixels"


# -- Part 4 data -------------------------------------------------------------------------------
def test_grey_levels_to_labels_and_one_hot():
    mask = np.array([[0, 85], [170, 255]], dtype=np.uint8)
    labels = grey_levels_to_labels(mask)
    assert labels.tolist() == [[0, 1], [2, 3]]
    one_hot = labels_to_one_hot(torch.as_tensor(labels)[None])
    assert one_hot.shape == (1, 4, 2, 2)
    assert torch.equal(one_hot.argmax(1)[0], torch.as_tensor(labels))


# -- Part 4 UNet metrics -----------------------------------------------------------------------
def test_dice_perfect_and_half_overlap():
    target = torch.tensor([[[0, 0, 1, 1]]])  # [B=1, H=1, W=4]
    acc = DiceAccumulator(n_classes=2)
    acc.update(target.clone(), target)
    assert torch.allclose(acc.per_class(), torch.ones(2))
    assert acc.pixel_accuracy() == 1.0

    acc = DiceAccumulator(n_classes=2)
    acc.update(torch.tensor([[[0, 0, 0, 1]]]), target)  # class 1: |P∩G|=1, |P|=1, |G|=2 -> 2/3
    assert torch.allclose(acc.per_class(), torch.tensor([0.8, 2 / 3]), atol=1e-6)


def test_soft_dice_loss_zero_for_confident_correct_prediction():
    target = torch.randint(0, 4, (2, 8, 8))
    one_hot = labels_to_one_hot(target)
    logits = one_hot * 50.0  # near one-hot softmax
    assert soft_dice_loss(logits, one_hot).item() < 1e-3


def test_unet_output_channels_and_size():
    net = UNet(1, 4, base_channels=8, depth=3)
    assert net(torch.rand(1, 1, 64, 64)).shape == (1, 4, 64, 64)


# -- Part 4 VAE / GAN --------------------------------------------------------------------------
def test_vae_forward_and_loss():
    vae = ConvVAE(image_size=64, latent_dim=2, base_channels=8)
    x = torch.rand(3, 1, 64, 64)
    x_hat, mu, log_var = vae(x)
    assert x_hat.shape == x.shape and mu.shape == (3, 2)
    total, rec, kl = vae_loss(x_hat, x, mu, log_var)
    assert torch.isfinite(total) and kl >= 0


def test_gan_shapes():
    g, d = Generator(latent_dim=16, image_size=64, base_channels=8), Discriminator(image_size=64, base_channels=8)
    fake = g(torch.randn(4, 16))
    assert fake.shape == (4, 1, 64, 64) and fake.min() >= -1 and fake.max() <= 1
    assert d(fake).shape == (4,)


def test_checkpoint_roundtrip_mixed_dict(tmp_path):
    """PyTorch >= 2.6 refuses non-tensor objects unless weights_only=False."""
    from common.checkpoint import load_checkpoint, save_checkpoint

    payload = {"state_dict": {"w": torch.tensor([1.0, 2.0])}, "image_size": 128, "tag": "vae"}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(payload, path)
    loaded = load_checkpoint(path)
    assert loaded["image_size"] == 128 and loaded["tag"] == "vae"
    assert torch.equal(loaded["state_dict"]["w"], payload["state_dict"]["w"])


def test_oasis_mask_pairing_same_name_and_seg_prefix(tmp_path):
    from PIL import Image
    from part4_recognition.oasis_data import OASISDataset

    root = tmp_path / "keras_png_slices_data"
    img_dir = root / "keras_png_slices_train"
    seg_dir = root / "keras_png_slices_seg_train"
    img_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)
    arr = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "case_001_slice_0.nii.png")
    Image.fromarray(arr).save(seg_dir / "seg_001_slice_0.nii.png")
    Image.fromarray(arr).save(img_dir / "other_002_slice_1.nii.png")
    Image.fromarray(arr).save(seg_dir / "other_002_slice_1.nii.png")  # identical filename
    ds = OASISDataset(root, "train", image_size=8, with_masks=True)
    assert len(ds) == 2
    img, mask = ds[0]
    assert img.shape == (1, 8, 8) and mask.shape == (8, 8)


def test_missing_mask_never_falls_back_to_sorted_pairing(tmp_path):
    import pytest
    from part4_recognition.oasis_data import OASISDataset

    image = tmp_path / "case_001_slice_1.nii.png"
    seg = tmp_path / "seg"
    seg.mkdir()
    (seg / "seg_999_slice_1.nii.png").touch()
    with pytest.raises(FileNotFoundError):
        OASISDataset._resolve_masks([image], seg)


def test_invalid_mask_levels_rejected():
    import pytest
    with pytest.raises(ValueError, match="Unknown mask"):
        grey_levels_to_labels(np.array([[0, 84, 171, 255]], dtype=np.uint8))


def test_ambiguous_mask_identity_rejected(tmp_path):
    import pytest
    from part4_recognition.oasis_data import OASISDataset

    (tmp_path / "mask_001_slice_1.nii.png").touch()
    (tmp_path / "duplicate_001_slice_1.nii.png").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        OASISDataset._matching_mask(Path("case_001_slice_1.nii.png"), tmp_path)


def test_hard_categorical_predictions():
    from part4_recognition.unet.predict import predict_one_hot

    model = UNet(1, 4, base_channels=4, depth=2).eval()
    output = predict_one_hot(model, torch.rand(2, 1, 32, 32))
    assert output.shape == (2, 4, 32, 32)
    assert torch.equal(output.sum(1), torch.ones(2, 32, 32))
    assert set(output.unique().tolist()) <= {0.0, 1.0}


def test_cifar_training_report_and_eval_guard(tmp_path, monkeypatch):
    """Exercise CLI/timing/checkpoint plumbing with synthetic data, not benchmark evidence."""
    import json
    import pytest
    from part3_cnn.dawnbench import train_cifar10 as trainer

    monkeypatch.setattr(trainer, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--eval_only"])
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2

    def fake_data(*args, **kwargs):
        return {"train_x": torch.rand(4, 3, 40, 40), "train_y": torch.zeros(4, dtype=torch.long),
                "test_x": torch.rand(4, 3, 32, 32), "test_y": torch.zeros(4, dtype=torch.long)}

    monkeypatch.setattr(trainer, "load_cifar10", fake_data)
    monkeypatch.setattr(trainer, "resnet18", lambda **kw: torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 10)))
    monkeypatch.setattr(sys, "argv", ["train", "--epochs", "1", "--batch_size", "2",
                                    "--target_acc", "0", "--tta", "--device", "cpu",
                                    "--output_dir", str(tmp_path)])
    trainer.main()
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["n_train"] == report["n_test"] == 4
    assert report["target_reached"]["epoch"] == 1
    assert report["elapsed_with_evaluation_s"] >= report["train_time_s"] > 0
    assert (tmp_path / "resnet18_cifar10.pt").is_file()


def test_audit_rejects_cross_split_case_leakage(tmp_path):
    import pytest
    from PIL import Image
    from scripts.audit_oasis import audit

    for split in ("train", "validate", "test"):
        images = tmp_path / f"keras_png_slices_{split}"
        masks = tmp_path / f"keras_png_slices_seg_{split}"
        images.mkdir()
        masks.mkdir()
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "case_001_slice_0.nii.png")
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks / "seg_001_slice_0.nii.png")
    with pytest.raises(ValueError, match="leakage"):
        audit(tmp_path)


def test_unet_validation_only_keeps_test_unloaded_and_small_last_batch(tmp_path, monkeypatch):
    import json
    from part4_recognition.unet import train as trainer

    seen_splits = []

    class TinyData(torch.utils.data.Dataset):
        def __init__(self, root, split, *args, **kwargs):
            seen_splits.append(split)
            assert split != "test", "test set must stay untouched while choosing a model"
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return torch.ones(1, 8, 8), torch.full((8, 8), index, dtype=torch.long)
        def label_fractions(self):
            return np.array([1 / 3, 1 / 3, 1 / 3, 0])

    monkeypatch.setattr(trainer, "OASISDataset", TinyData)
    monkeypatch.setattr(trainer, "UNet", lambda *args: torch.nn.Conv2d(1, 4, 1))
    monkeypatch.setattr(trainer, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--epochs", "1", "--batch_size", "2",
                                    "--validation_only", "--device", "cpu", "--output_dir", str(tmp_path)])
    trainer.main()
    assert seen_splits == ["train", "validate"]
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["n_train"] == 3
    assert np.isfinite(report["history"]["train_loss"]).all()
    assert not (tmp_path / "results.json").exists()
    assert (tmp_path / "unet_best.pt").exists()


def test_reference_archive_comparison_detects_changed_bytes(tmp_path):
    import zipfile
    from scripts.verify_oasis_archive import verify

    root = tmp_path / "data"
    root.mkdir()
    (root / "example.png").write_bytes(b"original PNG bytes")
    archive = tmp_path / "reference.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("keras_png_slices_data/example.png", b"original PNG bytes")
    assert verify(root, archive)["matches"]
    (root / "example.png").write_bytes(b"changed PNG bytes")
    assert verify(root, archive)["missing_or_different"] == ["example.png"]
