"""Generate new samples and latent interpolations from a saved GAN checkpoint."""
from pathlib import Path
import argparse
import sys

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from part4_recognition.gan.model import Generator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["args"]
    model = Generator(config["latent_dim"], config["image_size"], config["base_channels"])
    model.load_state_dict(state["generator"])
    model.to(args.device).eval()
    rng = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(64, config["latent_dim"], generator=rng)
    pairs = torch.randn(8, 2, config["latent_dim"], generator=rng)
    t = torch.linspace(0, 1, 9).view(1, 9, 1)
    interpolation = ((1 - t) * pairs[:, :1] + t * pairs[:, 1:]).flatten(0, 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, inputs, columns in [("new_samples", noise, 8), ("interpolations", interpolation, 9)]:
            images = torch.cat([model(batch.to(args.device)).cpu() for batch in inputs.split(32)])
            assert torch.isfinite(images).all(), "Non-finite generated images"
            save_image((images.clamp(-1, 1) + 1) / 2, args.output_dir / f"{name}.png", nrow=columns)
    print(f"Loaded epoch {state['epoch']}; images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
