import torch
from torch import nn

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_vae import WanVideoVAE


class _RecordingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def encode(self, video: torch.Tensor, scale: object) -> torch.Tensor:
        assert scale == "test-scale"
        self.inputs.append(video.clone())
        return video.mean(dim=1, keepdim=True)


def _make_actor() -> tuple[FastWAM, _RecordingEncoder]:
    encoder = _RecordingEncoder()
    vae = WanVideoVAE.__new__(WanVideoVAE)
    nn.Module.__init__(vae)
    vae.model = encoder
    vae.scale = "test-scale"

    actor = FastWAM.__new__(FastWAM)
    nn.Module.__init__(actor)
    actor.vae = vae
    actor.device = torch.device("cpu")
    return actor, encoder


def test_input_image_latent_batch_matches_serial_vae_encoding() -> None:
    actor, encoder = _make_actor()
    images = torch.arange(4 * 3 * 224 * 448, dtype=torch.float32).reshape(
        4, 3, 224, 448
    )

    batched = actor._encode_input_image_latents_tensor(images, tiled=False)

    assert batched.shape == (4, 1, 1, 224, 448)
    assert [tuple(video.shape) for video in encoder.inputs] == [
        (1, 3, 1, 224, 448),
    ] * 4
    assert all(
        torch.equal(encoded, expected.unsqueeze(0).unsqueeze(2))
        for encoded, expected in zip(encoder.inputs, images, strict=True)
    )

    encoder.inputs.clear()
    serial = torch.cat(
        [
            actor._encode_input_image_latents_tensor(image, tiled=False)
            for image in images
        ]
    )

    assert torch.equal(batched, serial)
