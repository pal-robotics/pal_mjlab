"""SuperMimic encoder-decoder actor as an rsl_rl model."""

import torch
from pal_mjlab.tasks.supermimic.rl.models import EncoderDecoderModel
from tensordict import TensorDict


def _obs(batch: int = 6) -> TensorDict:
  g = torch.Generator().manual_seed(0)
  return TensorDict(
    {
      "proprio": torch.randn(batch, 20, generator=g),
      "goal": torch.randn(batch, 12, generator=g),
      "critic": torch.randn(batch, 9, generator=g),
    },
    batch_size=[batch],
  )


def _model(**kw) -> EncoderDecoderModel:
  torch.manual_seed(0)
  return EncoderDecoderModel(
    _obs(),
    {"actor": ["proprio", "goal"], "critic": ["critic"]},
    "actor",
    output_dim=4,
    hidden_dims=(16, 8),
    encoder_dims=(16,),
    latent_dim=3,
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    **kw,
  )


def test_goal_is_encoded_and_proprio_bypasses_the_encoder():
  m = _model()
  assert m.obs_groups == ["goal", "proprio"] and m.goal_dim == 12
  assert m.encoder[0].in_features == 12
  assert m.mlp[0].in_features == 3 + 20
  obs = _obs()
  out = m(obs)
  assert out.shape == (6, 4)
  # Changing only the proprio part changes the action through the decoder alone.
  obs2 = obs.clone()
  obs2["proprio"] += 1.0
  assert not torch.allclose(m(obs2), out)


def test_gradients_reach_the_encoder():
  m = _model()
  m(_obs(), stochastic_output=True)
  m.get_output_log_prob(torch.zeros(6, 4)).sum().backward()
  assert m.encoder[0].weight.grad.abs().sum() > 0


def test_export_matches_the_model():
  m = _model()
  obs = _obs()
  m.update_normalization(obs)
  flat = torch.cat([obs["goal"], obs["proprio"]], dim=-1)
  exported = m.as_jit()
  assert torch.allclose(exported(flat), m(obs), atol=1e-6)
