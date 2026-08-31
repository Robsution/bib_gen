import torch

from src.models import BIBGenerator, CategoricalMultiplicity, DeepSetsDiscriminator


def test_baseline_model_shapes():
    batch_size = 4
    max_daughters = 5
    feature_dim = 11
    parent = torch.randn(batch_size, 3)
    normalized_n = torch.full((batch_size, 1), 0.4)
    noise = torch.randn(batch_size, 7)

    generator = BIBGenerator(
        noise_dim=7,
        parent_dim=3,
        max_daughters=max_daughters,
        kinematic_dim=8,
        species_dim=3,
        hidden_dim=16,
    )
    discriminator = DeepSetsDiscriminator(feature_dim=feature_dim, parent_dim=3, hidden_dim=16)

    features, mask = generator(noise, parent, normalized_n)
    score = discriminator(features, parent, mask, normalized_n)

    assert features.shape == (batch_size, max_daughters, feature_dim)
    assert mask.shape == (batch_size, max_daughters, 1)
    assert score.shape == (batch_size, 1)


def test_categorical_multiplicity_shape():
    model = CategoricalMultiplicity(parent_dim=3, max_n=5, hidden_dim=16)
    assert model(torch.randn(4, 3)).shape == (4, 6)
