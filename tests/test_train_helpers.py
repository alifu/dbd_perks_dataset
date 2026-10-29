import torch

import train_perks_v2 as train


def test_classifier_outputs_one_logit_per_class_for_128px_input():
    model = train.PerkClassifier(5, pretrained=False).eval()
    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 5)


def test_normalization_lives_inside_the_model():
    model = train.PerkClassifier(3, pretrained=False)
    assert torch.allclose(model.mean.flatten(), torch.tensor(train.IMAGENET_MEAN))
    assert torch.allclose(model.std.flatten(), torch.tensor(train.IMAGENET_STD))


def test_freezing_backbone_leaves_only_the_head_trainable():
    model = train.PerkClassifier(3, pretrained=False)
    train.set_backbone_trainable(model, False)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in model.head_parameters()}
    train.set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_load_labels_are_sorted_perk_ids():
    labels = train.load_labels()
    assert labels == sorted(labels) and len(labels) == len(set(labels)) > 250


def test_evaluate_counts_top1_top3_and_confusions():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            # always predicts class 1, then 0, then 2
            return torch.tensor([[0.5, 0.9, 0.1]]).repeat(x.shape[0], 1)

    loader = [(torch.zeros(3, 3, 4, 4), torch.tensor([1, 0, 2]))]
    metrics = train.evaluate(Fixed(), loader, torch.device("cpu"), 3)

    assert metrics["top1"] == 1 / 3
    assert metrics["top3"] == 1.0
    assert metrics["confusions"] == {(0, 1): 1, (2, 1): 1}
    summary = train.summarize(metrics, ["a", "b", "c"])
    assert summary["worstClasses"][0]["accuracy"] == 0.0
    assert {item["expected"] for item in summary["topConfusions"]} == {"a", "c"}


def test_perks_sharing_an_icon_are_not_trainable():
    perks = [
        {"id": "a", "iconFile": "a.png"},
        {"id": "b", "iconFile": "Unknown_QuestionMark.png"},
        {"id": "c", "iconFile": "Unknown_QuestionMark.png"},
        {"id": "d", "iconFile": "d.png"},
    ]
    assert train.recognizable_perk_ids(perks) == ["a", "d"]
