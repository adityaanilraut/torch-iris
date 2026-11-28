import pytest
import torch
import torch_iris
from torch.utils.data import DataLoader, TensorDataset

from examples.mnist_iris import LAST_CHECKPOINT_PATH, load_training_checkpoint, train_epochs, training_state


@pytest.mark.skipif(not torch_iris.is_available(), reason="requires Intel Iris Xe")
def test_mnist_checkpoint_resume_with_dropout(tmp_path):
    device = torch.device("iris:0")
    data = TensorDataset(torch.randn(8, 1, 28, 28, generator=torch.Generator().manual_seed(99)), torch.arange(8))
    val_loader = DataLoader(data, batch_size=4)

    class RecordingLoader(DataLoader):
        def __iter__(self):
            for images, labels in super().__iter__():
                order.append(labels.tolist())
                yield images, labels

    def run(directory, epochs, resume=None, seed=17):
        order.clear()
        torch.manual_seed(seed)
        torch_iris.manual_seed(seed)
        shuffle = torch.Generator().manual_seed(seed)
        loader = RecordingLoader(data, batch_size=4, shuffle=True, generator=shuffle)
        state = training_state(device)
        directory.mkdir(exist_ok=True)
        start, accuracy = 1, -1.0
        if resume is not None:
            checkpoint = load_training_checkpoint(resume, *state, shuffle)
            start, accuracy = checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
        history = train_epochs(*state, loader, val_loader, device, start, epochs, accuracy, directory)
        saved = torch.load(directory / LAST_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        return history, list(order), saved

    enabled = torch_iris.is_cpu_fallback_enabled()
    torch_iris.set_cpu_fallback_enabled(False)
    order = []
    try:
        full, full_order, expected = run(tmp_path / "full", 3)
        first, first_order, first_state = run(tmp_path / "split", 1)
        checkpoint = tmp_path / "split" / LAST_CHECKPOINT_PATH
        second, second_order, actual = run(tmp_path / "split", 3, resume=checkpoint, seed=123)
    finally:
        torch_iris.set_cpu_fallback_enabled(enabled)
    assert full == first + second
    assert full_order == first_order + second_order
    assert second[0]["next_learning_rate"] == 0.0005
    assert not torch.equal(first_state["iris_rng_state"], actual["iris_rng_state"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
