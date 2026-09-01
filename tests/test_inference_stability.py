from care_ego import inference


def test_invalid_cuda_graph_batch_size_disables_graph(monkeypatch) -> None:
    monkeypatch.setenv("WAKE_CUDA_GRAPH_BATCH_SIZE", "not-an-integer")

    assert inference._cuda_graph_batch_size() == 0


def test_negative_cuda_graph_batch_size_disables_graph(monkeypatch) -> None:
    monkeypatch.setenv("WAKE_CUDA_GRAPH_BATCH_SIZE", "-2")

    assert inference._cuda_graph_batch_size() == 0
