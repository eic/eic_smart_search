

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload



class _FlakyCudaST:
    """Fake SentenceTransformer: cuda load raises the WSL2 meta-tensor error."""

    def __init__(self, model_name, device=None, **kwargs):
        if device == "cuda":
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self):
        return 384


def test_sentence_transformers_provider_falls_back_to_cpu(monkeypatch) -> None:
    import sentence_transformers

    from app.llm.embeddings import SentenceTransformersEmbeddingProvider

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FlakyCudaST)
    provider = SentenceTransformersEmbeddingProvider(
        model_name="fake-model", dimension=384, device="cuda"
    )
    assert provider.model.device == "cpu"


def test_sentence_transformers_provider_cpu_failure_raises(monkeypatch) -> None:
    import pytest
    import sentence_transformers

    from app.llm.embeddings import SentenceTransformersEmbeddingProvider

    class _AlwaysBroken:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("meta tensor")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _AlwaysBroken)
    with pytest.raises(NotImplementedError):
        SentenceTransformersEmbeddingProvider(
            model_name="fake-model", dimension=384, device="cpu"
        )
