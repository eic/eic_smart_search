from app.llm.embeddings import TeiEmbeddingProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


def test_tei_embedding_provider_uses_tei_payload_and_query_instruction(monkeypatch) -> None:
    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse([[0.1, 0.2]])

    monkeypatch.setattr("app.llm.embeddings.httpx.post", fake_post)
    provider = TeiEmbeddingProvider(
        url="http://embedding:80",
        dimension=2,
        query_instruction="Given a web search query, retrieve relevant passages that answer the query",
    )

    assert provider.embed_texts(["plain document"]) == [[0.1, 0.2]]
    assert provider.embed_query("How do I run tutorials?") == [0.1, 0.2]

    assert calls[0]["url"] == "http://embedding:80/embed"
    assert calls[0]["json"] == {"inputs": ["plain document"]}
    assert calls[1]["json"] == {
        "inputs": [
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query: How do I run tutorials?"
        ]
    }
