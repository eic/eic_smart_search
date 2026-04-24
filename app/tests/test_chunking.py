from app.ingestion.base import DocumentPayload
from app.ingestion.chunking import HeadingAwareChunker


def test_heading_aware_chunker_preserves_section_path() -> None:
    document = DocumentPayload(
        source_name="test",
        source_type="website",
        external_id="https://example.test/doc",
        title="Doc",
        url="https://example.test/doc",
        section_path=["Documentation"],
        content="# Simulation Tutorials\n\nUse the tutorial runner.\n\n## Setup\n\nInstall dependencies first.",
    )

    chunks = HeadingAwareChunker(target_tokens=12, overlap_tokens=0).chunk(document)

    assert chunks
    assert chunks[0].heading_path == ["Documentation", "Simulation Tutorials"]
    assert any("Setup" in " / ".join(chunk.heading_path) for chunk in chunks)

