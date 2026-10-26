from __future__ import annotations

import httpx

from app.ingestion.podio_datamodel import (
    PodioDatamodelConnector,
    _doxygen_class_url,
    _parse_member,
    _yaml_blob_url,
)


class _Settings:
    DATAMODEL_YAML_URLS = ""
    REQUEST_TIMEOUT_SECONDS = 5.0


_MODEL = {
    "datatypes": {
        "edm4eic::Track": {
            "Description": "Track information",
            "Members": [
                "float chi2 // Total chi2",
                "std::int32_t ndf // Number of degrees of freedom",
            ],
            "OneToOneRelations": ["edm4eic::Trajectory trajectory // Trajectory of this track"],
            "OneToManyRelations": ["edm4eic::Measurement2D measurements // Measurements"],
        },
        "edm4eic::Trajectory": {"Description": "Raw trajectory from tracking"},
    },
    "components": {
        "edm4eic::Cov2f": {"Members": ["float xx", "float yy"]},
    },
    "links": {
        "edm4eic::TrackClusterLink": {"From": "edm4eic::Track", "To": "edm4eic::Cluster"},
    },
}


def _docs():
    connector = PodioDatamodelConnector(_Settings())
    return connector._model_documents(_MODEL, source_url="https://example.com/edm4eic.yaml")


def test_one_document_per_type() -> None:
    docs = _docs()
    assert len(docs) == 4
    ids = {d.external_id for d in docs}
    assert "datamodel:edm4eic::Track" in ids
    assert "datamodel:edm4eic::Cov2f" in ids
    assert "datamodel:edm4eic::TrackClusterLink" in ids


def test_track_document_content() -> None:
    track = next(d for d in _docs() if d.external_id == "datamodel:edm4eic::Track")
    assert "Track information" in track.content
    assert "`chi2` (`float`) — Total chi2" in track.content
    assert "One-to-one relations" in track.content
    assert "`trajectory` (`edm4eic::Trajectory`)" in track.content
    assert track.url == "https://eic.github.io/EDM4eic/classedm4eic_1_1_track.html"
    assert track.section_path == ["EDM4EIC", "datatypes"]


def test_referenced_by_reverse_index() -> None:
    trajectory = next(d for d in _docs() if d.external_id == "datamodel:edm4eic::Trajectory")
    assert "Referenced by" in trajectory.content
    assert "`edm4eic::Track`" in trajectory.content


def test_link_document_names_endpoints() -> None:
    link = next(d for d in _docs() if d.external_id == "datamodel:edm4eic::TrackClusterLink")
    assert "Links `edm4eic::Track` ↔ `edm4eic::Cluster`" in link.content


def test_doxygen_url_casing() -> None:
    assert (
        _doxygen_class_url("edm4eic::TrackClusterMatch")
        == "https://eic.github.io/EDM4eic/classedm4eic_1_1_track_cluster_match.html"
    )
    assert (
        _doxygen_class_url("edm4hep::MCParticle")
        == "https://edm4hep.web.cern.ch/classedm4hep_1_1_m_c_particle.html"
    )
    assert _doxygen_class_url("unknown::Thing") == ""


def test_parse_member_variants() -> None:
    assert _parse_member("float chi2 // Total chi2") == ("float", "chi2", "Total chi2")
    assert _parse_member("edm4hep::Vector3f position") == ("edm4hep::Vector3f", "position", "")


# ---- Doxygen publication check -------------------------------------------

_RAW_URL = "https://raw.githubusercontent.com/eic/EDM4eic/main/edm4eic.yaml"
_YAML_TEXT = """---
schema_version: 2
datatypes:
  edm4eic::Track:
    Description: "Track information"
  edm4eic::Trajectory:
    Description: "Raw trajectory from tracking"
components:
  edm4eic::Cov2f:
    Members:
      - float xx
links:
  edm4eic::TrackClusterLink:
    From: edm4eic::Track
    To: edm4eic::Cluster
"""
_ANNOTATED = """<html><body><table>
<tr><td><a class="el" href="classedm4eic_1_1_track.html" target="_self">Track</a></td></tr>
<tr><td><a class="el" href="classedm4eic_1_1_trajectory.html">Trajectory</a></td></tr>
<tr><td><a class="el" href="classedm4eic_1_1_cov2f.html">Cov2f</a></td></tr>
</table></body></html>"""


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _docs_with(handler) -> tuple[dict, list[str]]:
    calls: list[str] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return handler(request)

    connector = PodioDatamodelConnector(_Settings())
    with _client(wrapped) as client:
        docs = connector._model_documents(
            _MODEL, source_url=_RAW_URL, yaml_text=_YAML_TEXT, client=client
        )
    return {d.external_id: d for d in docs}, calls


def test_published_class_keeps_doxygen_url() -> None:
    docs, calls = _docs_with(lambda request: httpx.Response(200, text=_ANNOTATED))
    track = docs["datamodel:edm4eic::Track"]
    assert track.url == "https://eic.github.io/EDM4eic/classedm4eic_1_1_track.html"
    assert track.metadata["doc_page"] == "doxygen"
    # annotated.html fetched once per Doxygen base, not per type.
    assert calls == ["https://eic.github.io/EDM4eic/annotated.html"]


def test_unpublished_link_falls_back_to_yaml_line() -> None:
    docs, _ = _docs_with(lambda request: httpx.Response(200, text=_ANNOTATED))
    link = docs["datamodel:edm4eic::TrackClusterLink"]
    assert link.url == "https://github.com/eic/EDM4eic/blob/main/edm4eic.yaml#L13"
    assert link.metadata["doc_page"] == "yaml"
    assert _YAML_TEXT.splitlines()[12].strip() == "edm4eic::TrackClusterLink:"


def test_annotated_fetch_failure_keeps_doxygen_urls() -> None:
    docs, _ = _docs_with(lambda request: httpx.Response(503))
    link = docs["datamodel:edm4eic::TrackClusterLink"]
    assert link.url == (
        "https://eic.github.io/EDM4eic/classedm4eic_1_1_track_cluster_link.html"
    )
    assert link.metadata["doc_page"] == "doxygen"

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    docs, _ = _docs_with(boom)
    assert all(d.metadata["doc_page"] == "doxygen" for d in docs.values())


def test_yaml_blob_url_shapes() -> None:
    assert _yaml_blob_url(_RAW_URL) == "https://github.com/eic/EDM4eic/blob/main/edm4eic.yaml"
    assert (
        _yaml_blob_url("https://raw.githubusercontent.com/key4hep/EDM4hep/refs/heads/main/edm4hep.yaml")
        == "https://github.com/key4hep/EDM4hep/blob/main/edm4hep.yaml"
    )
    assert (
        _yaml_blob_url("https://github.com/eic/EDM4eic/raw/v8.10.0/edm4eic.yaml")
        == "https://github.com/eic/EDM4eic/blob/v8.10.0/edm4eic.yaml"
    )
    assert _yaml_blob_url("https://example.com/edm4eic.yaml") == "https://example.com/edm4eic.yaml"
