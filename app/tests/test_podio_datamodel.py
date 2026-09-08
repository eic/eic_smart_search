from __future__ import annotations

from app.ingestion.podio_datamodel import (
    PodioDatamodelConnector,
    _doxygen_class_url,
    _parse_member,
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
