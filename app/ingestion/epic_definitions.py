"""ePIC DD4hep constants connector: compact/definitions.xml → indexed docs.

The geometry's global constants file holds the detector **system IDs** —
the numeric values that ``edm4hep::ParticleID.getType()`` and the cellID
``system`` bitfield actually return (e.g. BarrelDIRC_ID = 90). Real user
questions ("what is PID type 90?") are unanswerable without this mapping,
which lives nowhere else. One doc carries the full ID table; a second
carries the remaining constants (dimensions, thresholds) in compact form.

Source URL comes from ``EPIC_DEFINITIONS_URL``; unset = source disabled.
"""
from __future__ import annotations

import logging
import re

import httpx

from app.core.config import Settings
from app.ingestion.base import DocumentPayload
from app.ingestion.safe_fetch import safe_get

logger = logging.getLogger(__name__)

_CONSTANT_RE = re.compile(
    r'<constant\s+name="([^"]+)"\s+value="([^"]+)"\s*/>\s*(?:<!--\s*([^>]*?)\s*-->)?'
)

# Colloquial ↔ geometry-name aliases (the collaboration rarely says
# "HcalEndcapN" out loud). EndcapP = positive z = forward = hadron-going;
# EndcapN = negative z = backward = electron-going.
_ALIASES = [
    ("HcalEndcapN", "nHCal, backward HCal (electron-going side)"),
    ("EcalEndcapN", "EEMC, backward ECal (electron-going side)"),
    ("EcalEndcapP", "FEMC, forward ECal (hadron-going side)"),
    ("LFHCAL / HcalEndcapP", "forward HCal (hadron-going side)"),
    ("EcalBarrel", "BIC / BEMC, barrel imaging calorimeter (ScFi + AstroPix)"),
    ("HcalBarrel", "BHCal, barrel HCal"),
    ("BackwardRICH", "pfRICH (proximity-focusing RICH, electron-going side)"),
    ("ForwardRICH", "dRICH (dual-radiator RICH, hadron-going side)"),
    ("BarrelDIRC", "hpDIRC"),
    ("BarrelTOF / EndcapTOF", "AC-LGAD time-of-flight layers"),
]


class EpicDefinitionsConnector:
    source_type = "dd4hep_constants"
    visibility = "public"

    def __init__(self, settings: Settings, *, source_name: str = "epic_definitions") -> None:
        self.settings = settings
        self.source_name = source_name
        self.url = str(getattr(settings, "EPIC_DEFINITIONS_URL", "") or "")
        self.timeout = getattr(settings, "REQUEST_TIMEOUT_SECONDS", 20.0)

    def iter_documents(self, max_items: int | None = None) -> list[DocumentPayload]:
        if not self.url:
            return []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = safe_get(client, self.url)
                response.raise_for_status()
                xml = response.text
        except Exception as exc:
            logger.warning("epic_definitions_fetch_failed", extra={"error": str(exc)[:200]})
            return []
        constants = _CONSTANT_RE.findall(xml)
        if not constants:
            return []
        ids = [(n, v, note) for n, v, note in constants if n.endswith("_ID")]
        others = [(n, v) for n, v, _ in constants if not n.endswith("_ID")]
        html_url = self.url.replace(
            "raw.githubusercontent.com/", "github.com/"
        ).replace("/main/", "/blob/main/", 1)

        preamble = (
            "The ePIC DD4hep geometry assigns every subdetector a numeric "
            "**system ID** (defined in `compact/definitions.xml`). These are "
            "the values returned by `edm4hep::ParticleID.getType()` (the PID "
            "detector-system tag) and stored in the cellID `system` bitfield. "
            "In geometry names, `EndcapP` is the positive-z / forward / "
            "hadron-going side and `EndcapN` the negative-z / backward / "
            "electron-going side."
        )
        id_lines = [
            preamble
            + " For example, PID type 90 means BarrelDIRC, 92 means "
            "BarrelTOF, 131 means BackwardRICH (pfRICH).",
            "",
            "## Naming convention and detector aliases",
            "",
            "The same detector often has a geometry name and a colloquial "
            "name:",
            "",
            "| geometry name | also known as |",
            "|---|---|",
        ]
        id_lines += [f"| `{name}` | {alias} |" for name, alias in _ALIASES]
        id_lines += [
            "",
            "## System ID table",
            "",
            "| constant | system ID | note |",
            "|---|---|---|",
        ]
        row_aliases = {
            "HcalEndcapN": "nHCal / backward HCal",
            "EcalEndcapN": "EEMC / backward ECal",
            "EcalEndcapP": "FEMC / forward ECal",
            "HcalEndcapPInsert": "forward HCal insert",
            "LFHCAL": "forward HCal",
            "HcalEndcapP": "LFHCAL / forward HCal",
            "EcalBarrel": "BIC / BEMC barrel imaging calorimeter",
            "HcalBarrel": "BHCal / barrel HCal",
            "BackwardRICH": "pfRICH",
            "ForwardRICH": "dRICH",
            "BarrelDIRC": "hpDIRC",
            "BarrelTOF": "AC-LGAD barrel TOF",
            "ForwardTOF": "AC-LGAD forward TOF",
            "VertexBarrel": "SVT vertex layers",
            "LumiSpecCAL": "luminosity spectrometer ECal",
            "ZDC": "zero-degree calorimeter",
        }

        def _row_note(name: str, note: str) -> str:
            base = name.removesuffix("_ID")
            base = re.sub(r"_\d+$", "", base)
            alias = row_aliases.get(base, "")
            return " — ".join(p for p in (alias, note) if p)

        id_lines += [
            f"| `{name}` | {value} | {_row_note(name, note or '')} |"
            for name, value, note in ids
        ]
        id_lines += ["", f"Source: {html_url}"]

        docs = [
            DocumentPayload(
                source_name=self.source_name,
                source_type=self.source_type,
                external_id="epic-definitions:system-ids",
                title="ePIC detector system IDs (DD4hep constants, PID type values)",
                url=html_url,
                content="\n".join(id_lines),
                visibility=self.visibility,
                section_path=["ePIC geometry", "constants"],
                filetype="xml",
                metadata={"constants": len(ids)},
            )
        ]

        # One compact doc per detector family so the family's full ID list
        # plus its aliases land inside a single chunk — the 132-row master
        # table gets split by the chunker and retrieval then sees the wrong
        # half (real failure: nHCal question surfaced the pfRICH row).
        families = [
            ("calorimeters", "ePIC calorimeter system IDs (BIC/BEMC barrel, EEMC backward ECal, FEMC forward ECal, BHCal, nHCal/HcalEndcapN backward HCal, LFHCAL forward HCal, ZDC, B0ECal, luminosity)", re.compile(r"[EH]cal|LFHCAL|ZDC|Lumi", re.I)),
            ("pid", "ePIC PID system IDs (dRICH/ForwardRICH, pfRICH/BackwardRICH, hpDIRC/BarrelDIRC, AC-LGAD TOF barrel and forward)", re.compile(r"RICH|DIRC|TOF|PID|PFRICH", re.I)),
            ("tracking", "ePIC tracker system IDs (SVT vertex barrel/endcaps, sagitta tracker barrel, endcap trackers, MPGD)", re.compile(r"Vertex|Tracker|MPGD|Sagitta|SVT", re.I)),
            ("far-detectors", "ePIC far-forward and far-backward system IDs (B0, Roman pots, off-momentum, ZDC, low-Q2 taggers)", re.compile(r"B0|RomanPot|OffM|Tagger|LowQ2|Forward|Backward", re.I)),
        ]
        assigned: set[str] = set()
        for key, title, pattern in families:
            rows = [
                (n, v, note) for n, v, note in ids
                if n not in assigned and pattern.search(n)
            ]
            if not rows:
                continue
            assigned.update(n for n, _, _ in rows)
            lines = [
                preamble,
                "",
                "| constant | system ID | note |",
                "|---|---|---|",
            ]
            lines += [
                f"| `{n}` | {v} | {_row_note(n, note or '')} |" for n, v, note in rows
            ]
            lines += ["", f"Source: {html_url}"]
            docs.append(
                DocumentPayload(
                    source_name=self.source_name,
                    source_type=self.source_type,
                    external_id=f"epic-definitions:ids-{key}",
                    title=title,
                    url=html_url,
                    content="\n".join(lines),
                    visibility=self.visibility,
                    section_path=["ePIC geometry", "constants", key],
                    filetype="xml",
                    metadata={"constants": len(rows)},
                )
            )
        if others:
            other_lines = [
                "Global ePIC DD4hep constants (dimensions, offsets, angles) "
                "from `compact/definitions.xml`. Values may be expressions "
                "over other constants.",
                "",
            ]
            other_lines += [f"- `{name}` = `{value}`" for name, value in others]
            other_lines += ["", f"Source: {html_url}"]
            docs.append(
                DocumentPayload(
                    source_name=self.source_name,
                    source_type=self.source_type,
                    external_id="epic-definitions:constants",
                    title="ePIC global geometry constants (DD4hep definitions.xml)",
                    url=html_url,
                    content="\n".join(other_lines),
                    visibility=self.visibility,
                    section_path=["ePIC geometry", "constants"],
                    filetype="xml",
                    metadata={"constants": len(others)},
                )
            )
        return docs[: max_items or len(docs)]
