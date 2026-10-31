"""Neural DSP preset (``.xml``) reader. SKELETON: reads the container and the parameter table; mapping to the training schema is not done.

What a real file is (checked on a preset from the Neural DSP Unity forum): despite the extension, Neural DSP plugins save a **JUCE binary
ValueTree**, not text XML. Layout::

    <plugin id>\\0  \\x01 ... editorSize / routing / tempo / presetNameProp / presetPathProp / presetUIDProp / presetVersionProp ...
    PARAM\\0 \\x01\\x02 id\\0 \\x01<len>\\x05 <parameter name>\\0 value\\0 \\x01\\t\\x04 <8 bytes: little-endian IEEE double>  (one block per knob)
    ... cabsim / filePath1 / filePath2 ...

``plugin`` is the leading id (``neural_dsp_gojira``), ``name`` is ``presetNameProp``, and ``params`` maps each parameter name
(``ampType``, ``cab1Level``, ``cleanAmpBass`` ...) to its stored double. Some exports are real XML (``<PARAM id="gain" value="0.5"/>``); that form
is read too.

Two binary variants exist in the forum uploads (both handled):

* ``juce_binary``    older Archetype plugins: ``PARAM``/``id``/``value`` blocks holding little-endian doubles (the layout above).
* ``juce_binary_v2`` newer "-X" plugins (id like ``gojira-X``): ``name\0 \x01<len> \x05 <text value>\0`` string properties; numbers and
  ``true``/``false`` are written as text (``outputGain 5.0625``, ``gateThreshold -77.2``, ``doublerActive false``), nested under ``parameters`` /
  ``subModels``. Property names repeat in different sub-models (``sectionActive``, ``selectedAmp``), so ``params`` is keyed by the model path
  (``tree_key``): ``amp.sectionActive``, ``eq.sectionActive``, ``amp.leadAmp.leadMaster``; plugin-wide knobs (``gateActive``) stay bare.

VERIFIED on a real preset: stored values are RAW display values, not normalised 0-1 (``cab1Level = -12.0`` dB, ``ampType = 2.0``,
``gateThreshold = -77.2``), so they must be rescaled with the plugin's parameter ranges (``common.schema.load_schema``) before use as 0-1 labels.
Enumerated choices are stored as indices. NOT YET DONE: that rescaling and the per-plugin name mapping to the training schema.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

PARAM_BINARY = re.compile(rb"PARAM\x00\x01\x02id\x00\x01.\x05([^\x00]+)\x00value\x00\x01\t\x04(.{8})", re.S)
NAME_PROP = re.compile(rb"presetNameProp\x00\x01.\x05([^\x00]*)\x00", re.S)
VERSION_PROP = re.compile(rb"presetVersionProp\x00\x01.\x05([^\x00]*)\x00", re.S)
PLUGIN_ID = re.compile(rb"^([A-Za-z0-9_\-. ]{3,64})\x00")
UNVERIFIED = "values are raw display values (e.g. dB), not 0-1: rescale with the plugin schema before use"
STRING_PROP = re.compile(rb"([A-Za-z][A-Za-z0-9_]*)\x00\x01.\x05([^\x00]*)\x00")


@dataclass
class NeuralXmlPreset:
    plugin: str
    name: str
    params: dict[str, float]
    version: str = ""
    container: str = "juce_binary"                        # or "xml"
    warnings: list[str] = field(default_factory=list)

    def param(self, name: str) -> float | None:
        return self.params.get(name)

    def active(self, prefix: str) -> bool | None:
        """Module on/off switch, e.g. ``active('cab')`` reads ``cabActive`` (None when the preset has no such switch)."""
        v = self.params.get(f"{prefix}Active")
        return None if v is None else v >= 0.5


def _parse_binary(data: bytes) -> NeuralXmlPreset:
    m = PLUGIN_ID.match(data)
    params: dict[str, float] = {}
    for raw_name, raw_value in PARAM_BINARY.findall(data):
        params[raw_name.decode("utf-8", "replace")] = struct.unpack("<d", raw_value)[0]
    name, version = NAME_PROP.search(data), VERSION_PROP.search(data)
    if not params:
        return _parse_binary_v2(data, m)
    preset = NeuralXmlPreset(m.group(1).decode("ascii") if m else "", name.group(1).decode("utf-8", "replace") if name else "", params,
                             version.group(1).decode("utf-8", "replace") if version else "")
    if not m:
        preset.warnings.append("plugin id not found at the start of the file")
    return preset


def _length(data: bytes, i: int) -> tuple[int, int]:
    """A length / count: one byte (a file path of 174 bytes is stored as ``0xAE``, so this is not a varint). ``0xFF`` is not seen in the scraped files and is refused."""
    if data[i] == 0xFF:
        raise ValueError(f"unsupported long length at {i}")
    return data[i], i + 1


def _cstr(data: bytes, i: int) -> tuple[str, int]:
    j = data.index(b"\x00", i)
    return data[i:j].decode("utf-8", "replace"), j + 1


def _count(data: bytes, i: int) -> tuple[int, int]:
    """A props / children block: a single ``\0`` (none) or ``\x01 <n>``."""
    if data[i] == 0:
        return 0, i + 1
    if data[i] != 1:
        raise ValueError(f"bad block marker {data[i]:#x} at {i}")
    return _length(data, i + 1)


def _walk(data: bytes, i: int, path: tuple[str, ...], out: list[tuple[tuple[str, ...], str, str]]) -> int:
    """One node: ``name\0 <props> <children>``; a property is ``name\0 \x01 <len> <body>`` and a text body is ``\x05 <text> \0``.
    Appends ``(node path, property name, text value)`` for every text property."""
    name, i = _cstr(data, i)
    here = path + (name,)
    n, i = _count(data, i)
    for _ in range(n):
        key, i = _cstr(data, i)
        if data[i] != 1:
            raise ValueError(f"bad property marker at {i}")
        size, i = _length(data, i + 1)
        body = data[i:i + size]
        i += size
        if body[:1] == b"\x05":
            out.append((here, key, body[1:].rstrip(b"\x00").decode("utf-8", "replace")))
    n, i = _count(data, i)
    for _ in range(n):
        i = _walk(data, i, here, out)
    return i


STRUCTURAL = {"subModels", "appModel", "parameters", "fixedParameters", "meters", "preset", "midiMap", "listElements"}


def tree_key(node_path: tuple[str, ...], prop: str) -> str:
    """Hierarchical key for a property: its section / model chain, then the property name, e.g. ``amp.sectionActive``, ``eq.sectionActive``,
    ``amp.leadAmp.leadMaster``, ``postFX.postDelay.postDelayTime``. A ``...Parameters`` node is named by its stem (``ampParameters`` -> ``amp``);
    structural containers (``subModels``, ``appModel``, ``parameters`` ...) are dropped, so the plugin-wide knobs (``gateActive``) stay bare.
    Selector properties on the app model are filed under what they select (``selectedAmp`` -> ``amp.selectedAmp``)."""
    chain = [re.sub(r"Parameters$", "", n) for n in node_path[1:] if n not in STRUCTURAL]
    if node_path[-1] == "appModel":
        m = re.match(r"selected([A-Z][A-Za-z]*)$", prop)
        chain = [m.group(1).lower()] if m else ["app"]
    return ".".join(chain + [prop])


def _parse_binary_v2(data: bytes, m) -> NeuralXmlPreset:
    """Tree-aware reader for the X-series container. Sub-models repeat property names (``sectionActive`` in every section, ``selectedAmp`` ...),
    so each value is keyed by the path of the model it sits in (:func:`tree_key`) instead of its bare name."""
    props: list[tuple[tuple[str, ...], str, str]] = []
    warnings: list[str] = []
    try:
        end = _walk(data, 0, (), props)
        if end != len(data):
            warnings.append(f"tree reader stopped at byte {end} of {len(data)}")
    except (ValueError, IndexError) as e:
        warnings.append(f"tree reader failed ({e}); using the flat reader, repeated names overwrite each other")
        props = [((), k.decode("utf-8", "replace"), v.decode("utf-8", "replace")) for k, v in STRING_PROP.findall(data)]
    params: dict[str, float] = {}
    meta: dict[str, str] = {}
    for node_path, k, v in props:
        key = tree_key(node_path, k) if node_path else k
        if not node_path or len(node_path) == 1:
            meta.setdefault(k, v)                                 # root properties: version, name, isFavorite
        if v in ("true", "false"):
            params[key] = 1.0 if v == "true" else 0.0
        else:
            try:
                params[key] = float(v)
            except ValueError:
                pass
    pre = NeuralXmlPreset(m.group(1).decode("ascii") if m else "", meta.get("name", ""), params, meta.get("version", ""), container="juce_binary_v2")
    pre.warnings += warnings
    return pre


def _parse_text(data: bytes) -> NeuralXmlPreset:
    root = ElementTree.fromstring(data.decode("utf-8-sig"))
    params = {}
    for el in root.iter():
        if el.tag.upper() == "PARAM" and el.get("id") is not None:
            try:
                params[el.get("id")] = float(el.get("value"))
            except (TypeError, ValueError):
                continue
    return NeuralXmlPreset(root.tag, root.get("presetName") or root.get("name") or "", params, root.get("version") or "", container="xml")


def parse_neural_xml(path: str | Path) -> NeuralXmlPreset:
    data = Path(path).read_bytes()
    stripped = data.lstrip(b"\xef\xbb\xbf \r\n\t")
    preset = _parse_text(data) if stripped.startswith(b"<") else _parse_binary(data)
    if not preset.params:
        raise ValueError(f"{path}: no PARAM entries found (not a Neural DSP preset, or an unknown container)")
    preset.warnings.append(UNVERIFIED)
    return preset
