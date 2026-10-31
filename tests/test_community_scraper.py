"""Offline tests for the community spiders (reddit / neural_forum / tonehunt helpers), download.py, archive.py and the two parsers.
Binary fixtures reproduce the layout of a real Neural DSP preset downloaded from the Unity forum (JUCE value tree under a .xml name)."""

from __future__ import annotations

import io
import json
import struct
import urllib.error
import zipfile

import pytest

from patch_corpus.parsers.nam_model import parse_ir, parse_nam
from patch_corpus.parsers.neural_xml import parse_neural_xml
from patch_scraper import neural_forum, reddit, tonehunt
from patch_scraper.archive import Sink, sniff_ext
from patch_scraper.download import Fetcher, Ledger, classify, extract_urls, harvest
from patch_scraper.manifest import read_rows


def param(name: str, value: float) -> bytes:
    return b"PARAM\x00\x01\x02id\x00\x01\x0c\x05" + name.encode() + b"\x00value\x00\x01\t\x04" + struct.pack("<d", value) + b"\x00"


def ndsp_xml(preset="Take a Bite", **params) -> bytes:
    body = b"".join(param(k, v) for k, v in (params or {"gain": 0.5, "ampsActive": 1.0}).items())
    return (b"neural_dsp_gojira\x00\x01\x07editorSize\x00\x01\x05\x01\x02\x00\x00\x00presetNameProp\x00\x01\r\x05" + preset.encode()
            + b"\x00presetVersionProp\x00\x01\x07\x051.0.0\x00\x01w" + body + b"cabsim\x00\x01\x03index\x00")


def nam(name="Metallica Rhythm", gear="Mesa") -> bytes:
    return json.dumps({"version": "0.5.4", "architecture": "WaveNet", "config": {}, "weights": [0.1, 0.2],
                       "sample_rate": 48000, "metadata": {"name": name, "gear_make": gear, "gear_model": "Rectifier"}}).encode()


def wav(seconds=0.1, rate=48000) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    import wave
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"\x00\x01" * n)
    return buf.getvalue()


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, v in files.items():
            zf.writestr(k, v)
    return buf.getvalue()


@pytest.fixture
def sink(tmp_path):
    return Sink(tmp_path / "neuraldsp", tmp_path / "nam", tmp_path / "work")


# ------------------------------------------------------------------------------------------------ links
def test_classify_and_extract():
    assert classify("https://www.dropbox.com/s/abc/Pack.zip?dl=0") == ("dropbox", "https://www.dropbox.com/s/abc/Pack.zip?dl=1")
    assert classify("https://drive.google.com/file/d/1AbC_-x/view?usp=sharing") == ("drive", "https://drive.google.com/uc?export=download&id=1AbC_-x")
    assert classify("https://drive.google.com/drive/folders/1AbC") == ("unsupported", None)
    assert classify("https://mega.nz/file/xyz")[0] == "unsupported"
    assert classify("https://unity.neuraldsp.com/uploads/short-url/x.xml")[0] == "direct"
    assert classify("https://www.youtube.com/watch?v=1")[0] == "ignore"
    assert extract_urls("grab it (https://a.com/x.zip), and https://b.com/y.nam.") == ["https://a.com/x.zip", "https://b.com/y.nam"]


# ------------------------------------------------------------------------------------------------ sink / archive
def test_sink_validates_and_logs(sink, tmp_path):
    rows = sink.add(ndsp_xml(), "Take a Bite.xml", {"source": "neuraldsp_forum", "title": "Gojira preset thread"})
    assert len(rows) == 1 and rows[0]["plugin"] == "neural_dsp_gojira" and rows[0]["kind"] == "neuraldsp_xml"
    assert (tmp_path / "neuraldsp" / rows[0]["file"]).exists()
    assert sink.add(b"<html><body>Please log in</body></html>", "preset.xml") == []          # login page saved as .xml
    assert sink.add(b"not json", "amp.nam") == []
    assert sink.add(ndsp_xml(), "again.xml") == []                                          # byte-identical file stored once
    (r_nam,) = sink.add(nam(), "amp.nam", {"source": "tonehunt", "title": "Metallica Master of Puppets rhythm", "tags": ["thrash metal"]})
    (r_ir,) = sink.add(wav(), "cab.wav", {"source": "tonehunt"})
    assert r_nam["artist"] == "Metallica" and r_nam["genre"] == "metal"
    assert read_rows(tmp_path / "nam" / "manifest.jsonl")[0]["file"] == r_nam["file"] and r_ir["kind"] == "impulse_response"
    assert sink.stats["rejected"] == 2 and sink.stats["duplicate"] == 1


def test_archive_filters_junk_and_moves_only_valid(sink, tmp_path):
    z = make_zip({"Pack/Heavy.xml": ndsp_xml("Heavy", gain=0.9), "Pack/Clean.xml": ndsp_xml("Clean", gain=0.1), "__MACOSX/Pack/._Heavy.xml": b"junk",
                  "Pack/.DS_Store": b"x", "Pack/readme.txt": b"hi", "Pack/cover.png": b"png", "Pack/amp.nam": nam(), "Pack/IR/cab.wav": wav(),
                  "Pack/fake.xml": b"<html>404</html>", "../../evil.xml": ndsp_xml("Evil", gain=0.3)})
    rows = sink.add(z, "Pack.zip", {"source": "reddit"})
    assert sorted(r["original_name"] for r in rows) == ["Clean.xml", "Heavy.xml", "amp.nam", "cab.wav", "evil.xml"]
    assert all(r["archive"] == "Pack.zip" for r in rows)
    assert {p.name for p in (tmp_path / "neuraldsp").iterdir()} >= {r["file"] for r in rows if r["file"].endswith(".xml")}
    assert not (tmp_path.parent / "evil.xml").exists()                                      # path traversal never touches the disk
    assert not any(tmp_path.joinpath("work").iterdir())                                     # unpacked leftovers are removed
    assert not any(p.suffix in (".txt", ".png") for d in ("neuraldsp", "nam") for p in (tmp_path / d).iterdir())


def test_nested_corrupt_and_unnamed_downloads(sink):
    inner = make_zip({"a.nam": nam("A")})
    assert len(sink.add(make_zip({"inner.zip": inner}), "outer.zip")) == 1
    assert sink.add(b"PK\x03\x04 broken", "bad.zip") == [] and sink.stats["archive_failed"] == 1
    assert sniff_ext(ndsp_xml()) == ".xml" and sniff_ext(nam()) == ".nam" and sniff_ext(wav()) == ".wav"
    assert len(sink.add(ndsp_xml("Other", gain=0.7), "SHARED")) == 1                         # Dropbox / Drive give no extension
    assert sink.add(b"Rar!\x1a\x07\x00junk", "x.rar") == []                                   # rar without unrar tooling: logged, not fatal


# ------------------------------------------------------------------------------------------------ downloader
class FakeClient:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, data=None, cache=True):
        self.calls.append(url)
        r = self.routes.get(url)
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        return r


def fetcher_for(routes):
    c = FakeClient(routes)
    return Fetcher(make_client=lambda base: c), c


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, None)


def test_harvest_dead_links_and_ledger(sink, tmp_path):
    good = "https://files.example.com/p/Good.xml"
    routes = {good: ndsp_xml(), "https://files.example.com/p/Dead.zip": http_error(404), "https://files.example.com/p/Priv.zip": http_error(403),
              "https://files.example.com/p/Login.zip": b"<!DOCTYPE html><html>sign in</html>", "https://files.example.com/p/Net.zip": urllib.error.URLError("dns"),
              "https://files.example.com/p/Robots.zip": PermissionError("robots.txt disallows"), "https://files.example.com/p/Limit.zip": http_error(429)}
    fetcher, client = fetcher_for(routes)
    ledger = Ledger(tmp_path / "links.jsonl")
    got = {u: harvest(u, {"source": "reddit"}, fetcher, sink, ledger) for u in routes}
    assert len(got[good]) == 1
    status = {u.rsplit("/", 1)[1]: ledger.latest[u]["status"] for u in routes}
    assert status == {"Good.xml": "ok", "Dead.zip": "dead", "Priv.zip": "forbidden", "Login.zip": "html", "Net.zip": "network_error",
                      "Robots.zip": "robots_blocked", "Limit.zip": "rate_limited"}
    assert harvest("https://mega.nz/file/x", {}, fetcher, sink, ledger) == [] and ledger.latest["https://mega.nz/file/x"]["status"] == "unsupported"
    n = len(client.calls)
    for u in routes:                                                                      # second run: done links are skipped, transient ones retried
        harvest(u, {"source": "reddit"}, fetcher, sink, ledger)
    assert len(client.calls) - n == 2 and len(read_rows(tmp_path / "links.jsonl")) == len(routes) + 1 + 2


def test_drive_confirm_page_and_dropbox(sink, tmp_path):
    page = (b'<html><form action="https://drive.usercontent.google.com/download"><input type="hidden" name="id" value="F1">'
            b'<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u1"></form></html>')
    routes = {"https://drive.google.com/uc?export=download&id=F1": page,
              "https://drive.usercontent.google.com/download?id=F1&confirm=t&uuid=u1": make_zip({"a.nam": nam("D")}),
              "https://www.dropbox.com/s/k/Pack.zip?dl=1": make_zip({"b.xml": ndsp_xml("B", gain=0.2)})}
    fetcher, _ = fetcher_for(routes)
    ledger = Ledger(tmp_path / "links.jsonl")
    assert len(harvest("https://drive.google.com/file/d/F1/view", {"source": "reddit"}, fetcher, sink, ledger)) == 1
    assert len(harvest("https://www.dropbox.com/s/k/Pack.zip?dl=0", {"source": "reddit"}, fetcher, sink, ledger)) == 1


# ------------------------------------------------------------------------------------------------ reddit
def listing(*posts, after=None):
    return json.dumps({"data": {"after": after, "children": [{"data": p} for p in posts]}}).encode()


def test_reddit_parse_and_crawl(sink, tmp_path):
    post = {"id": "a1", "title": "Free Gojira presets", "selftext": "Dropbox: https://www.dropbox.com/s/k/Pack.zip?dl=0 and vid https://youtu.be/x",
            "url": "https://www.reddit.com/r/NeuralDSP/comments/a1/", "permalink": "/r/NeuralDSP/comments/a1/x/", "author": "bob", "subreddit": "NeuralDSP",
            "link_flair_text": "Presets", "created_utc": 1.0}
    gone = {**post, "id": "a2", "removed_by_category": "moderator"}
    api = reddit.API
    route = {f"{api}/r/NeuralDSP/search.json?q=preset&restrict_sr=1&sort=new&limit=100&raw_json=1": listing(post, gone)}
    rd = reddit.Reddit(FakeClient(route), token="tok")
    assert reddit.post_links(reddit.parse_listing(json.loads(listing(post)))[0][0]) == ["https://www.dropbox.com/s/k/Pack.zip?dl=0"]
    fetcher, _ = fetcher_for({"https://www.dropbox.com/s/k/Pack.zip?dl=1": make_zip({"x.xml": ndsp_xml()})})
    stats = reddit.crawl(rd, fetcher, sink, Ledger(tmp_path / "l.jsonl"), subs={"NeuralDSP": ["preset"]}, max_pages=1)
    assert stats == {"posts": 1, "links": 1, "stored": 1}
    assert sink.add(ndsp_xml(), "x.xml") == []                                           # already stored
    row = read_rows(tmp_path / "neuraldsp" / "manifest.jsonl")[0]
    assert row["source"] == "reddit" and row["tags"] == ["Presets"] and row["source_url"].endswith("/r/NeuralDSP/comments/a1/x/")


def test_reddit_needs_token(monkeypatch):
    monkeypatch.delenv("REDDIT_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="robots.txt"):
        reddit.Reddit()


def test_reddit_failed_search_is_not_fatal():
    rd = reddit.Reddit(FakeClient({}), token="tok")                                      # every search 404s
    assert list(rd.search("NeuralDSP", "preset", 2)) == []


# ------------------------------------------------------------------------------------------------ neural forum
def test_forum_crawl(sink, tmp_path):
    B = neural_forum.BASE
    topic = {"title": "Archetype Gojira Preset Thread", "post_stream": {"stream": [11, 12, 13], "posts": [
        {"id": 11, "post_number": 1, "username": "jacob", "created_at": "2024-01-02T00:00:00Z", "cooked":
         '<a href="/uploads/short-url/AAA.xml">Take a Bite.xml</a> <a href="/u/jacob">jacob</a> <a href="/t/other/9">x</a>'},
        {"id": 12, "post_number": 2, "username": "kay", "cooked": '<a href="https://www.dropbox.com/s/q/Pack.zip?dl=0">zip</a> <a href="https://youtu.be/1">v</a>'
         ' <a href="https://drive.google.com/file/d/ABC/view">drive</a>'}]}}
    more = {"post_stream": {"posts": [{"id": 13, "post_number": 3, "username": "li", "cooked": '<a href="/uploads/short-url/GONE.zip">z</a>'}]}}
    routes = {f"{B}/c/18.json?page=0": json.dumps({"topic_list": {"more_topics_url": "/c/plugins/18?page=1", "topics": [
                  {"id": 5, "title": "Archetype Gojira Preset Thread"}, {"id": 6, "title": "iLok installation failure"}]}}).encode(),
              f"{B}/c/18.json?page=1": json.dumps({"topic_list": {"topics": [{"id": 5, "title": "Archetype Gojira Preset Thread"}]}}).encode(),
              f"{B}/t/5.json": json.dumps(topic).encode(), f"{B}/t/5/posts.json?post_ids%5B%5D=13": json.dumps(more).encode(),
              f"{B}/uploads/short-url/AAA.xml": ndsp_xml("Take a Bite", gain=0.4), "https://www.dropbox.com/s/q/Pack.zip?dl=1": make_zip({"n.nam": nam("N")})}
    client = FakeClient(routes)
    fetcher = Fetcher(clients={B: client}, make_client=lambda base: client)
    ledger = Ledger(tmp_path / "l.jsonl")
    stats = neural_forum.crawl(sink, fetcher, ledger, client)
    assert stats["listed"] == 2 and stats["skipped_title"] == 1 and stats["threads"] == 1 and stats["posts"] == 3 and stats["stored"] == 2
    assert stats["skipped_drive"] == 1 and ledger.latest["https://drive.google.com/file/d/ABC/view"]["status"] == "robots_blocked"
    assert not any("drive" in c for c in client.calls)                                    # Drive is never requested
    assert ledger.latest[f"{B}/uploads/short-url/GONE.zip"]["status"] == "dead"          # dead attachment logged, crawl continues
    row = next(r for r in read_rows(tmp_path / "neuraldsp" / "manifest.jsonl"))
    assert row["author"] == "jacob" and row["plugin"] == "neural_dsp_gojira" and row["plugin_thread"] == "Archetype Gojira"


def test_forum_bad_json_is_not_fatal():
    assert neural_forum.get_json(FakeClient({"https://x/y": b"<html>"}), "https://x/y") is None


# ------------------------------------------------------------------------------------------------ tonehunt helpers
def test_tonehunt_helpers():
    assert tonehunt.search_url("JCM800", 2) == "https://www.tone3000.com/search?query=JCM800&page=2"
    assert tonehunt.queries_for(["Metallica"], ["JCM800"]) == ["JCM800", "Metallica tone"]
    assert tonehunt.queries_for(None, None) == tonehunt.AMP_QUERIES


# ------------------------------------------------------------------------------------------------ parsers
def test_neural_xml_parser(tmp_path):
    p = tmp_path / "a.xml"
    p.write_bytes(ndsp_xml("Brutal", ampType=2.0, cab1Level=0.75, cabActive=1.0, ampsActive=0.0))
    pre = parse_neural_xml(p)
    assert (pre.plugin, pre.name, pre.version) == ("neural_dsp_gojira", "Brutal", "1.0.0")
    assert pre.params == {"ampType": 2.0, "cab1Level": 0.75, "cabActive": 1.0, "ampsActive": 0.0}
    assert pre.active("cab") is True and pre.active("amps") is False and pre.active("nope") is None and pre.warnings
    t = tmp_path / "b.xml"
    t.write_text('<?xml version="1.0"?><PLUGIN presetName="X"><PARAM id="gain" value="0.3"/><PARAM id="bass" value="x"/></PLUGIN>')
    assert parse_neural_xml(t).params == {"gain": 0.3}
    (tmp_path / "c.xml").write_bytes(b"neural_dsp_x\x00no params")
    with pytest.raises(ValueError):
        parse_neural_xml(tmp_path / "c.xml")


def test_neural_xml_v2_container(tmp_path, sink):
    data = (b"gojira-X\x00\x01\x03version\x00\x01\x07\x051.0.2\x00name\x00\x01\x11\x05Clean Rvb\x00\x00parameters\x00\x01\x07"
            b"inputGain\x00\x01\x03\x050\x00outputGain\x00\x01\x08\x055.0625\x00doublerActive\x00\x01\x07\x05false\x00")
    assert len(sink.add(data, "x.xml")) == 1
    p = tmp_path / "v2.xml"
    p.write_bytes(data)
    pre = parse_neural_xml(p)
    assert (pre.container, pre.plugin, pre.name, pre.version) == ("juce_binary_v2", "gojira-X", "Clean Rvb", "1.0.2")
    assert pre.params == {"version": 1.0, "inputGain": 0.0, "outputGain": 5.0625, "doublerActive": 0.0} or pre.params["outputGain"] == 5.0625


def test_nam_and_ir_parsers(tmp_path):
    (tmp_path / "a.nam").write_bytes(nam())
    m = parse_nam(tmp_path / "a.nam", load_weights=True)
    assert (m.architecture, m.name, m.gear, m.n_weights, m.sample_rate, m.weights) == ("WaveNet", "Metallica Rhythm", "Mesa Rectifier", 2, 48000, [0.1, 0.2])
    (tmp_path / "b.nam").write_text('{"architecture": "LSTM", "weights": []}')
    assert parse_nam(tmp_path / "b.nam").warnings
    (tmp_path / "c.nam").write_text('{"hello": 1}')
    with pytest.raises(ValueError):
        parse_nam(tmp_path / "c.nam")
    (tmp_path / "ir.wav").write_bytes(wav(0.5))
    ir = parse_ir(tmp_path / "ir.wav")
    assert (ir.channels, ir.sample_rate, ir.sample_bits) == (1, 48000, 16) and abs(ir.seconds - 0.5) < 1e-3


# ------------------------------------------------------------------------------------------------ manual packs
def test_ingest_local(tmp_path):
    from patch_scraper.archive import ingest_local

    stage = tmp_path / "stage" / "Metallica Black Album pack"
    stage.mkdir(parents=True)
    (stage / "Enter Sandman.xml").write_bytes(ndsp_xml("Enter Sandman", gain=0.8))
    (stage / "bundle.zip").write_bytes(make_zip({"Sad But True.xml": ndsp_xml("Sad But True", gain=0.9), "__MACOSX/._x.xml": b"j", "read.txt": b"t"}))
    (stage / "notes.txt").write_text("ignore me")
    sink = Sink(tmp_path / "neuraldsp", tmp_path / "nam", tmp_path / "work")
    stats = ingest_local(tmp_path / "stage", sink, source="mega")
    assert stats["files"] == 2 and stats["stored"] == 2
    rows = read_rows(tmp_path / "neuraldsp" / "manifest.jsonl")
    assert {r["source"] for r in rows} == {"mega"} and all(r["artist"] == "Metallica" and r["genre"] == "metal" for r in rows)
    assert ingest_local(tmp_path / "stage", sink)["stored"] == 0                          # re-run: nothing new


def test_http_gives_up_on_long_retry_after(monkeypatch):
    import email.message
    from patch_scraper import http

    msg = email.message.Message()
    msg["Retry-After"] = "70103"

    def boom(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", msg, None)

    monkeypatch.setattr(http.urllib.request, "urlopen", boom)
    monkeypatch.setattr(http.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError(f"slept {s}")))
    c = http.PoliteClient("https://api.example.com", delay_sec=0, check_robots=False, max_retry_after=60)
    with pytest.raises(urllib.error.HTTPError):
        c.get("https://api.example.com/x", cache=False)
