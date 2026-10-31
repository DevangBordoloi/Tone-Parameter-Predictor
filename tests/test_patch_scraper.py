"""Offline tests for patch_scraper. HTML fixtures are trimmed copies of the real GuitarPatches markup (as of 2026-10);
the network is replaced by a fake client and fake search / download functions."""

from __future__ import annotations

import json

from patch_corpus.metadata import find_meta
from patch_scraper import audio, balance, genre
from patch_scraper.artists import FamousArtists
from patch_scraper.guitarpatches import parse_detail, parse_listing as gp_listing
from patch_scraper.manifest import read_rows, write_rows
from patch_scraper.sites import Item, crawl, clean_song

PATCH = json.dumps({"data": {"tone": {}}})


def gp_cell(pid, name, artist, unit="Katana", date="2026-06-21"):
    return (f'<div class="cell"><div class="card" onclick="href=window.location=\'patches.php?mode=show&unit={unit}&ID={pid}\';">'
            f'<h1 class="title is-5">{name}</h1><img/><h2 class="subtitle is-6">{artist}</h2><p class="is-size-7">{date}</p></div></div>')


GP_DETAIL = '''<label class="label mb-0">Contributor</label><h3 class="title is-6 mt-0">MetsxxFan02</h3>
<p class="title is-5 has-text-centered mb-0">Rating</p><h6 class='title is-4 has-text-centered'>&nbsp;<p class='is-size-7'>%s</p></h6>
<a href="download.php?unit=Katana&mode=download&ID=14879" class='button is-link is-large'>Download patch</a>'''


def test_guitarpatches_parse():
    items = gp_listing("<div class=\"grid\">" + gp_cell(1, "Sweet Child O Mine", "Guns N Roses") + gp_cell(2, "U2 part 2", "U2")
                       + gp_cell(3, "Metallica - One", "Metallica") + "</div>", "Katana")
    rows = [i.row for i in items]
    assert [r["song"] for r in rows] == ["Sweet Child O Mine", None, "One"]
    assert rows[0]["source_url"] == "https://guitarpatches.com/patches.php?mode=show&unit=Katana&ID=1" and items[0].needs_detail
    d = parse_detail(GP_DETAIL % "No ratings yet")
    assert d["rating"] is None and d["author"] == "MetsxxFan02"
    assert d["download"] == "https://guitarpatches.com/download.php?unit=Katana&mode=download&ID=14879"
    assert parse_detail(GP_DETAIL % "4.5 (12 votes)")["rating"] == 4.5
    assert parse_detail(GP_DETAIL % "4.5 (12 votes)")["rating_count"] == 12


def test_clean_song_and_famous_matching():
    assert clean_song("Metallica - One", "Metallica") == "One" and clean_song("Enter Sandman", "Metallica") == "Enter Sandman"
    assert clean_song("U2 part 2", "U2") is None and clean_song("U2", "U2") is None and clean_song("clean", "Muse") is None
    assert clean_song("Queen of Hearts", "Queen") == "Queen of Hearts"
    f = FamousArtists()
    assert f.match("Metallica - Master of Puppets") == ("metal", "Metallica")
    assert f.match("Queen of Hearts") is None and f.match("Queen") == ("rock", "Queen")
    assert f.match("guns n roses")[1] == "Guns N' Roses" and f.match("Eric Clapton")[0] == "blues"
    assert f.match("Taylor Swift")[0] == "pop" and f.match("Miles Davis")[0] == "jazz" and f.match("Nobody Famous") is None
    genres = set(f.table)
    assert {"pop", "rock", "r&b/soul", "metal", "jazz", "country", "funk", "punk", "blues"} <= genres


class FakeSite:
    """Adapter stand-in: listing pages are fixed lists of Items; downloads come from a dict."""
    name, file_ext, base_url, cookie_env = "fake", ".hlx", "https://x", None

    def __init__(self, pages):
        self.listing = pages

    def pages(self, client, query):
        yield from self.listing

    def enrich(self, client, item):
        item.row["rating"] = item.row.get("rating")
        item.needs_detail = False


class FakeClient:
    def __init__(self, files):
        self.files = files

    def get(self, url, **kw):
        return self.files[url]


def item(n, artist, song, rating=4.0, dl=True, **kw):
    row = {"source_id": f"fake-{n}", "source": "fake", "source_url": f"https://x/{n}", "artist": artist, "song": song,
           "rating": rating, **kw}
    return Item(url=row["source_url"], row=row, download_url=f"https://x/f{n}" if dl else None)


def test_crawl_engine(tmp_path):
    files = {"https://x/f1": PATCH.encode(), "https://x/f2": PATCH.encode(),                         # f2 is a byte-identical duplicate
             "https://x/f3": b"<html>log in</html>", "https://x/f5": PATCH.replace("tone", "t2").encode(),
             "https://x/f6": PATCH.replace("tone", "t3").encode()}
    site = FakeSite([[item(1, "Metallica", "One"), item(2, "Metallica", "Sad But True"), item(3, "Queen", "Bohemian Rhapsody")],
                     [item(4, "Nobody", "Song"), item(5, "Muse", "Hysteria", rating=1.0), item(6, "Muse", None),
                      item(7, "Pantera", "Walk", rating=None, dl=False)]])
    famous = FamousArtists()
    st = crawl(site, tmp_path, FakeClient(files), famous=famous, min_rating=3.5, progress=lambda s: None)
    assert (st.new, st.duplicate, st.bad_file, st.not_wanted, st.low_rating, st.no_song, st.login_required) == (2, 1, 1, 1, 1, 1, 1)
    # new = Metallica "One" (file) + Pantera (no download link -> metadata only); the duplicate and the bad download are not stored
    rows = {r["source_id"]: r for r in read_rows(tmp_path / "manifest.jsonl")}
    assert set(rows) == {"fake-1", "fake-7"}
    assert rows["fake-1"]["genre"] == "metal" and rows["fake-1"]["famous_artist"] == "Metallica"
    assert (tmp_path / rows["fake-1"]["file"]).read_text() == PATCH
    assert rows["fake-7"]["file"] is None and rows["fake-7"]["status"] == "login_required"       # metadata kept without the file
    # the manifest is read by patch_corpus.metadata
    meta = find_meta(tmp_path / rows["fake-1"]["file"])
    assert (meta.artist, meta.song, meta.rating, meta.meta_origin) == ("Metallica", "One", 4.0, "manifest")
    # re-run with the login available: the metadata-only row gets its file, nothing else is fetched twice
    files["https://x/f7"] = PATCH.replace("tone", "t7").encode()
    site2 = FakeSite([[item(1, "Metallica", "One"), item(7, "Pantera", "Walk", rating=None)]])
    st2 = crawl(site2, tmp_path, FakeClient(files), famous=famous, progress=lambda s: None)
    assert (st2.filled, st2.already, st2.new) == (1, 1, 0)
    assert read_rows(tmp_path / "manifest.jsonl")[1]["file"].endswith(".hlx")


def test_crawl_all_artists_and_limit(tmp_path):
    files = {f"https://x/f{n}": PATCH.replace("tone", f"t{n}").encode() for n in range(1, 5)}
    site = FakeSite([[item(n, f"Artist{n}", f"Song{n}") for n in range(1, 5)]])
    st = crawl(site, tmp_path, FakeClient(files), famous=None, limit=2, progress=lambda s: None)
    assert st.new == 2 and len(read_rows(tmp_path / "manifest.jsonl")) == 2
    only = FakeSite([[item(n, f"Artist{n}", None, dl=False) for n in range(5, 7)]])
    assert crawl(only, tmp_path / "b", FakeClient(files), famous=None, progress=lambda s: None).no_song == 2
    st = crawl(only, tmp_path / "c", FakeClient(files), famous=None, allow_artist_only=True, progress=lambda s: None)
    assert (st.no_song, st.new) == (0, 2)


def test_genre_mapping_and_tagging(tmp_path):
    assert genre.map_tags([("progressive metal", 10), ("rock", 3)]) == "metal"
    assert genre.map_tags([("blues rock", 5), ("blues", 5)]) == "blues"
    assert genre.map_tags([("seen live", 2)]) == "other" and genre.map_tags([]) == "other"
    write_rows(tmp_path / "manifest.jsonl", [
        {"file": "a.hlx", "artist": "Foo", "song": "S1"}, {"file": "b.hlx", "artist": "foo", "song": "S2"},
        {"file": None, "artist": "Known", "song": "S3", "genre": "pop"}])
    calls = []
    totals = genre.tag(tmp_path, fetch=lambda a: calls.append(a) or [("jazz fusion", 7), ("fusion", 4)])
    assert totals == {"fusion": 2, "pop": 1}
    rows = read_rows(tmp_path / "manifest.jsonl")
    assert rows[0]["genre"] == "fusion" and "jazz fusion" in rows[0]["tags"] and rows[2]["genre"] == "pop"
    assert len(calls) == 1                                                   # one lookup for "Foo"/"foo", none for the pre-tagged row


def test_balance_report_and_cap(tmp_path):
    src, out = tmp_path / "s", tmp_path / "b"
    src.mkdir()
    rows = []
    for i, (g, r) in enumerate([("rock", 5), ("rock", 3), ("rock", 4), ("jazz", 4)]):
        (src / f"{i}.hlx").write_text(PATCH)
        rows.append({"file": f"{i}.hlx", "artist": f"a{i}", "song": f"s{i}", "genre": g, "rating": r, "source": "x"})
    rows.append({"file": None, "artist": "m", "song": "m", "genre": "rock", "rating": 5, "source": "x"})       # metadata only
    write_rows(src / "manifest.jsonl", rows)
    c = balance.counts(src)
    assert c["rock"]["patches"] == 3 and c["rock"]["songs"] == 3 and c["jazz"]["patches"] == 1           # the metadata-only row is not a preset
    assert "jazz" in balance.report(src, 2)
    balance.cap(src, 2, out)
    kept = read_rows(out / "manifest.jsonl")
    assert sorted(r["rating"] for r in kept if r["genre"] == "rock") == [4, 5] or len([r for r in kept if r["genre"] == "rock"]) <= 2
    assert all((out / r["file"]).exists() for r in kept)


def test_youtube_scoring():
    good = {"title": "Eric Johnson - Cliffs of Dover (Official Audio)", "duration": 243, "channel": "Eric Johnson - Topic"}
    live = {"title": "Eric Johnson - Cliffs of Dover LIVE 1990", "duration": 243, "channel": "Eric Johnson"}
    lesson = {"title": "Cliffs of Dover guitar lesson", "duration": 600, "channel": "Some Teacher"}
    wrong_len = {"title": "Eric Johnson - Cliffs of Dover", "duration": 400, "channel": "Eric Johnson"}
    s = lambda c, d=241.0: audio.score_candidate(c, "Cliffs of Dover", "Eric Johnson", d)        # noqa: E731
    assert s(good) >= 0.95 and s(live) == 0 and s(lesson) == 0
    assert s(wrong_len) < 0.8 <= s(good)
    assert audio.score_candidate(good, "Cliffs of Dover", "Eric Johnson", None) >= 0.95        # renormalised without a duration
    assert audio.score_candidate({**live, "title": "Live and Let Die"}, "Live and Let Die", "Wings", 190) > 0   # 'live' inside the title


def test_spotify_pick():
    payload = {"tracks": {"items": [
        {"id": "1", "name": "Cliffs of Dover - Live", "artists": [{"name": "Other"}], "duration_ms": 1000, "external_ids": {}},
        {"id": "2", "name": "Cliffs of Dover", "artists": [{"name": "Eric Johnson"}], "duration_ms": 243000, "external_ids": {"isrc": "USX"}}]}}
    t = audio.pick_spotify_track(payload, "Cliffs Of Dover", "Eric Johnson")
    assert (t.spotify_id, t.isrc, t.duration_s) == ("2", "USX", 243.0)
    assert audio.pick_spotify_track(payload, "Totally Different", "Nobody") is None


def test_audio_match_writes_urls(tmp_path):
    presets, urls = tmp_path / "presets" / "katana", tmp_path / "urls"
    write_rows(presets / "manifest.jsonl", [
        {"file": None, "artist": "Eric Johnson", "song": "Cliffs of Dover", "genre": "fusion"},
        {"file": "p2.hlx", "artist": "Eric Johnson", "song": "Cliffs of Dover", "genre": "fusion"},        # same song, second patch
        {"file": None, "artist": "Nobody", "song": "Nothing", "genre": "other"}])

    def fake_search(q):
        return [{"title": "Eric Johnson - Cliffs of Dover", "duration": 243, "channel": "Eric Johnson - Topic", "url": "https://yt/x"},
                {"title": "Artist - Song karaoke", "duration": 200, "channel": "k", "url": "https://yt/k"}]

    fake_spotify = lambda song, artist: audio.Track("sp1", song, artist, "ISRC1", 243.0) if artist == "Eric Johnson" else None   # noqa: E731
    stats = audio.match(tmp_path / "presets", urls, search=fake_search, spotify=fake_spotify)
    assert (stats["songs"], stats["matched"], stats["unmatched"], stats["no_spotify"]) == (2, 1, 1, 1)
    songs = {r["artist"]: r for r in read_rows(urls / "song_urls.jsonl")}
    ej = songs["Eric Johnson"]
    assert ej["status"] == "matched" and ej["youtube_url"] == "https://yt/x" and ej["spotify_url"] == "https://open.spotify.com/track/sp1"
    assert ej["isrc"] == "ISRC1" and ej["match_score"] >= 0.9 and ej["song_audio"] is None            # no audio unless asked
    assert songs["Nobody"]["status"] == "unmatched" and songs["Nobody"]["youtube_url"] is None
    assert len(read_rows(urls / "unmatched.jsonl")) == 1
    rows = read_rows(presets / "manifest.jsonl")
    assert rows[0]["youtube_url"] == rows[1]["youtube_url"] == "https://yt/x"                        # both patches of the song get the URLs
    # a re-run does not search again
    again = audio.match(tmp_path / "presets", urls, search=lambda q: 1 / 0, spotify=fake_spotify)
    assert again["already"] == 2 and again["songs"] == 0
    # audio is downloaded only on request
    write_rows(presets / "manifest.jsonl", [{"file": None, "artist": "Eric Johnson", "song": "Cliffs of Dover"}])
    got = []
    audio.match(tmp_path / "presets", tmp_path / "urls2", search=fake_search, spotify=fake_spotify, download_audio=True,
                download=lambda url, dur, name: got.append((url, dur, name)) or tmp_path / "a.wav")
    assert got == [("https://yt/x", 243.0, "eric-johnson-cliffs-of-dover")]



def test_audio_spotify_429_falls_back_and_retry_unmatched(tmp_path):
    import urllib.error

    presets, urls = tmp_path / "presets" / "katana", tmp_path / "urls"
    write_rows(presets / "manifest.jsonl", [{"file": "a.tsl", "artist": "Eric Johnson", "song": "Cliffs of Dover"},
                                            {"file": "b.tsl", "artist": "Muse", "song": "Hysteria"}])
    calls = []

    def limited(song, artist):
        calls.append(song)
        raise urllib.error.HTTPError("u", 429, "rate", {}, None)

    def search(q):
        if "Hysteria" in q:
            return [{"title": "Muse - Hysteria", "duration": 227, "channel": "Muse - Topic", "url": "https://yt/h"}]
        return [{"title": "Cliffs of Dover live 1990", "duration": 300, "channel": "x", "url": "https://yt/live"}]

    stats = audio.match(tmp_path / "presets", urls, search=search, spotify=limited)
    assert stats["spotify_fallback"] == 1 and len(calls) == 1                          # Spotify is not asked again after the 429
    assert (stats["matched"], stats["unmatched"]) == (1, 1)
    assert [r["artist"] for r in read_rows(urls / "unmatched.jsonl")] == ["Eric Johnson"]
    better = lambda q: [{"title": "Eric Johnson - Cliffs of Dover", "duration": 243, "channel": "Eric Johnson - Topic", "url": "https://yt/c"}]  # noqa: E731
    again = audio.match([tmp_path / "presets"], urls, search=better, use_spotify=False, retry_unmatched=True)
    assert again["songs"] == 1 and again["matched"] == 1 and again["already"] == 1
    assert read_rows(urls / "unmatched.jsonl") == []                                     # rewritten from the latest attempts
    rows = read_rows(presets / "manifest.jsonl")
    assert rows[0]["youtube_url"] == "https://yt/c" and rows[1]["youtube_url"] == "https://yt/h"


def test_audio_network_failure_is_retried(tmp_path):
    presets, urls = tmp_path / "presets" / "katana", tmp_path / "urls"
    write_rows(presets / "manifest.jsonl", [{"file": "a.tsl", "artist": "Muse", "song": "Hysteria"}])

    def down(q):
        raise OSError("[WinError 10013] socket blocked")

    s = audio.match(tmp_path / "presets", urls, search=down, use_spotify=False)
    assert s["search_failed"] == 1 and read_rows(urls / "song_urls.jsonl")[0]["status"] == "search_failed"
    ok = lambda q: [{"title": "Muse - Hysteria", "duration": 227, "channel": "Muse - Topic", "url": "https://yt/h"}]  # noqa: E731
    again = audio.match(tmp_path / "presets", urls, search=ok, use_spotify=False)            # no flag needed: failures are always retried
    assert again["matched"] == 1 and read_rows(presets / "manifest.jsonl")[0]["youtube_url"] == "https://yt/h"


def test_itunes_and_deezer_pickers():
    it = {"results": [{"trackName": "Hysteria", "artistName": "Muse", "trackTimeMillis": 227000, "trackId": 7},
                      {"trackName": "Hysteria (Live)", "artistName": "Def Leppard", "trackTimeMillis": 300000, "trackId": 8}]}
    tr = audio.pick_itunes_track(it, "Hysteria", "Muse")
    assert (tr.title, tr.duration_s, tr.source) == ("Hysteria", 227.0, "itunes")
    assert audio.pick_itunes_track(it, "Something Else Entirely", "Nobody") is None
    dz = {"data": [{"id": 5, "title": "Hysteria - Remastered", "duration": 227, "artist": {"name": "Muse"}}]}
    assert (audio.pick_deezer_track(dz, "Hysteria", "Muse").duration_s, audio.pick_deezer_track(dz, "Hysteria", "Muse").source) == (227.0, "deezer")
    assert audio.pick_deezer_track({"data": [{"id": 5, "title": "Hysteria", "duration": 0, "artist": {"name": "Muse"}}]}, "Hysteria", "Muse") is None


def test_duration_lookup_switches_off_a_refusing_provider():
    import urllib.error

    class Fake:
        def __init__(self, reply):
            self.reply, self.calls = reply, 0

        def get(self, url, **kw):
            self.calls += 1
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    look = audio.DurationLookup.__new__(audio.DurationLookup)
    itunes = Fake(urllib.error.HTTPError("u", 403, "forbidden", {}, None))
    deezer = Fake(json.dumps({"data": [{"id": 5, "title": "Hysteria", "duration": 227, "artist": {"name": "Muse"}}]}).encode())
    look.providers = [("itunes", itunes, audio.ITUNES_SEARCH, audio.pick_itunes_track), ("deezer", deezer, audio.DEEZER_SEARCH, audio.pick_deezer_track)]
    look.dead, look.errors = set(), {}
    assert look("Hysteria", "Muse").source == "deezer" and look("Hysteria", "Muse").source == "deezer"
    assert itunes.calls == 1 and look.dead == {"itunes"}                                  # refused once, not asked again


def test_reference_duration_rejects_a_wrong_length_and_reverify_rechecks(tmp_path):
    presets, urls = tmp_path / "presets" / "katana", tmp_path / "urls"
    write_rows(presets / "manifest.jsonl", [{"file": "a.tsl", "artist": "Muse", "song": "Hysteria"}])
    long_cut = lambda q: [{"title": "Muse - Hysteria", "duration": 600, "channel": "Muse - Topic", "url": "https://yt/long"}]   # noqa: E731
    first = audio.match(tmp_path / "presets", urls, search=long_cut, use_spotify=False)
    assert first["matched"] == 1                                                         # no duration known: the title alone is enough
    look = lambda song, artist: audio.Track("1", song, artist, None, 227.0, "itunes")      # noqa: E731
    again = audio.match(tmp_path / "presets", urls, search=long_cut, use_spotify=False, duration_lookup=look)
    assert again["already"] == 1 and again["songs"] == 0                                 # a match is kept unless asked to re-verify
    rv = audio.match(tmp_path / "presets", urls, search=long_cut, use_spotify=False, duration_lookup=look, reverify=True)
    assert rv["songs"] == 1 and rv["unmatched"] == 1 and rv["duration_itunes"] == 1       # 600 s vs 227 s: rejected once the length is known
    row = read_rows(urls / "song_urls.jsonl")[-1]
    assert (row["status"], row["duration_source"], row["duration_s"], row["youtube_candidate"]) == ("unmatched", "itunes", 227.0, "https://yt/long")
    assert read_rows(presets / "manifest.jsonl")[0]["youtube_url"] is None
    right = lambda q: [{"title": "Muse - Hysteria", "duration": 229, "channel": "Muse - Topic", "url": "https://yt/ok"}]    # noqa: E731
    fixed = audio.match(tmp_path / "presets", urls, search=right, use_spotify=False, duration_lookup=look, retry_unmatched=True)
    assert fixed["matched"] == 1
    m = read_rows(presets / "manifest.jsonl")[0]
    assert (m["youtube_url"], m["duration_source"]) == ("https://yt/ok", "itunes")


def test_reference_pick_accepts_a_title_inside_a_noisy_preset_name():
    it = {"results": [{"trackName": "Nothing Else Matters", "artistName": "Metallica", "trackTimeMillis": 388000, "trackId": 1},
                      {"trackName": "One", "artistName": "Metallica", "trackTimeMillis": 447000, "trackId": 2}]}
    assert audio.pick_itunes_track(it, "Nothing Else Matters Cln & Dst", "Metallica").duration_s == 388.0
    assert audio.pick_itunes_track(it, "Nothing Else Matters Cln & Dst", "Some Other Band") is None          # the artist has to agree
    assert audio.pick_itunes_track(it, "Someone Won the lottery", "Metallica") is None                         # 'One' is not a word of it
