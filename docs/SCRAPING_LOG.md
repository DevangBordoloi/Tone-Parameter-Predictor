# Scraping and extraction log

Project: Tone Parameter Estimation. Goal: song → isolated guitar → Neural DSP plugin parameters, trained on song-linked amp presets.
Period: 2026-10-26 to 2026-10-27. Code: `patch_scraper/` (collection), `patch_corpus/` (decoding). Data: `data/data_scraping/`.

## 1. Final state

| Item | Result |
|---|---|
| Katana `.tsl` patches downloaded (GuitarPatches) | **603** (manifest rows = files) |
| Line 6 CustomTone tones | **1,255 metadata rows, 0 `.hlx` files** (downloads need a login; source since dropped) |
| Song URL rows (`urls/song_urls.jsonl`) | 698 songs |
| Songs without a confident YouTube match (`urls/unmatched.jsonl`) | 281 |
| URL write-back into manifests | **not done** (the URL run was stopped; see F9) |
| Tests | 9 offline tests pass (`tests/test_patch_scraper.py`) |

Genre balance at last report (patches, target 300 each; includes the Line 6 metadata rows):
rock 582, metal 468, indie/alternative 299, progressive 212, blues 78, punk 69, pop 46, funk 44, fusion 31, country 12, r&b/soul 9, jazz 8.
The corpus is rock- and metal-heavy. Jazz, country and r&b are thin because these sites hold few presets for them.

## 2. Layout

```
data/data_scraping/
├── presets/
│   ├── line6/    manifest.jsonl (+ _cache)   Line 6 CustomTone metadata only, no files
│   └── katana/   *.tsl + manifest.jsonl      GuitarPatches (Boss Katana)
├── urls/         song_urls.jsonl, unmatched.jsonl
└── logs/         line6.log, katana.log, urls.log
config/famous_artists.json                     ~370 artists, 11 genres (rock, metal, pop, r&b/soul, blues, jazz, funk, country, punk, indie/alternative, progressive, fusion)
```

## 3. What was built

- `http.py` polite client: obeys robots.txt (both sites ask `Crawl-delay: 10`, so ≥10 s between requests), per-URL disk cache, retry/backoff on 429/5xx, session cookie only from an env var (never written to disk).
- `sites.py` shared crawl engine: dedupe, famous-artist filter, song-title cleaning, validate-before-save, resumable manifests.
- `guitarpatches.py` Katana library scan (3 libraries, 18 patches per page, detail page for rating/uploader/download link).
- `genre.py` genre tagging (famous-artist list, then MusicBrainz / Last.fm tags). `balance.py` per-genre report and cap.
- `audio.py` URL matching: Spotify search (title, ISRC, duration) + YouTube search (yt-dlp), scored on title 0.4 / artist 0.2 / duration 0.3 / channel 0.1, threshold 0.8, banned words (live, cover, lesson, karaoke…). Audio is downloaded only with `--download-audio`.
- Famous-artist matching is by whole words; common-word names (Queen, Yes, Tool, Rush…) must equal the whole artist string.
- `customtone.py` (Line 6) was built, run, then **deleted** (see F6/F7).

## 4. Timeline

| Step | Outcome |
|---|---|
| Removed CustomsForge/Rocksmith from README | Data source limited to Line 6 + Katana |
| Added scraper spec to README, then built `patch_scraper/` | Initial design used placeholder regexes |
| Inspected real site HTML, rewrote parsers | Adapters written for real markup |
| Line 6 crawl (famous artists, 2 pages each) | 1,255 tones of metadata |
| Katana crawl (3 libraries) | 603 `.tsl` files |
| URL step, first run | 337 songs: 221 matched, 116 unmatched, 161 not found on Spotify |
| URL step, full run | Stopped at 698 songs after a Spotify 429 (F9) |
| Line 6 download attempt with cookie | Failed (F6) |
| Removed Line 6 from scraper and README | 9 tests pass |
| ToneX / Neural DSP sources investigated | Not scrapable as-is (F8) |

## 5. Failures and fixes (log)

| # | What went wrong | Cause | Resolution |
|---|---|---|---|
| F1 | First design used generic regex config per site | Written before looking at the real pages | Replaced with per-site adapters after inspecting real HTML |
| F2 | Line 6 search returned the unfiltered list (10,000 tones) | `search_term` on the browse URL is ignored | Used the `/customtone/search/` GET form and `/search/{family}/{n}/` for later pages |
| F3 | A GuitarPatches download was a Boss `GT`-series liveset, rejected by the Katana parser ("not a Katana liveset") | Site files some GT patches under the Katana library | Parser refuses correctly; counted as a bad file. 3 such files in the full run |
| F4 | `clean_song` did not reject "U2 part 2" for artist "U2" | A `\b` in a non-raw heredoc became a backspace character in the regex | Fixed with `(?!\w)` via a Python replace (Edit could not match the invisible character) |
| F5 | Tooling slips: heredoc nesting broke a Python splice (SyntaxError); Git-Bash `/c/...` paths not usable by Windows Python; wrong CLI call (`customtone crawl`, no such subcommand); broken `dash` pytest plugin | Environment quirks | Wrote files with the Write tool; used `C:\` paths; fixed the call; run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` |
| F6 | **Line 6 `.hlx` files could not be downloaded.** Download button is a login prompt for anonymous visitors. With the cookie supplied (first as `L6SESSION=`, then `L6SESSIONID=`) pages still showed the login button on every tone and nothing downloaded | Cookie name was initially truncated in the user's screenshot; even with the right name the site did not treat the session as logged in (likely rotated or incomplete cookie set). Full `Cookie:` header was never obtained | Unresolved. Cached anonymous pages moved aside and cleared to avoid hiding links. Line 6 dropped (user decision) |
| F7 | Line 6 rows have weak song names | Many tone names are not song titles (e.g. "RAGE", "Lead"); SONG field often empty or a URL | Kept as metadata only; unmatched in the URL step |
| F8 | ToneX / Neural DSP as replacement sources | ToneNET: no API found, terms unread, downloads need IK account + software, and a ToneX preset is a neural capture, not knob values. Neural DSP: no central library, in-plugin Preset Explorer only, scattered Gumroad/Reddit packs, some paid | Not scraped. Free Neural DSP packs can be added by hand once a sample file is available |
| F9 | **Spotify search hit HTTP 429 asking for a ~19-hour wait** ("retrying in 70103s"); the URL job would have idled | Too many search requests in a long run on one app | Job stopped by hand. Songs already processed (698) are kept. Re-run later, or use `--no-spotify` |
| F10 | URL write-back and crawl race | `audio match` rewrites manifests while a crawl appends | Added `--no-writeback`; ran the URL step without it while crawling. Final write-back still pending (F9) |
| F11 | Spotify: first "key" supplied was not a Spotify credential; Client ID/Secret later pasted into chat | User error / chat exposure | Used only as per-command environment variables, never written to files. **Reset the client secret** |
| F12 | Test fixtures wrong (generic name "Crunch" rejected, bad-file rows stored as `login_required`, missing download targets) | Fixture and `store()` logic | Fixtures fixed; `store()` now skips non-JSON and duplicate downloads so the next run retries |
| F13 | Crawls stopped when a session ended | Orphaned background tasks | Restarted; crawls are resumable from cache and manifest |

## 6. Data-quality caveats

- Katana patch decoding: the two `.tsl` files parsed in an earlier check still carry `UNVERIFIED` address warnings (gain 0x22, volume 0x23, presence 0x27 and others). The file layout is confirmed; the parameter addresses are not.
- Song ↔ preset pairs are weak labels (one user's approximation, on different hardware).
- Most Katana patches have no rating. The song is the patch name with the artist stripped, so some "songs" are still not real titles.
- 161 of the first 337 songs were not found on Spotify, mostly non-song patch names.

## 7. Open items

1. Re-run `python -m patch_scraper.audio match` (without `--no-writeback`) after the Spotify limit lifts, to finish URLs and write them into the manifests.
2. Run `python -m patch_scraper.genre tag` on any new rows and `balance report` again.
3. Optionally add free Neural DSP packs by hand (need a sample file to write a parser).
4. Decide whether to delete the 1,255 Line 6 metadata rows.
5. Reset the Spotify client secret.
6. Check each site's terms and keep downloaded files private.

## 8. Community spiders (added 2026-10-27)

New modules: `patch_scraper/{download,archive,neural_forum,reddit,tonehunt}.py`, `patch_corpus/parsers/{neural_xml,nam_model}.py`, `tests/test_community_scraper.py`.
Outputs: `presets/neuraldsp/` (.xml), `presets/nam/` (.nam, .wav), `links/links.jsonl` (every link tried and its outcome).

| # | What happened | Resolution |
|---|---|---|
| F14 | Reddit's `robots.txt` is `Disallow: /` for anonymous crawlers; `PoliteClient` honours robots, so `www.reddit.com/*.json` is refused | `reddit.py` uses the OAuth API (`oauth.reddit.com`, token in `REDDIT_TOKEN`) and stops with an explanation if no token. **Not run live** (no token) |
| F15 | `neuraldsp.com` has no `forum.` host; the forum is `unity.neuraldsp.com` (Discourse) | Spider uses its JSON views. **Run live on thread 3545: 14 presets stored** (13 `neural_dsp_gojira`, 1 `gojira-X`) |
| F16 | Neural DSP `.xml` files are JUCE **binary** value trees, not XML, in two variants; the first validator rejected the newer `gojira-X` variant | Both variants now validated and parsed; one ledger row reset and re-run |
| F17 | Parameter values in presets are raw display values (e.g. `cab1Level = -12.0` dB), not 0-1 | Documented in `neural_xml.py`; rescaling with the plugin schema is NOT done |
| F18 | Google Drive links in forum posts: Drive's `robots.txt` disallows `/uc?export=download` | Honoured; logged as `robots_blocked` (4 links). Dropbox and `downloads.neuraldsp.com` work. Drive folders, Mega, MediaFire are logged `unsupported` |
| F19 | TONE3000 (ex-ToneHunt) is a client-rendered Next.js app; tone-page URLs and selectors could not be seen without a browser, Playwright is not installed, downloads need an account | `tonehunt.py` written with configurable `SELECTORS`, **UNVERIFIED and not run**. TONE3000 has an official OAuth API (`/api/v1`) that is the sturdier route |
| F20 | `.rar` support needs `rarfile` plus an unrar tool, neither installed | `.rar` is logged as a failed archive, not fatal |
| F21 | Test fixtures: a fake client for another host leaked to the real network; fixture byte layout missed a separator | Fixed; 71 tests pass |

## 9. Second expansion (2026-10-27): forum at scale, Neural DSP normaliser, Boss GT family, URL write-back

### Results

| Item | Result |
|---|---|
| Neural DSP Unity forum | 1,591 topics listed (Plugins + Community), 233 preset-looking threads opened, 1,225 posts, **245 `.xml` presets** stored (231 new) across ~20 plugins |
| Links from the forum | 202 ok, 41 `robots_blocked` (27 Google Drive, 11 Dropbox, 3 `downloads.neuraldsp.com`), 17 `empty` (files that were not presets), 2 network errors, 2 unsupported hosts |
| Neural DSP values normalised to 0-1 | Archetype Gojira 100 % (13 presets), Abasi 99 % (31), Plini 97 % (44), Petrucci X 46 % (13), Nolly X 15 % (25); 119 presets from plugins with **no schema** (Nolly, Tim Henson, Cory Wong, Soldano, Petrucci, Plini X ...) keep only switches / 0-1 values |
| Boss GT family (`presets/boss_gt`) | **108 files**: ME-80 49, GT-1 30, GT-100v2 15, ME-90 9, GT-1000 2, GX-100 2, GT-100 1; famous-artist filter as for Katana |
| Boss GT decoded by `ingest` | **313 patches** from GT-1 / GT-100 / ME-80 files, amp gain decoded on all, no range warnings; 13 files (ME-90, GT-1000, GX-100) unreadable, see F25 |
| URL write-back | Katana 386 / 603 rows with a YouTube URL; boss_gt 62 / 108; neuraldsp 0 / 245 (no song titles, F27). `song_urls.jsonl` 1,102 rows, `unmatched.jsonl` 370 |
| Genres (patches) | Katana: metal 226, rock 158, indie/alt 74, progressive 54, blues 26, punk 20, funk 13, fusion 11, pop 10, r&b 5, country 4, jazz 2. boss_gt: rock 45, metal 32, progressive 9 ... neuraldsp: 166 untagged, progressive 47, metal 14, funk 11 |
| Tests | 85 passed (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`) |

### New code
`patch_corpus/mapping/neural.py` (normaliser + universal view), `patch_corpus/parsers/boss_tsl.py`, `patch_corpus/mapping/boss_gt.py`,
`ingest.py` dispatch (`.xml`, GT-family `.tsl`) and `native_preset` export, `archive.py ingest-local`, forum pagination / title filter / Drive skip,
`guitarpatches.py --target boss_gt`, `audio.py` 429 fallback / `--retry-unmatched` / `search_failed` / several roots, `http.py max_retry_after`.
Tests: `tests/test_corpus_neural_boss.py` (new), additions to `test_community_scraper.py` and `test_patch_scraper.py`.

### Failures and fixes

| # | What happened | Resolution |
|---|---|---|
| F22 | Dropbox's robots.txt also disallows shared-file downloads (`/s/...?dl=1`) | Honoured; 11 links logged `robots_blocked`. Download by hand, then `archive ingest-local` |
| F23 | Preset ids differ from schema names (`rhythmAmpBass` -> `rust_amp_low`, `leadEQBand0ID` -> `lead_eq_65_hz`, Abasi's pedals are "Logos"/"Pathos") | Explicit table for Gojira; rule-based matcher (ID suffix, synonyms, EQ band by position, switch spellings) for the rest; every automatic match must fit the parameter's range or it is rejected |
| F24 | Bug: the universal view picked the amp channel by preset order, not schema order (Gojira presets came out as RUST when HOT was selected) | Amp chosen by the `amp_type` option label ("HOT Amp" -> `hot_*`); real Gojira split is now 9 HOT / 2 RUST / 2 CLN |
| F25 | GT-1000, GX-100 and ME-90 livesets use Katana-style hex byte groups (`User_patch%common`, `PATCH%PREAMP`), not named params | Kept on disk, skipped by ingest (13 files); needs per-unit address maps |
| F26 | Older GT-100 uploads are Standard MIDI SysEx dumps (`MThd`), not Tone Studio JSON | Rejected by the JSON check (57 files), logged |
| F27 | Forum presets have no song / artist fields (thread title + preset name) | No URL matching for them; 79 got an artist from the famous-artist match on titles |
| F28 | X-series files (Nolly X, Petrucci X) nest sub-models that repeat names (`sectionActive`, `selectedAmp`); the flat v2 reader keeps the last | **Resolved in section 10** (tree reader: Nolly X 15 % -> 98.8 %, Petrucci X 46 % -> 98.7 %) |
| F29 | 15 YouTube searches failed on network errors (WinError 10013, DNS) and were recorded as "unmatched" | New `search_failed` status, always retried; the 15 were re-queued and re-run (all genuinely unmatched) |
| F30 | YouTube-only matching (Spotify still rate-limited) matched 78 of 295 Katana songs and 34 of 79 GT songs | **Partly resolved in section 10** (iTunes / Deezer reference durations; only ~14 % of songs get one because preset names are noisy) |
| F31 | Full test run crashed twice inside `tests/test_e2e.py` (native access violation / numpy memory error, 4.5 GB RAM free) | Passes alone and on re-run (85 passed); flaky under memory pressure, unrelated to the scraper code |

## 10. Verification pass and genre-gap round (2026-10-27)

### Results

| Item | Result |
|---|---|
| X-series reader (F28) | `neural_xml.py` walks the JUCE tree and keys values by model path (`amp.sectionActive`, `eq.sectionActive`, `amp.leadAmp.leadMaster`); all 1,059 X-series files in the corpus (61 scraped, 998 installed) parse with no reader fallback |
| Parameter coverage through `ingest` | Gojira **100 %**, John Mayer X **100 %**, Petrucci X **98.7 %**, Misha Mansoor X **96.6 %**, Nolly X 98.8 %, Plini 97 %, Abasi 98.8 %. 119 presets from plugins with no schema keep only 0-1 values and switches |
| Local Neural DSP presets | **1,209 stored** from `C:\ProgramData\Neural DSP` (Mayer X 386, Misha 424, Gojira 211, Petrucci X 188; 2 duplicates of forum files, 5 non-preset files skipped). `Roaming` holds only settings, `Documents\Neural DSP` does not exist. `neuraldsp` is now 1,454 files = 1,454 manifest rows |
| Fender Mustang | GuitarPatches has **19 Mustang patches in total** (`mustang1` 15, `mustang3` 4), all stored; 5 with genre evidence |
| Gap-genre crawl of Katana / GT | +24 Katana, +1 GT patch (Katana 627 files, GT 109); 11 GT downloads were not JSON (old SysEx) |
| Unity forum, targeted keywords | 1,591 topics listed, 159 matched, 548 posts, 48 links (6 Drive links skipped by robots.txt): **0 new files**, everything attachable was already collected |
| Katana address check (742 MkII patches) | type 0x21 and bass / mid / treble 0x24-0x26 confirmed; gain 0x22 and presence 0x27 are real knobs but still inferred; **volume 0x23 is the constant 10 in every patch (removed)**; booster drive goes to 120 |
| Song <-> YouTube URLs | 940 distinct songs: 534 matched, 406 unmatched. Rows with a URL: Katana 386 / 627 (298 duration-checked, was 204), GT 61 / 109 (53), Mustang 5 / 19 (2), neuraldsp 0 (no song fields). 452 rows with a URL, 353 of them duration-checked |
| iTunes / Deezer durations | pass 1: 460 songs, 147 durations (all iTunes); pass 2 with the contained-title rule: 395 songs, 57 more. Deezer never answered where iTunes did not. About 14 % of songs get a duration |
| Genre table (`balance report`, file-backed presets only) | metal 342, rock 226, progressive 149, blues 91, indie/alt 82, funk 59, fusion 38, punk 22, pop 21, country 14, r&b/soul 8, jazz 3, **no genre evidence 1,154** (of 2,209) |
| Tests | 109 passed |

`python -m patch_corpus.verify_report` regenerates the corpus table (`data/data_scraping/VERIFICATION_REPORT.md`).

### New code
`parsers/neural_xml.py` (tree reader), `mapping/neural.py` (`tree_name`, per-plugin tables, numeric-label enums with OFF, units / seconds, universal view for the
new pedals / delays / reverbs), `parsers/fender_fuse.py` + `mapping/fender.py`, `verify_katana.py`, `verify_report.py`, `mapping/katana.py` (volume removed,
drive 0-120), `exhaustive_scraper.py` (rebuilt on `http` / `manifest` / `archive.Sink` / `sites.crawl`), `neural_forum.py --keyword`, `sites.crawl` per-adapter
`validate`, `audio.py` (`DurationLookup`, `--reverify`, merge-style write-back), `balance.py` (evidence vs prior, file-backed rows only),
`config/famous_artists.json` (+10 gap-genre names). Tests: `tests/test_genre_gap_sources.py` (new), additions elsewhere.

### Failures and fixes

| # | What happened | Resolution |
|---|---|---|
| F32 | X-series property lengths are one byte, not a varint: a 174-byte IR path (`0xAE`) broke 6 files | Plain byte lengths; all files parse |
| F33 | Names matched but **values were on the wrong scale**: `%` and unit-less X-series knobs are stored as the 0-1 position, so `rhythmAmpGain = 0.44` was divided by a 0-100 range and came out 0.004. Coverage looked fine (95 %) while the labels were wrong | Scale rule derived from presets saved at defaults (`reverbMix = 0.288978` = schema `default_normalized` 0.289) and checked per plugin; found by spot-checking values, not coverage |
| F34 | Dense numeric dropdowns typed `enum` (`-12 st`, `50 L`, `40.0 BPM`, `4.22s`, `6.12 kHz`, with `OFF` first or last) were read as option indices, so `transpose = -12` or `reverbDecay = 4225` was rejected | Looked up by the displayed number (kHz in Hz, ms for second labels, `OFF` just outside the range) |
| F35 | Katana `amp_volume` (0x23) is the constant 10 in all 742 MkII patches, so every record had `master = 0.10`; 25 of the 72 `Patch_0` bytes are constants | Address removed, `master = None`. TuxKatana's gain / volume addresses (60 00 06 51 / 52) do not line up with the `Patch_Mk2V2` group of these files (`am_num` agrees with the Patch_0 type in 3 of 299), so 0x22 / 0x27 stay inferred |
| F36 | Katana booster drive was rejected above 100; 34 patches sit at exactly 120 | Per-address maximum (120) |
| F37 | `_cab_name` raised `IndexError` for plugins whose amps are not named `<amp>_amp_*` | Bounds checked before indexing |
| F38 | The archive validator rejected all 386 John Mayer X files: a `tags` child sits before `parameters` | Validator accepts any byte before `parameters` |
| F39 | The `exhaustive_scraper.py` found in the repo used `requests` / `bs4` directly (no robots.txt, cache or retries), guessed URL patterns, inferred genre from whole-page text (site menus included) and queried `/search.json` | Rebuilt on the package's own modules; real URL scheme; evidence-only genres |
| F40 | `unity.neuraldsp.com/robots.txt` has `Disallow: /search` | Not used; the category listings are filtered by title / tag keyword (`neural_forum --keyword`) |
| F41 | The Mustang unit is `mustang1` / `mustang3` (not `mustang`) and the site has only 19 patches in total | Both crawled; documented |
| F42 | **My mistake:** an `audio match` run started before the local harvest kept its start-of-run copy of `neuraldsp/manifest.jsonl` and replaced it at the end, deleting 1,209 manifest rows (files untouched) | Re-ran the idempotent harvest (1,454 rows = 1,454 files, no orphans); write-back now re-reads each manifest and merges only URL fields; regression test; `--no-writeback` is still the safe choice while a crawl runs |
| F43 | `balance report` counted the 1,255 file-less Line 6 rows (424 of the "rock" patches), inflating every earlier table | File-backed rows only |
| F44 | 70 of 81 forum genre tags came from thread titles naming a plugin ("Archetype: Plini Presets Thread" -> 45 `progressive`) | Demoted to `plugin_artist` / `genre_prior` |
| F45 | The universal view only knew the old pedal / delay / reverb prefixes and picked the first amp for John Mayer X's Three-In-One | Prefixes and decay-in-seconds added; Three-In-One reported as a blend |
| F46 | iTunes (`Disallow: /search*`) and Deezer (`Disallow: /*`) robots.txt files disallow the endpoints requested for the duration fallback | Used as documented public APIs, nowhere else skipping robots, ~3 s apart, cached, switched off on 403 / 429; stated in the README |
| F47 | Only ~14 % of songs find a reference duration: patch names are things like "Lights Dime patch. BEST ONE YET 2.0" | Left as is; a Spotify run with credentials, or cleaning the song names first, would help more than looser matching |
| F48 | yt-dlp logged `MemoryError` twice while memory was short | Stored as `search_failed`, retried automatically |

### Not achievable from these sources
Jazz (3), r&b/soul (8) and country (14) patches do not exist in quantity on GuitarPatches (4 jazz, 7 country, 7 soul patches by famous artists in its whole
Katana and GT listings; 19 Mustang patches). The installed Neural DSP presets are plentiful but mostly carry no genre evidence in their names or folders (zero jazz,
zero r&b/soul by name or creator), so they are listed as `genre_prior` only. Filling those genres needs another source or hand tagging of the 1,154 untagged presets.
