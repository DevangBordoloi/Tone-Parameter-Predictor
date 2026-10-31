"""Folder layout of the scraped corpus (everything lives under ``data/data_scraping``)::

    data/data_scraping/
        presets/
            katana/   *.tsl   manifest.jsonl      GuitarPatches (Boss Katana)
            boss_gt/  *.tsl   manifest.jsonl      GuitarPatches (Boss GT-1000 / GT-100 / GT-1 / ME-80 / ME-90 / GX-100): unseen-hardware set
            fender_mustang/ *.fuse manifest.jsonl GuitarPatches (Fender Mustang I-V, plain-XML FUSE presets)
            neuraldsp/ *.xml  manifest.jsonl      community Neural DSP presets (forum, Reddit, ...) and the factory / artist presets of the installed plugins
            nam/      *.nam *.wav manifest.jsonl  Neural Amp Modeler captures and impulse responses
        links/links.jsonl     every download link tried, with its outcome (ok / dead / blocked ...)
        urls/
            song_urls.jsonl   one row per song: YouTube / Spotify URLs, match score
            unmatched.jsonl   songs with no confident YouTube match
            audio/            only when audio is downloaded with --download-audio
"""

from __future__ import annotations

from common.paths import DATA_DIR

SCRAPE_ROOT = DATA_DIR / "data_scraping"
PRESETS = SCRAPE_ROOT / "presets"
KATANA = PRESETS / "katana"
BOSS_GT = PRESETS / "boss_gt"
FENDER = PRESETS / "fender_mustang"
URLS = SCRAPE_ROOT / "urls"
NEURALDSP = PRESETS / "neuraldsp"
NAM = PRESETS / "nam"
LINKS = SCRAPE_ROOT / "links" / "links.jsonl"
ARCHIVE_CACHE = URLS / "_cache" / "archives"          # archives are unpacked here, valid files are moved out, the rest is deleted
