# Dataset spec

A **dataset** is a self-contained corpus: a fixed set of audio clips, plus every annotation
of those clips from every source (a forced aligner, a human annotator, ...), in one place.
There is no more notion of "dataset A" and "dataset B" for two takes on the same clips —
two annotation passes over the same audio are two **labels** on the same manifest row, not
two datasets.

A **datasets store** is a folder or an S3 bucket that holds one or more datasets side by
side. Its top-level entries (subfolders locally, key prefixes in a bucket) are the datasets
it holds, and the entry's name **is** the dataset's name. Picking a dataset means picking
one of those top-level names — see `--datasets-folder` / `--datasets-bucket` / `--dataset`
in `hebrew_training/align_tag_server.py`.

## Layout

```
<dataset-name>/
  metadata.json
  manifest.jsonl
  audio/
    <id-1>.wav
    <id-2>.wav
    ...
```

Locally this is `<datasets-folder>/<dataset-name>/...`. In a bucket it's the same tree under
the key prefix `<dataset-name>/`.

## `metadata.json`

```json
{"name": "plenum"}
```

| key  | type | description                                                        |
| ---- | ---- | ------------------------------------------------------------------ |
| name | str  | The dataset's name. Should match the folder/prefix it lives under. |

## `manifest.jsonl`

One JSON object per line, one line per clip (an "entry"). `audio`, `text`, and `duration`
describe the clip itself and are **shared by every label** — an entry is one recording,
annotated one or more times, not one recording per annotator. If two sources ever disagree
about what the underlying audio or transcript actually is, they are describing two
different clips, and belong in two different manifest rows (each with its own `id` and
audio file), not two labels on one row.

| key      | type        | description                                                                                                                                                              |
| -------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| id       | str         | Unique id of the entry within the dataset. Generated once at creation time, copy-paste friendly (e.g. a short hex string), and never reused. It's the only stable identifier for an entry — nothing else (not the audio filename, not any metadata field) is guaranteed unique. |
| metadata | json        | Arbitrary, caller-defined metadata for the entry (e.g. `{"recording": "111484"}`, the id of a longer source recording the clip was excerpted from). The server passes this through; it does not read specific keys out of it.                                                    |
| audio    | str         | Filename of the entry's audio, relative to the dataset folder (so usually `"audio/<id>.wav"`). The audio is expected to already be exactly the clip — no lead-in/lead-out context baked in; a tool that wants annotation context adds it at serving time, not in the dataset.   |
| text     | str         | The text spoken in the audio.                                                                                                                                                                                                                                                     |
| duration | number      | The entry's duration in seconds, start to end, rounded to 3 decimal places. Every label's `words` must fit within `[0, duration]`.                                                                                                                                               |
| labels   | array[json] | One entry per annotation source. At least one label is required.                                                                                                                                                                                                                  |

### label object (each entry of `labels`)

| key    | type        | description                                                                                                          |
| ------ | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| source | str         | Name of the label's source — an aligner's name (`"hebrew"`, `"mms"`) or a human annotator's name. Unique within one entry's `labels` array. |
| words  | array[json] | The annotated words, ideally covering the entirety of `text`.                                                          |

### word object (each entry of a label's `words`)

| key   | type   | description                                                                          |
| ----- | ------ | ------------------------------------------------------------------------------------- |
| word  | str    | The word's text, no leading/trailing whitespace.                                      |
| start | number | Word start within the entry's audio, seconds, rounded to 4 decimal places. `0 <= start <= end`. |
| end   | number | Word end within the entry's audio, seconds, rounded to 4 decimal places. `end <= duration`.     |

## Example row

```json
{
  "id": "6a1925c74160",
  "metadata": {"recording": "111484"},
  "audio": "audio/6a1925c74160.wav",
  "text": "אני שב ומודה לכל מי שהיה מעורב ושותף,",
  "duration": 3.08,
  "labels": [
    {"source": "hebrew", "words": [{"word": "אני", "start": 0.1812, "end": 0.2617}, "..."]},
    {"source": "mms",    "words": [{"word": "אני", "start": 0.0,    "end": 0.0604},  "..."]}
  ]
}
```

## Invariants worth stating explicitly

- **One id, one truth.** `id` is generated once and is the only thing anything should key
  off of (saved annotator marks, claim bookkeeping, exports). Nothing else about an entry is
  guaranteed stable.
- **The clip is one audio, one transcript, one duration — for every label.** `audio`/`text`/
  `duration` live on the row, not the label, specifically so two label sources can never
  silently describe two different underlying clips.
- **No baked-in context.** A dataset's audio is exactly `[0, duration]` — trimmed, not padded.
  Any "hear a bit before/after the word" feature belongs to the tool serving the clip, not to
  the dataset on disk.
- **A dataset is one folder/prefix, read as a whole.** `<store>/<dataset-name>/...` is the
  entire unit; nothing about a dataset spans multiple top-level folders/prefixes.
- **At least one label, always.** A manifest row with an empty `labels` array is invalid —
  every clip in the dataset needs to have been annotated by *someone*.

## Converting a legacy per-source manifest into this shape

`hebrew_training/convert_legacy_dataset.py` does this for the old "one jsonl file per
source, clips pre-padded with context" layout (what `data/gold_set` used to be). It's a
reasonable starting point for converting any other legacy corpus into this spec.

## Possible future extensions (not implemented yet)

- Per-word confidence scores.
- Non-audio dataset kinds (video, image) reusing the same manifest + labels shape.
- Multiple audio renditions per entry (e.g. different sample rates or codecs) — would need
  a new repeatable field; not yet designed, don't assume `audio` can become a list without
  updating every reader.
