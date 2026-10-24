# Vocalization datasets

Training data for the voc2vec vocalization classifier (`audio_text.py`). Audio files here are not
tracked or shipped; add whatever dataset you pick. Nothing is hard-coded to one dataset.

## Layout

Either a folder per class:

```
data/vocalizations/<dataset>/<label>/*.wav      (.wav .flac .mp3 .ogg)
```

or a manifest CSV (columns `path,label`; relative paths are resolved against the CSV's folder):

```
path,label
clips/0001.wav,yes_like
clips/0002.wav,calling
```

## Adding a dataset

Drop its folder (or manifest) under `data/vocalizations/` and pass it to `train`. Several can be
combined; labels with the same name are pooled. Embeddings are cached in `models/cache/`, so
re-training after adding a dataset only embeds the new files.

First candidate: CNVVE (6 non-verbal voice classes, healthy and dysarthric speakers, built for
assistive HCI). It is not downloaded here yet.

## Train

```
python audio_text.py train --data data/vocalizations/<dataset> [more dirs or csvs] --out models/voc_clf.joblib
python audio_text.py classify some.wav
```

`train` prints stratified k-fold UAR (k = min(10, smallest class size)). Clips from the same speaker
land in different folds, so the number is optimistic for unseen speakers.
