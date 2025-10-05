# DVC Pipeline Scripts

This directory contains batch processing scripts for the DVC pipeline.

## Scripts

### 1. `transcribe_batch.py`
Transcribes all audio files in `data/raw/` to text.

**Usage:**
```bash
python scripts/transcribe_batch.py --input data/raw --output data/transcribed --model base
```

**Arguments:**
- `--input`: Input directory with audio files (default: `data/raw`)
- `--output`: Output directory for transcriptions (default: `data/transcribed`)
- `--model`: Whisper model size - tiny, base, small, medium, large (default: `base`)

**Supported Audio Formats:**
- MP3, WAV, M4A, OGG, FLAC, MP4

---

### 2. `train_summarizer.py`
Fine-tunes the summarization model (optional).

**Usage:**
```bash
python scripts/train_summarizer.py --data data/transcribed --output models/summarizer --epochs 3
```

**Arguments:**
- `--data`: Input directory with training transcripts (default: `data/transcribed`)
- `--output`: Output directory for trained model (default: `models/summarizer`)
- `--base-model`: Base model to fine-tune (default: `google/pegasus-xsum`)
- `--epochs`: Number of training epochs (default: `3`)

**Note:** Training requires at least 100+ transcript samples. If fewer samples exist, the script will skip training and use the pre-trained model.

---

### 3. `analyze_batch.py`
Generates summaries, action items, and sentiment analysis.

**Usage:**
```bash
python scripts/analyze_batch.py --input data/transcribed --output data/processed
```

**Arguments:**
- `--input`: Input directory with transcripts (default: `data/transcribed`)
- `--output`: Output directory for analysis results (default: `data/processed`)
- `--summarizer`: Summarization model (default: `google/pegasus-xsum`)

**Outputs:**
- Individual analysis files: `{filename}_analysis.json`
- Batch summary: `batch_summary.json`

---

## Running with DVC

### Run entire pipeline:
```bash
dvc repro
```

### Run specific stage:
```bash
dvc repro transcribe  # Only transcription
dvc repro train       # Only training
dvc repro analyze     # Only analysis
```

### Check pipeline status:
```bash
dvc status
```

### Visualize pipeline:
```bash
dvc dag
```

---

## Pipeline Stages

1. **transcribe**: Converts audio files to text transcripts
2. **train**: Fine-tunes summarization model (optional)
3. **analyze**: Generates summaries and analysis

Each stage only runs when its dependencies change!
