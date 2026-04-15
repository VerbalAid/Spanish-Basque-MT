# Spanish → Informal Basque MT

Translation of informal Spanish Instagram DMs into natural, informal Basque — preserving Gen-Z register, code-switching, phonetic stylisation, and dialectal forms (Bizkaian/Gipuzkoan).

**Best result: chrF++ 38.29** — beats the Latxa 70B baseline of 36.1.

---

## Task

| | |
|---|---|
| **Language pair** | Spanish → Basque |
| **Domain** | Instagram DMs — informal, youth register |
| **Train / Dev** | ~130 / 30 sentence pairs (ISMD corpus) |
| **Metric** | chrF++ (word_order=2) |
| **Baseline** | Latxa 70B → 36.1 chrF++ |

---

## Evaluation Metric — chrF++

chrF++ (Character n-gram F-score with word-order penalty) measures overlap between hypothesis and reference at the **character level**, with an added word-order component.

**How it works:**
- Computes precision and recall over character n-grams (default: 1–6)
- F-score is the harmonic mean of the two
- The `++` (word_order=2) adds a word bigram component on top, penalising reordering

**Why chrF++ for this task:**
- Basque is morphologically rich — a single stem can generate dozens of surface forms. Character-level matching handles this gracefully where BLEU would penalise valid inflections
- Works well on short, informal sentences where BLEU's brevity penalty and sparse n-gram matching produce unstable scores
- Standard metric for low-resource and morphologically complex MT evaluation

**Limitations:**
- Can reward outputs that share surface characters with the reference even when semantically wrong
- The word-order penalty is relatively weak — significant reordering is not heavily punished
- A single reference translation means score is sensitive to the specific phrasing chosen by the annotator, especially for informal text with high stylistic variation

---

## Approaches

### 1 — NLLB-200 1.3B (zero-shot)
- Facebook's massively multilingual seq2seq model covering 200 languages
- Basque is a supported language (`eus_Latn`); forced as the BOS token at decoding time to steer output
- No fine-tuning — pure zero-shot inference
- Trained on formal parallel web text (CCAligned, WikiMatrix, etc.) so informal register is largely unseen

### 2 — Helsinki-NLP opus-mt-es-eu (zero-shot)
- MarianMT model trained specifically on the Spanish→Basque OPUS corpus
- Faster and lighter than NLLB-200; stronger baseline for this language pair due to dedicated training
- Still formal-register — the OPUS data is mostly subtitles and web crawl, not DM-style text
- Used as the primary fine-tuning base for approaches 3–5

### 3 — Fine-tune on in-domain data
- Takes the pretrained Helsinki model and continues training on the shared task train set (~130 pairs)
- Custom `DMDataset` class handles tokenisation and label alignment for seq2seq training
- Uses HuggingFace `Seq2SeqTrainer` with `predict_with_generate=True` so eval uses beam decoding, not teacher forcing
- Risk of overfitting given the tiny dataset; mitigated somewhat by starting from a strong pretrained checkpoint

### 4 — Back-translation data augmentation
- Uses the reverse `Helsinki-NLP/opus-mt-eu-es` model to translate Basque references back into Spanish
- These synthetic Spanish sentences are paired with the original Basque references to create new training pairs
- Roughly doubles the training data without needing external resources
- Quality is noisy — the back-translated Spanish won't match the informal style of the originals — but adds lexical diversity

### 5 — External corpus (OPUS OpenSubtitles v2018)
- Downloads the full OPUS OpenSubtitles es-eu TMX file (~remote gzip stream), parses the XML, and extracts parallel pairs
- Caps at 10,000 sentence pairs to stay within Colab memory limits
- Combined with the in-domain data before fine-tuning
- OpenSubtitles is closer to informal spoken language than web crawl, making it a reasonable domain bridge — but still not DM register

### 6 — TowerInstruct-7B (FAILED)
- Unbabel's translation-specialist 7B LLM, loaded in 4-bit quantisation via bitsandbytes
- Designed specifically for translation with instruction-following capability
- Failed due to memory/compatibility issues on Colab T4 — 4-bit quant of a 7B model with the Colab environment's bitsandbytes version was unstable
- Translation function and evaluation code are intact; would be worth retrying with a newer runtime

### 7 — LLM few-shot prompting
- Skips fine-tuning entirely — uses large pretrained LLMs via API with a carefully engineered prompt
- System prompt defines a **bilingual Basque/Spanish teenager persona** with explicit style rules (no formal Batua, match abbreviations, preserve elongations)
- 6 diverse few-shot examples sampled from the training set show the model exactly what output style is expected
- Temperature=0.0 for deterministic output — important for maximising chrF++ against a fixed reference
- Tested across four models in increasing capability; GPT-4o was the clear winner

**Why prompting outperforms fine-tuning here:**
- The training set is too small (~130 pairs) for fine-tuning to reliably teach informal register from scratch
- GPT-4o has seen vast amounts of informal multilingual text in pretraining; few-shot examples unlock that knowledge without weight updates

---

## Results

| Model | chrF++ |
|-------|--------|
| Latxa 70B (baseline) | 36.1 |
| Llama-3.3-70B (Groq) | — |
| GPT-4o-mini | — |
| **GPT-4o** | **38.29** |

---

## Setup

Runs in Google Colab (T4). Dependencies installed inline.

**API keys** — set as environment variables:

```python
import os
os.environ["GROQ_API_KEY"] = "..."
os.environ["OPENAI_API_KEY"] = "..."
```

In Colab, use Secrets (🔑 sidebar):
```python
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

**Data** — place files at `/content/drive/MyDrive/Shared_Task/`:
```
train.txt        # ~130 Spanish–Basque pairs
devel.txt        # 30 evaluation pairs
test.txt         # unlabelled test set
extraDataEU.txt  # monolingual informal Basque
```

---

## Author

Darragh Kerins · [GitHub](https://github.com/VerbalAid)
