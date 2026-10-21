# aidams-lab4-ChyGlasscock-LiewYuChen-Nguyen-SuchetWang

# From Text to Table: Measuring an LLM Extraction Pipeline

Course lab for **AIDAMS S5 – IDSI 51003 (Uncertainty Quantification)**.

The notebook evaluates an LLM that extracts climate data (Scope 1/2 emissions, crude steel production, net zero target year) from excerpts of steel-company sustainability reports. It treats the LLM as a measurement instrument: how accurate is it, how sure can we be about that accuracy, where does it fail, and what does it cost to fix?

## Files

| File | Content |
|---|---|
| `excerpts.json` | 20 report excerpts (E01–E20) |
| `gold_labels.json` | Hand-labeled correct values for each excerpt |
| `recorded_responses.json` | 120 recorded LLM responses: 1 greedy (temperature 0) + 5 samples (temperature 0.7) per excerpt |
| `aurelia_report_chunks.json` | A fictional report cut into 16 chunks, with 6 retrieval queries |

## How to run

1. Open the notebook in **Google Colab**.
2. Upload the four JSON files with the upload button in the **Files** panel. They go directly in the root folder (no `data/` folder needed); the notebook reads them by file name.
3. Run the cells from top to bottom.

The main analysis uses only the recorded responses (`BACKEND = "recorded"`), so **no API key is needed**. If you do not have an API key, skip the optional **A2** cell: with **Runtime → Run all**, execution would stop there with an error.

Dependencies (preinstalled in Colab): `numpy`, `pandas`, `matplotlib`, `scipy`, `pydantic`, `scikit-learn`.

### Optional parts

- **A2 – Run your own model** needs an LLM API key, stored in Colab **Secrets** (key icon in the left sidebar) with *Notebook access* turned on. This submission used a Google Gemini key via the `google-genai` SDK.
  - Gemini keys from Google AI Studio may now start with `AQ.` instead of `AIza`. They work with the native Gemini SDK, not with the OpenAI-compatible endpoint.
  - Some Gemini models have no free-tier quota (`429 RESOURCE_EXHAUSTED … limit: 0`). The notebook tries the available Flash models and uses the first one that responds.
  - With the current Anthropic Python SDK, `messages.create()` no longer accepts `temperature`, so the course's `ask_llm` Anthropic branch fails as written.
- **D2 – Dense embeddings** needs `pip install sentence-transformers` and downloads the public `all-MiniLM-L6-v2` model (no token needed; the Hugging Face `HF_TOKEN` warning can be ignored).
