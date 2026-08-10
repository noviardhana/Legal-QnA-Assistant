# Legal QnA Assistant — Fine-tuned LLM + GRPO + RAG

An end-to-end generative-AI pipeline for an Indonesian-language **Legal Team QnA assistant**, covering three stages: **supervised fine-tuning (QLoRA)**, **reinforcement learning with GRPO**, and a **Retrieval-Augmented Generation (RAG)** system over four Indonesian labor-law statutes (UU), wrapped in an interactive **Streamlit chatbot dashboard**.

demo: [`Legal-QnA-Assistant`](https://legal-qna-assistant.streamlit.app/)

---

## 1. Overview

| Stage | What it does | Notebook |
|---|---|---|
| Fine-tuning | QLoRA (4-bit) supervised fine-tuning of a Llama-3.2 Instruct model on an Indonesian instruction dataset | `Fine-tuning_submission_PGABL_*.ipynb` |
| GRPO (optional/advanced) | Reinforcement learning on top of the fine-tuned model, teaching it to reason inside `<think>...</think>` tags before answering | `GRPO_submission_PGABL_*.ipynb` |
| RAG | Retrieval pipeline over 4 Indonesian labor-law PDFs, combining ensemble retrieval, HyDE, and cross-encoder reranking with a web-search fallback | `RAG_submission_PGABL_*.ipynb` |
| Dashboard | Streamlit chatbot UI with persistent chat history, deployable to Streamlit Cloud | `app.py` (this repo) |

**Final model chain:** base model → SFT (QLoRA) → GRPO → merged 16-bit model pushed to Hugging Face Hub → used by the RAG pipeline and the Streamlit dashboard for inference.

---

## 2. Model & Dataset

- **Base model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (switched from Llama-3-8B to fit comfortably on a free-tier Tesla T4 GPU; officially supported by Unsloth, Meta-provided architecture).
- **Fine-tuning dataset:** [`Ichsan2895/alpaca-gpt4-indonesian`](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian) — an Indonesian instruction dataset. Note: despite the "Alpaca" name, the actual columns in this dataset are `input` (instruction) and `output` (answer), not the standard `instruction`/`input`/`output` triple.
- **Chat template:** Llama-3 chat format via `tokenizer.apply_chat_template`.
- **RAG knowledge base:** 4 PDF documents containing Indonesian labor-related statutes (UU), provided by the course.

---

## 3. Fine-tuning (QLoRA)

- **Quantization:** 4-bit with double quantization (`load_in_4bit=True`, Unsloth default).
- **LoRA adapters:** attached to both Multi-Head Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and Feed-Forward Network (`gate_proj`, `up_proj`, `down_proj`) modules, rank `r=8`.
- **Trainer:** `SFTTrainer` from `trl`, configured via `SFTConfig` (not `TrainingArguments` — see [Lessons Learned](#6-lessons-learned--troubleshooting-notes)), with `packing=True` for throughput.
- **Hyperparameter experiments:** two configurations compared on train/validation loss curves (learning rate, batch size, warmup, weight decay) to pick a setup that minimizes eval loss without overfitting.
- **Final training run:** ≥800 steps using the best configuration found.
- **Model export:** merged to 16-bit and pushed to the Hugging Face Hub via `model.push_to_hub_merged(..., save_method="merged_16bit")`.

---

## 4. GRPO (Advanced / Optional)

Reinforcement learning stage on top of the fine-tuned model, using `GRPOTrainer` from `trl` + Unsloth. The model is trained to expose its reasoning process in a `<think>...</think>` block before producing a final answer, guided by **four custom reward functions**:

| Reward function | Purpose |
|---|---|
| `format_reward_func` | Shaped reward (capped at +1.0) for correctly opening/closing the `<think>` tag in the right position; penalty (−0.5) for hallucinated duplicate tags. |
| `reasoning_length_reward` | Proportional reward based on the length of the reasoning content inside `<think>`, tolerant of completions truncated by the token limit. |
| `correctness_reward` | +1.0 if the final answer contains the ground-truth output or scores highly on ROUGE-L against it. |
| `language_reward_func` | −0.5 penalty if the final answer drifts into English; +1.0 for pure Indonesian output. |

**Memory mitigation on a free-tier GPU:** `num_generations` and `max_completion_length` are kept modest (e.g. 2–4 and 256–320 respectively) to avoid CUDA OOM on a Tesla T4.

**Required test case** (verified manually after training):

> **Prompt:** "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?"
> **Expected behavior:** model outputs a `<think>...</think>` reasoning block referencing the relevant regulation (PP No. 35/2021), followed by a final answer in Indonesian.

The GRPO-tuned model is merged and pushed to the Hub the same way as the SFT model.

---

## 5. RAG Pipeline

### Basic
- **PDF loading & chunking:** `PyPDFLoader` + `RecursiveCharacterTextSplitter` with explicit chunk size / overlap.
- **Embeddings & vector store:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (open-source, multilingual) embeddings stored in a local **FAISS** index.
- **Generation:** the fine-tuned (or GRPO) model, prompted with an explicit `{context}` / `{question}` template.
- **Interface:** simple Gradio `gr.Interface` for quick testing inside the notebook.

### Skilled
- **Metadata enrichment:** each chunk is tagged with `source` (filename) and `law_number` / `law_year` (statute number/year), extracted via regex from the document text.
- **Citations:** every answer lists the source statute(s) used.
- **Ensemble Retriever:** combines a semantic retriever (FAISS, weight 0.6) with a keyword retriever (BM25, weight 0.4), retrieving ≥5 documents.
- **Parent-Child chunking:** small child chunks (300/50 chars) are used for vector search; retrieval maps back to larger parent chunks (2000/200 chars) that are passed to the LLM as full context.

### Advanced
- **HyDE (Hypothetical Document Embeddings):** the LLM first generates ≥2 hypothetical ("hallucinated") answers to the question; these are concatenated with the original query before retrieval, enriching the semantic signal.
- **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks retrieved parent chunks and keeps only the top-3 most relevant.
- **Conditional web fallback:** the Top-1 reranker relevance score is extracted; if it falls below a calibrated threshold, the system discards the local documents and falls back to a **DuckDuckGo Search** query instead.

---

## 6. Lessons Learned / Troubleshooting Notes

These issues were encountered and resolved during development — kept here for reference in case the environment drifts again:

- **Dataset column mismatch:** the actual dataset columns are `input`/`output`, not `instruction`/`input`/`output`. Always print `dataset.column_names` before mapping.
- **`Trainer.__init__() got unexpected keyword 'tokenizer'`:** newer `transformers` versions removed the `tokenizer=` argument from `Trainer`; use `processing_class=tokenizer` instead.
- **`trl` version drift:** pinning `trl<0.9.0` causes incompatibility with newer `transformers`. Fix: `pip uninstall -y trl && pip install -U trl`.
- **`SFTConfig.__init__() got unexpected keyword 'push_to_hub_token'`:** caused by mixing `TrainingArguments` with Unsloth's `SFTTrainer` wrapper. Fix: build training config via `SFTConfig` directly (it subsumes `TrainingArguments` plus `dataset_text_field`, `max_seq_length`, `dataset_num_proc`, `packing`).
- **`GRPOConfig` has no `max_prompt_length`:** in this environment's `trl` version, only `max_completion_length` is exposed; prompt length is implicitly bounded by `max_seq_length` set at model load time. Use `inspect.signature(GRPOConfig.__init__)` to check available parameters if this changes again.
- **Reward function double-counting:** an early version of `format_reward_func` summed the "perfect format" bonus (+1.0) on top of the open/close partial rewards, exceeding the spec's "+1.0 maximum". Fixed by capping the score instead of accumulating it.
- **Slow training on Tesla T4:** free-tier T4 GPUs are the main bottleneck for an 8B model. Mitigations applied: switched to a 3B base model, reduced LoRA rank (`r=16→8`), disabled gradient checkpointing, reduced `max_seq_length` (2048→1024), enabled `packing=True`, increased batch size per device, and reduced eval/logging frequency.
- **`vLLM` import error with `fast_inference=True`:** requires a separate install and is prone to OOM on 15GB VRAM; disabled (`fast_inference=False`) for stability.
- **`langchain.text_splitter` / `langchain.retrievers` moved:** in current `langchain` versions, `RecursiveCharacterTextSplitter` lives in `langchain-text-splitters`, and `EnsembleRetriever` may need `langchain` (core) installed separately from `langchain-community`.
- **`duckduckgo_search` renamed to `ddgs`:** update both the `pip install` and the import statement.
- **Warnings that are safe to ignore:** `Both max_new_tokens and max_length seem to have been set` (harmless generation warning), `jupyter_client` `datetime.utcnow()` deprecation warning (unrelated to this project's code).
- **HF Inference API endpoint deprecated:** `api-inference.huggingface.co` now returns HTTP 410 (no longer supported). All inference calls in `app.py` must target `https://router.huggingface.co/hf-inference/models/{repo}` instead.
- **Free serverless inference may not serve custom fine-tuned models:** the router endpoint only guarantees "warm" models; a personal fine-tune like `llama3-legal-id-grpo` may 404 there. If so, deploy it via a dedicated [Inference Endpoint](https://huggingface.co/inference-endpoints) and point `HF_MODEL_REPO`/the API URL at that instead.
- **Error-handling order bug in `call_hf_inference`:** calling `resp.raise_for_status()` before inspecting the JSON body meant HF's structured "model is loading" error (returned with a 503) was swallowed by a generic exception instead of showing the intended friendly message. Fixed by parsing the JSON body first and only raising on HTTP errors if no structured `error` field is present.
- **`st.secrets` can raise before `.get()` runs:** if `.streamlit/secrets.toml` doesn't exist at all (e.g. local dev), accessing `st.secrets` can throw before a `.get()` call is reached. Wrapped in a `_get_config()` helper with try/except, falling back to environment variables.
- **Accidental secret commit:** a Hugging Face token was hardcoded in two notebook cells (`login(token="...")`), triggering GitHub's push protection on the first push. Fixed by using `getpass()` (interactive) or `os.environ["HF_API_TOKEN"]` loaded via `python-dotenv` instead of hardcoding, then amending the commit before pushing. The exposed token was revoked and regenerated on Hugging Face.

---

## 7. Repository Structure

```
.
├── app.py                      # Streamlit chatbot dashboard (this repo)
├── build_index.py              # One-time script: auto-downloads legal_docs/ from Google Drive
│                                #   (via gdown) if missing, then builds FAISS index + parent/child chunks
├── requirements.txt            # Lightweight deps for the Streamlit dashboard (CPU only)
├── requirements-training.txt   # Full deps for the training/RAG notebooks (GPU)
├── .streamlit/
│   └── secrets.toml.example    # Template for HF_MODEL_REPO / HF_API_TOKEN
├── index/                      # Generated by build_index.py — commit this so Streamlit Cloud can load it
│   ├── faiss_index/
│   ├── child_docs.pkl
│   └── parent_docs.pkl
└── notebooks/
    ├── Fine-tuning_submission_PGABL_*.ipynb
    ├── GRPO_submission_PGABL_*.ipynb
    └── RAG_submission_PGABL_*.ipynb
```

---

## 8. Running the Notebooks

1. Open each notebook in Google Colab (GPU runtime recommended).
2. Run `Fine-tuning_submission_*.ipynb` first — it pushes the fine-tuned model to your Hugging Face account.
3. (Optional) run `GRPO_submission_*.ipynb`, which loads the fine-tuned model and continues training with GRPO, then pushes the GRPO model.
4. Run `RAG_submission_*.ipynb` to build and test the retrieval pipeline against the 4 statute PDFs and the pushed model.
5. Fill in `HF_USERNAME` and a Hugging Face **write** token (`huggingface.co/settings/tokens`) at the top of each notebook before running the push cells. Prefer `getpass()` or an environment variable over hardcoding the token in a cell — hardcoded tokens get blocked by GitHub push protection and must be revoked if ever committed.

---

## 9. Deploying the Streamlit Dashboard

The dashboard calls your Hugging Face model through the **Inference API** rather than loading it locally — this keeps the app lightweight enough to run on Streamlit Cloud's free tier (which has no GPU).

### Step-by-step

1. **Build the retrieval index locally:**
   ```bash
   pip install -r requirements-training.txt
   python build_index.py
   ```
   `build_index.py` first checks for `legal_docs/` locally; if the 4 PDFs aren't present, it automatically downloads them from Google Drive using `gdown`. This creates the `index/` folder — commit it to your GitHub repo.

2. **Push this repo to GitHub**, including `app.py`, `requirements.txt`, and `index/` (do **not** commit `.streamlit/secrets.toml` or the raw `legal_docs/` PDFs — see `.gitignore`).

3. **Deploy on [Streamlit Cloud](https://share.streamlit.io):**
   - Connect your GitHub repo.
   - Set the main file to `app.py`.
   - Under **Settings → Secrets**, paste:
     ```toml
     HF_MODEL_REPO = "username-hf-anda/llama3-legal-id-grpo"
     HF_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
     ```
   - Deploy.

4. **Note on model availability:** the Hugging Face Inference API only serves models that are either popular enough to be "warm" or explicitly deployed via an Inference Endpoint. If your fine-tuned model returns a loading or 404 error, you may need to use a [dedicated Inference Endpoint](https://huggingface.co/inference-endpoints) instead of the free serverless API. Also make sure your code targets the current router endpoint (`router.huggingface.co/hf-inference`), not the deprecated `api-inference.huggingface.co`.

### Local testing

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 10. Dashboard Features

- Chat-style interface (`st.chat_message`) with persistent conversation history for the session.
- Sidebar: clear history, download conversation as `.txt`, and quick example questions.
- RAG pipeline: ensemble retrieval (FAISS + BM25) → HyDE query expansion → cross-encoder reranking → relevance-threshold fallback to DuckDuckGo Search → generation via Hugging Face Inference API → answer with citations.
- Cached resources (`st.cache_resource`) so the embedding model, FAISS index, and reranker are loaded once per server session, not per question.

---

## 11. Model Links

See [`link_huggingface.txt`](./link_huggingface.txt) for the deployed model URLs.