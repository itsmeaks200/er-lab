# Research synthesis — what works for this entity-resolution problem

This covers the problem (3 sources, S1 = clean reference, one-to-many, precision-weighted **macro F0.5 per S1**, singletons score 1 when left empty,
12.5M train / 11.7M test records, an unseen test country) and what the literature and winning Kaggle solutions say.
Every item ends in an experiment ID from `erlab/registry.py`.

## 1. Closest analogues and what won

| Source | Setting | What won | Transfer to us |
|---|---|---|---|
| **Foursquare Location Matching** (Kaggle 2022; POI name + address + coordinates) | multi-record matching, ~1.1M POIs | Multi-stage: text/geo kNN candidates → LightGBM filter (top ~20–40 per id) → rich features (Levenshtein, Jaro-Winkler, TF-IDF, embedding SVD) → 2nd LightGBM → **transformer cross-encoders** → **GNN post-processing on the candidate graph** (1st-place-level: 0.907 → 0.946). Others: ArcFace metric-learning bi-encoder for candidates; 3 progressive LightGBM stages. | Our pipeline is the same shape. Adds: cross-encoder on the uncertain band (N02) and graph/cross-source post-processing (M11 stage-2; GNN is a next step). |
| **Shopee Price Match** (Kaggle 2021; product titles + images) | cluster matching | ArcFace embeddings + TF-IDF kNN **union**, **per-query threshold with a "min-2" rule** (at least N neighbours), neighbour-embedding smoothing | Union of dense + sparse retrieval (G/S + C/D/F). The "per-query set decision" matches our expected-F0.5 set selection. |
| **Instacart Market Basket** (Kaggle 2017; macro F1 per order with a "None" option) | **per-group F-score with an empty-set option — the same structure as our metric** | Winners maximised the **expected F1 per order** (Jansche / Faron DP) including the probability of "None"; about +0.2 LB points vs a global threshold in public write-ups | `evaluate.expf_select`: Bayes-optimal expected-F0.5 set per S1 from calibrated probabilities, including the empty set and a term for "true match missed by blocking". Runs in every model experiment (`hold_expf`). |
| **Quora Question Pairs** | pairwise duplicate detection | GBDT on handcrafted similarity + "magic" graph features (question frequency, k-core, common neighbours) | Commonness features (`nf_*`, `af_*`) and candidate-graph degree features are the analogue of "magic" features. |

## 2. Blocking (candidate generation)
* **Sparkly (VLDB 2023):** plain **TF-IDF/BM25 top-k** beat 8 state-of-the-art blockers, including DeepBlocker. So the TF-IDF passes C/D/F are the backbone. B02 tests BM25 and a character pass.
* **SC-Block (2023):** a **supervised contrastive** bi-encoder with kNN reaches 99.5% pair completeness with far smaller candidate sets, and 1.5–8× faster pipelines. Our hash-encoder (pass G) is a lightweight SC-Block trained from scratch. B05 fine-tunes e5-small the same way.
* **Pre-trained embeddings for ER (VLDB 2023/2024):** SentenceBERT-family models (S-GTR-T5 best) give the highest blocking recall of 12 language models. B04 tests pretrained e5-small (multilingual, MIT licence, useful for Indic scripts).
* **Recall is the ceiling.** Choose K per pass from recall@K curves and union passes greedily. Report the oracle F0.5 given the candidates (every blocking and model run does).

## 3. Matching
* **GBDT on similarity features** is still the workhorse at this scale (Foursquare, Quora). Knowledge-base laws: shallow, regularised trees on synthetic noise (M02), diversity through different families (M05–M08), rank vs probability blending (M09), at most 5 hyperparameters tuned (M12).
* **Cross-encoders beat bi-encoders** for pairwise decisions ("Beyond Scale and Generation", 2026, 1,215 fine-tuning runs). **Ditto** (VLDB 2021) serialises records (`name: … | address: …`) and fine-tunes a small language model, up to +29% F1 over earlier methods. N02 trains a multilingual MiniLM cross-encoder on hard negatives and applies it **only to the uncertain band** (cost control), stacked into stage 2.
* **Contrastive pre-training (R-SupCon, Sudowoodo)** gives +3–4% F1 on product matching. Covered by B05/N01 (fine-tuned embeddings used as features).
* **Fine-tuned small LLMs** (Llama-3.1-8B, Qwen) help mostly on hard pairs and under distribution shift, which matters for France. They are expensive, so this is a candidate for a *final* reranker on the few thousand most uncertain test pairs. Planned for a later wave (Apache-licensed Qwen2.5-7B, which meets the ≤8B / Apache rule).
* **Fellegi–Sunter with term-frequency adjustment (Splink):** agreement on rare values carries more evidence. Covered by IDF-weighted overlap, rare-token counts and commonness features.

## 4. Post-processing and decisions
* **Unique-assignment law** (each S2/S3 record belongs to at most one S1, verified on 7.6M links): the `exclusive` option in the rule search.
* **Transitivity is dangerous.** Practitioners report that one false link merges unrelated clusters ("Entity Resolution in Practice", 2026). So we never propagate transitively. We use *cross-source confirmation* as a feature instead (M11), which lets the model decide.
* **GNN over the candidate graph** (Foursquare 1st) is the principled version of cross-source confirmation. A next-wave candidate if M11 shows a large gain.
* **Metric-aligned decisions.** Staged thresholds (global → exclusive → per-source → anchor/expansion → margin) versus the **expected-F0.5 set selection** above. Chosen by tune-fold F0.5, reported on hold.

## 5. Language and text handling
* Indic scripts: rule-based transliteration from Unicode character names plus a consonant skeleton. The research (Aksharantar, Dakshina) has learned models, but those count as external data or models. Pretrained multilingual encoders (e5 / MiniLM) cover scripts natively, a second route (B04/N01).
* Learned token-substitution maps from training pairs (e.g. state names to codes) are data-driven normalisation with no external lookup.

## 6. Guard-rails (from the knowledge base and the competition rules)
* Entity-grouped folds (fit/tune/hold by S1 hash). Never split by pair. Early stopping and thresholds use tune only; hold is only reported.
* No external lookup or geocoding. Pretrained models must be MIT/Apache and ≤8B parameters (e5-small MIT, Multilingual-MiniLM MIT).
* Ablation before adoption (M10). Blend only if prediction correlation is < 0.98 (M09 prints the correlation).

## 7. Priority order (expected value ÷ cost)
1. B01 → M01 → D01 establish the baseline and ceiling (oracle F0.5).
2. The decision rule (expf vs staged) and M11 stage-2 should give the cheapest large gains.
3. M02/M03/M12 tune the scorer. M05–M09 test diversity.
4. B04/B05/N01 test multilingual embeddings for recall on Indic, domain and trade names.
5. N02 cross-encoder on the uncertain band. Then an LLM judge or GNN if the gap to the oracle is still large.

### Sources
- Foursquare: https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ ; https://github.com/TheoViel/kaggle_foursquare ; https://future-architect.github.io/articles/20220720a/
- Shopee: https://www.kaggle.com/code/sandersli/shopee-arcface-tf-idf-knn-threshold-search ; https://github.com/jingxuanyang/Shopee-Product-Matching
- Instacart expected-F1: https://github.com/asagar60/Instacart-Market-Basket-Analysis/blob/main/f1optimization_faron.py ; https://medium.com/kaggle-blog/instacart-market-basket-analysis-feda2700cded
- Expected-F theory: Jansche / Ye et al. ICML 2012 https://icml.cc/2012/papers/175.pdf ; Waegeman et al. JMLR 2014 https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf
- Sparkly: https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf ; SC-Block: https://arxiv.org/pdf/2303.03132
- Pre-trained embeddings for ER: https://arxiv.org/abs/2304.12329
- Ditto: https://dl.acm.org/doi/10.14778/3421424.3421431 ; LM-based EM study: https://arxiv.org/abs/2607.24688
- Fine-tuning LLMs for EM: https://arxiv.org/abs/2409.08185 ; AnyMatch: https://arxiv.org/abs/2409.04073
- R-SupCon: https://arxiv.org/pdf/2202.02098 ; Sudowoodo: https://arxiv.org/abs/2207.04122
- Multi-source clustering / transitivity: https://arxiv.org/abs/2607.26298 ; https://arxiv.org/pdf/2506.02509
- Splink term-frequency: https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html
