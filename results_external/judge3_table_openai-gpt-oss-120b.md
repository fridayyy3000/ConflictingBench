# LLM-judge results (judge: openai/gpt-oss-120b for every row); cells are FC / Inc in %, PC is the remainder

## Across generators (All items, FC / Inc)

### CONFLICTS

| Method | Nemotron 3.5 Super (NVIDIA API) | Qwen3.8-27B (local vLLM, thinking off) | Gemma-4-31B-it (local vLLM, thinking off) |
|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 52.4 / 38.0 | 43.2 / 47.2 | 53.5 / 38.4 |
| Vanilla RAG | 50.2 / 43.7 | 56.5 / 33.8 | 71.0 / 16.2 |
| Hybrid RAG (BM25 + dense, RRF) | 62.9 / 28.8 | 55.9 / 35.6 | 70.1 / 17.5 |
| LlamaIndex default (top-k 2) | 50.9 / 40.4 | 52.6 / 38.6 | 51.5 / 37.5 |
| LlamaIndex default (top-k 8) | 64.0 / 28.4 | 61.4 / 28.4 | 69.4 / 19.6 |
| Rankify (monoT5 + basic-rag) | 60.3 / 31.4 | 55.0 / 36.2 | 72.5 / 18.1 |
| GOV-RAG (strict, dense; ablation) | 63.8 / 29.5 | 62.0 / 27.5 | 66.4 / 21.0 |
| GOV-RAG + hybrid retrieval (strict) | 70.3 / 22.9 | 63.8 / 27.1 | 67.2 / 20.3 |
| GOV-RAG answer-and-flag (dense) | 64.0 / 29.5 | 59.0 / 33.0 | 68.8 / 19.6 |
| GOV-RAG answer-and-flag + hybrid | 64.6 / 28.6 | 57.9 / 33.2 | 67.9 / 19.9 |

### RAMDocs

| Method | Nemotron 3.5 Super (NVIDIA API) | Qwen3.8-27B (local vLLM, thinking off) | Gemma-4-31B-it (local vLLM, thinking off) |
|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 8.2 / 63.6 | 3.0 / 84.2 | 4.0 / 85.8 |
| Vanilla RAG | 49.4 / 19.0 | 43.4 / 26.2 | 37.4 / 21.4 |
| Hybrid RAG (BM25 + dense, RRF) | 50.0 / 18.0 | 45.2 / 25.4 | 40.6 / 21.0 |
| LlamaIndex default (top-k 2) | 26.0 / 33.6 | 25.2 / 35.2 | 22.8 / 35.2 |
| LlamaIndex default (top-k 8) | 43.0 / 26.0 | 45.4 / 25.8 | 39.6 / 25.2 |
| Rankify (monoT5 + basic-rag) | 49.6 / 24.4 | 41.2 / 24.8 | 45.4 / 23.8 |
| GOV-RAG (strict, dense; ablation) | 45.0 / 17.8 | 43.4 / 23.2 | 35.0 / 21.2 |
| GOV-RAG + hybrid retrieval (strict) | 45.2 / 17.0 | 43.8 / 23.0 | 34.4 / 20.6 |
| GOV-RAG answer-and-flag (dense) | 52.2 / 18.0 | 44.4 / 23.4 | 38.2 / 23.0 |
| GOV-RAG answer-and-flag + hybrid | 53.4 / 16.4 | 45.0 / 22.4 | 37.4 / 22.6 |

## Generator: Nemotron 3.5 Super (NVIDIA API)

### CONFLICTS

| Method | All (n) | No conf. (n) | Compl. (n) | Outdated (n) | Misinfo (n) | Opinions (n) | With ref. (n) |
|---|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 52.4 / 38.0 (458) | 53.4 / 28.6 (161) | 74.8 / 17.4 (115) | 46.8 / 50.0 (62) | 100.0 / 0.0 (5) | 29.6 / 67.0 (115) | 52.7 / 32.9 (237) |
| Vanilla RAG | 50.2 / 43.7 (458) | 65.8 / 29.8 (161) | 54.8 / 32.2 (115) | 64.5 / 35.5 (62) | 80.0 / 20.0 (5) | 14.8 / 80.0 (115) | 66.2 / 30.8 (237) |
| Hybrid RAG (BM25 + dense, RRF) | 62.9 / 28.8 (458) | 80.1 / 10.6 (161) | 75.7 / 8.7 (115) | 72.6 / 25.8 (62) | 100.0 / 0.0 (5) | 19.1 / 77.4 (115) | 78.5 / 14.8 (237) |
| LlamaIndex default (top-k 2) | 50.9 / 40.4 (458) | 74.5 / 18.6 (161) | 50.4 / 31.3 (115) | 50.0 / 50.0 (62) | 60.0 / 40.0 (5) | 18.3 / 74.8 (115) | 67.5 / 27.4 (237) |
| LlamaIndex default (top-k 8) | 64.0 / 28.4 (458) | 82.6 / 10.6 (161) | 66.1 / 20.9 (115) | 74.2 / 22.6 (62) | 80.0 / 20.0 (5) | 29.6 / 64.3 (115) | 80.2 / 14.3 (237) |
| Rankify (monoT5 + basic-rag) | 60.3 / 31.4 (458) | 83.2 / 9.9 (161) | 61.7 / 20.0 (115) | 80.7 / 19.4 (62) | 100.0 / 0.0 (5) | 13.9 / 80.9 (115) | 82.7 / 12.7 (237) |
| GOV-RAG (strict, dense; ablation) | 63.8 / 29.5 (458) | 65.8 / 21.7 (161) | 76.5 / 16.5 (115) | 61.3 / 37.1 (62) | 60.0 / 40.0 (5) | 49.6 / 48.7 (115) | 64.6 / 26.6 (237) |
| GOV-RAG + hybrid retrieval (strict) | 70.3 / 22.9 (458) | 72.0 / 14.3 (161) | 87.8 / 7.0 (115) | 66.1 / 32.3 (62) | 40.0 / 60.0 (5) | 53.9 / 44.4 (115) | 69.6 / 20.7 (237) |
| GOV-RAG answer-and-flag (dense) | 64.0 / 29.5 (458) | 77.6 / 9.9 (161) | 88.7 / 7.0 (115) | 72.6 / 25.8 (62) | 100.0 / 0.0 (5) | 13.9 / 82.6 (115) | 76.8 / 14.3 (237) |
| GOV-RAG answer-and-flag + hybrid | 64.6 / 28.6 (458) | 74.5 / 11.2 (161) | 91.3 / 5.2 (115) | 74.2 / 24.2 (62) | 100.0 / 0.0 (5) | 17.4 / 80.0 (115) | 75.5 / 14.3 (237) |

### RAMDocs

| Method | All (n) | 1 gold (n) | 2 gold (n) | 3 gold (n) | With misinfo (n) | No misinfo (n) |
|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 8.2 / 63.6 (500) | 16.0 / 70.0 (100) | 9.5 / 55.5 (200) | 3.0 / 68.5 (200) | 8.8 / 66.7 (306) | 7.2 / 58.8 (194) |
| Vanilla RAG | 49.4 / 19.0 (500) | 59.0 / 23.0 (100) | 56.0 / 15.5 (200) | 38.0 / 20.5 (200) | 37.6 / 23.9 (306) | 68.0 / 11.3 (194) |
| Hybrid RAG (BM25 + dense, RRF) | 50.0 / 18.0 (500) | 58.0 / 26.0 (100) | 56.0 / 14.0 (200) | 40.0 / 18.0 (200) | 37.2 / 24.5 (306) | 70.1 / 7.7 (194) |
| LlamaIndex default (top-k 2) | 26.0 / 33.6 (500) | 46.0 / 30.0 (100) | 40.0 / 32.5 (200) | 2.0 / 36.5 (200) | 17.0 / 51.0 (306) | 40.2 / 6.2 (194) |
| LlamaIndex default (top-k 8) | 43.0 / 26.0 (500) | 45.0 / 22.0 (100) | 50.0 / 26.0 (200) | 35.0 / 28.0 (200) | 25.8 / 37.6 (306) | 70.1 / 7.7 (194) |
| Rankify (monoT5 + basic-rag) | 49.6 / 24.4 (500) | 50.0 / 27.0 (100) | 55.0 / 21.0 (200) | 44.0 / 26.5 (200) | 36.0 / 32.0 (306) | 71.1 / 12.4 (194) |
| GOV-RAG (strict, dense; ablation) | 45.0 / 17.8 (500) | 41.0 / 20.0 (100) | 53.0 / 15.0 (200) | 39.0 / 19.5 (200) | 28.1 / 23.5 (306) | 71.7 / 8.8 (194) |
| GOV-RAG + hybrid retrieval (strict) | 45.2 / 17.0 (500) | 41.0 / 19.0 (100) | 55.0 / 15.5 (200) | 37.5 / 17.5 (200) | 28.1 / 21.2 (306) | 72.2 / 10.3 (194) |
| GOV-RAG answer-and-flag (dense) | 52.2 / 18.0 (500) | 57.0 / 21.0 (100) | 61.5 / 15.0 (200) | 40.5 / 19.5 (200) | 41.2 / 24.8 (306) | 69.6 / 7.2 (194) |
| GOV-RAG answer-and-flag + hybrid | 53.4 / 16.4 (500) | 60.0 / 22.0 (100) | 61.5 / 14.0 (200) | 42.0 / 16.0 (200) | 42.5 / 20.9 (306) | 70.6 / 9.3 (194) |

## Generator: Qwen3.8-27B (local vLLM, thinking off)

### CONFLICTS

| Method | All (n) | No conf. (n) | Compl. (n) | Outdated (n) | Misinfo (n) | Opinions (n) | With ref. (n) |
|---|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 43.2 / 47.2 (458) | 48.5 / 39.1 (161) | 64.3 / 20.9 (115) | 35.5 / 64.5 (62) | 80.0 / 20.0 (5) | 17.4 / 76.5 (115) | 45.6 / 45.1 (237) |
| Vanilla RAG | 56.5 / 33.8 (458) | 77.6 / 14.3 (161) | 53.9 / 27.8 (115) | 72.6 / 22.6 (62) | 100.0 / 0.0 (5) | 19.1 / 74.8 (115) | 76.4 / 16.0 (237) |
| Hybrid RAG (BM25 + dense, RRF) | 55.9 / 35.6 (458) | 78.3 / 14.3 (161) | 51.3 / 31.3 (115) | 72.6 / 24.2 (62) | 100.0 / 0.0 (5) | 18.3 / 77.4 (115) | 76.8 / 16.5 (237) |
| LlamaIndex default (top-k 2) | 52.6 / 38.6 (458) | 70.2 / 18.0 (161) | 61.7 / 26.1 (115) | 50.0 / 48.4 (62) | 80.0 / 20.0 (5) | 19.1 / 75.7 (115) | 65.0 / 26.6 (237) |
| LlamaIndex default (top-k 8) | 61.4 / 28.4 (458) | 73.9 / 10.6 (161) | 78.3 / 14.8 (115) | 72.6 / 24.2 (62) | 100.0 / 0.0 (5) | 19.1 / 70.4 (115) | 74.3 / 14.3 (237) |
| Rankify (monoT5 + basic-rag) | 55.0 / 36.2 (458) | 80.1 / 11.2 (161) | 53.9 / 28.7 (115) | 80.7 / 17.7 (62) | 100.0 / 0.0 (5) | 5.2 / 90.4 (115) | 80.6 / 13.1 (237) |
| GOV-RAG (strict, dense; ablation) | 62.0 / 27.5 (458) | 67.1 / 16.8 (161) | 71.3 / 15.7 (115) | 69.3 / 24.2 (62) | 80.0 / 20.0 (5) | 40.9 / 56.5 (115) | 68.3 / 18.6 (237) |
| GOV-RAG + hybrid retrieval (strict) | 63.8 / 27.1 (458) | 70.2 / 17.4 (161) | 76.5 / 13.9 (115) | 67.7 / 27.4 (62) | 80.0 / 0.0 (5) | 39.1 / 54.8 (115) | 70.0 / 19.4 (237) |
| GOV-RAG answer-and-flag (dense) | 59.0 / 33.0 (458) | 72.0 / 15.5 (161) | 72.2 / 16.5 (115) | 71.0 / 27.4 (62) | 100.0 / 0.0 (5) | 19.1 / 78.3 (115) | 72.6 / 18.1 (237) |
| GOV-RAG answer-and-flag + hybrid | 57.9 / 33.2 (458) | 70.8 / 15.5 (161) | 72.2 / 15.7 (115) | 71.0 / 29.0 (62) | 100.0 / 0.0 (5) | 16.5 / 79.1 (115) | 71.7 / 18.6 (237) |

### RAMDocs

| Method | All (n) | 1 gold (n) | 2 gold (n) | 3 gold (n) | With misinfo (n) | No misinfo (n) |
|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 3.0 / 84.2 (500) | 6.0 / 90.0 (100) | 3.0 / 79.5 (200) | 1.5 / 86.0 (200) | 2.6 / 87.6 (306) | 3.6 / 78.9 (194) |
| Vanilla RAG | 43.4 / 26.2 (500) | 58.0 / 29.0 (100) | 48.0 / 24.0 (200) | 31.5 / 27.0 (200) | 34.0 / 36.3 (306) | 58.2 / 10.3 (194) |
| Hybrid RAG (BM25 + dense, RRF) | 45.2 / 25.4 (500) | 60.0 / 28.0 (100) | 48.5 / 25.5 (200) | 34.5 / 24.0 (200) | 35.3 / 33.7 (306) | 60.8 / 12.4 (194) |
| LlamaIndex default (top-k 2) | 25.2 / 35.2 (500) | 53.0 / 35.0 (100) | 35.5 / 34.0 (200) | 1.0 / 36.5 (200) | 16.0 / 52.0 (306) | 39.7 / 8.8 (194) |
| LlamaIndex default (top-k 8) | 45.4 / 25.8 (500) | 53.0 / 26.0 (100) | 53.0 / 23.0 (200) | 34.0 / 28.5 (200) | 31.1 / 37.2 (306) | 68.0 / 7.7 (194) |
| Rankify (monoT5 + basic-rag) | 41.2 / 24.8 (500) | 63.0 / 32.0 (100) | 43.0 / 22.0 (200) | 28.5 / 24.0 (200) | 32.7 / 31.7 (306) | 54.6 / 13.9 (194) |
| GOV-RAG (strict, dense; ablation) | 43.4 / 23.2 (500) | 51.0 / 20.0 (100) | 47.5 / 24.0 (200) | 35.5 / 24.0 (200) | 31.4 / 31.4 (306) | 62.4 / 10.3 (194) |
| GOV-RAG + hybrid retrieval (strict) | 43.8 / 23.0 (500) | 52.0 / 22.0 (100) | 49.0 / 22.5 (200) | 34.5 / 24.0 (200) | 30.7 / 31.4 (306) | 64.4 / 9.8 (194) |
| GOV-RAG answer-and-flag (dense) | 44.4 / 23.4 (500) | 50.0 / 27.0 (100) | 52.0 / 22.0 (200) | 34.0 / 23.0 (200) | 34.6 / 31.7 (306) | 59.8 / 10.3 (194) |
| GOV-RAG answer-and-flag + hybrid | 45.0 / 22.4 (500) | 50.0 / 26.0 (100) | 52.0 / 20.0 (200) | 35.5 / 23.0 (200) | 35.0 / 30.7 (306) | 60.8 / 9.3 (194) |

## Generator: Gemma-4-31B-it (local vLLM, thinking off)

### CONFLICTS

| Method | All (n) | No conf. (n) | Compl. (n) | Outdated (n) | Misinfo (n) | Opinions (n) | With ref. (n) |
|---|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 53.5 / 38.4 (458) | 55.9 / 32.9 (161) | 73.0 / 19.1 (115) | 53.2 / 41.9 (62) | 80.0 / 20.0 (5) | 29.6 / 64.3 (115) | 56.1 / 34.6 (237) |
| Vanilla RAG | 71.0 / 16.2 (458) | 66.5 / 15.5 (161) | 73.0 / 9.6 (115) | 69.3 / 22.6 (62) | 80.0 / 0.0 (5) | 75.7 / 20.9 (115) | 67.5 / 17.3 (237) |
| Hybrid RAG (BM25 + dense, RRF) | 70.1 / 17.5 (458) | 65.2 / 16.8 (161) | 67.0 / 18.3 (115) | 72.6 / 19.4 (62) | 80.0 / 0.0 (5) | 78.3 / 17.4 (115) | 67.1 / 17.7 (237) |
| LlamaIndex default (top-k 2) | 51.5 / 37.5 (458) | 60.9 / 28.6 (161) | 56.5 / 23.5 (115) | 48.4 / 46.8 (62) | 60.0 / 20.0 (5) | 34.8 / 60.0 (115) | 57.4 / 33.3 (237) |
| LlamaIndex default (top-k 8) | 69.4 / 19.6 (458) | 67.7 / 16.1 (161) | 68.7 / 15.7 (115) | 71.0 / 19.4 (62) | 60.0 / 40.0 (5) | 72.2 / 27.8 (115) | 68.3 / 18.1 (237) |
| Rankify (monoT5 + basic-rag) | 72.5 / 18.1 (458) | 72.0 / 13.7 (161) | 75.7 / 14.8 (115) | 82.3 / 14.5 (62) | 80.0 / 0.0 (5) | 64.3 / 30.4 (115) | 74.7 / 13.9 (237) |
| GOV-RAG (strict, dense; ablation) | 66.4 / 21.0 (458) | 59.6 / 23.6 (161) | 65.2 / 17.4 (115) | 66.1 / 21.0 (62) | 40.0 / 40.0 (5) | 78.3 / 20.0 (115) | 61.2 / 23.6 (237) |
| GOV-RAG + hybrid retrieval (strict) | 67.2 / 20.3 (458) | 58.4 / 22.4 (161) | 67.8 / 17.4 (115) | 67.7 / 19.4 (62) | 40.0 / 40.0 (5) | 80.0 / 20.0 (115) | 60.8 / 22.4 (237) |
| GOV-RAG answer-and-flag (dense) | 68.8 / 19.6 (458) | 62.7 / 16.8 (161) | 73.9 / 18.3 (115) | 69.3 / 22.6 (62) | 100.0 / 0.0 (5) | 70.4 / 24.4 (115) | 65.8 / 18.1 (237) |
| GOV-RAG answer-and-flag + hybrid | 67.9 / 19.9 (458) | 64.0 / 18.0 (161) | 69.6 / 16.5 (115) | 67.7 / 25.8 (62) | 100.0 / 0.0 (5) | 70.4 / 23.5 (115) | 65.8 / 19.8 (237) |

### RAMDocs

| Method | All (n) | 1 gold (n) | 2 gold (n) | 3 gold (n) | With misinfo (n) | No misinfo (n) |
|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 4.0 / 85.8 (500) | 6.0 / 91.0 (100) | 5.5 / 83.5 (200) | 1.5 / 85.5 (200) | 2.9 / 88.2 (306) | 5.7 / 82.0 (194) |
| Vanilla RAG | 37.4 / 21.4 (500) | 34.0 / 19.0 (100) | 46.0 / 18.5 (200) | 30.5 / 25.5 (200) | 18.6 / 29.4 (306) | 67.0 / 8.8 (194) |
| Hybrid RAG (BM25 + dense, RRF) | 40.6 / 21.0 (500) | 36.0 / 21.0 (100) | 48.0 / 21.5 (200) | 35.5 / 20.5 (200) | 23.5 / 27.8 (306) | 67.5 / 10.3 (194) |
| LlamaIndex default (top-k 2) | 22.8 / 35.2 (500) | 45.0 / 36.0 (100) | 33.5 / 34.0 (200) | 1.0 / 36.0 (200) | 13.7 / 51.6 (306) | 37.1 / 9.3 (194) |
| LlamaIndex default (top-k 8) | 39.6 / 25.2 (500) | 38.0 / 28.0 (100) | 49.0 / 23.5 (200) | 31.0 / 25.5 (200) | 22.2 / 36.0 (306) | 67.0 / 8.2 (194) |
| Rankify (monoT5 + basic-rag) | 45.4 / 23.8 (500) | 39.0 / 25.0 (100) | 54.0 / 21.5 (200) | 40.0 / 25.5 (200) | 31.4 / 31.7 (306) | 67.5 / 11.3 (194) |
| GOV-RAG (strict, dense; ablation) | 35.0 / 21.2 (500) | 35.0 / 17.0 (100) | 41.5 / 20.5 (200) | 28.5 / 24.0 (200) | 15.0 / 27.4 (306) | 66.5 / 11.3 (194) |
| GOV-RAG + hybrid retrieval (strict) | 34.4 / 20.6 (500) | 33.0 / 18.0 (100) | 41.0 / 17.5 (200) | 28.5 / 25.0 (200) | 15.7 / 26.5 (306) | 63.9 / 11.3 (194) |
| GOV-RAG answer-and-flag (dense) | 38.2 / 23.0 (500) | 38.0 / 24.0 (100) | 46.0 / 19.0 (200) | 30.5 / 26.5 (200) | 20.9 / 30.1 (306) | 65.5 / 11.9 (194) |
| GOV-RAG answer-and-flag + hybrid | 37.4 / 22.6 (500) | 38.0 / 23.0 (100) | 45.0 / 21.0 (200) | 29.5 / 24.0 (200) | 19.9 / 30.1 (306) | 65.0 / 10.8 (194) |

## Full FC / PC / Inc

| Dataset | Generator | Method | n | FC | PC | Inc | unlabelled |
|---|---|---|---|---|---|---|---|
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | Vanilla RAG | 458 | 71.0 | 12.9 | 16.2 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | Direct CoT (few-shot, no retrieval) | 458 | 53.5 | 8.1 | 38.4 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG (strict, dense; ablation) | 458 | 66.4 | 12.7 | 21.0 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG answer-and-flag (dense) | 458 | 68.8 | 11.6 | 19.6 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG answer-and-flag + hybrid | 458 | 67.9 | 12.2 | 19.9 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG + hybrid retrieval (strict) | 458 | 67.2 | 12.4 | 20.3 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | Hybrid RAG (BM25 + dense, RRF) | 458 | 70.1 | 12.4 | 17.5 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | LlamaIndex default (top-k 2) | 458 | 51.5 | 10.9 | 37.5 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | LlamaIndex default (top-k 8) | 458 | 69.4 | 10.9 | 19.6 | 0 |
| conflicts | Gemma-4-31B-it (local vLLM, thinking off) | Rankify (monoT5 + basic-rag) | 458 | 72.5 | 9.4 | 18.1 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | Vanilla RAG | 458 | 50.2 | 6.1 | 43.7 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | Direct CoT (few-shot, no retrieval) | 458 | 52.4 | 9.6 | 38.0 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG (strict, dense; ablation) | 458 | 63.8 | 6.8 | 29.5 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG answer-and-flag (dense) | 458 | 64.0 | 6.5 | 29.5 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG answer-and-flag + hybrid | 458 | 64.6 | 6.8 | 28.6 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG + hybrid retrieval (strict) | 458 | 70.3 | 6.8 | 22.9 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | Hybrid RAG (BM25 + dense, RRF) | 458 | 62.9 | 8.3 | 28.8 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | LlamaIndex default (top-k 2) | 458 | 50.9 | 8.7 | 40.4 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | LlamaIndex default (top-k 8) | 458 | 64.0 | 7.6 | 28.4 | 0 |
| conflicts | Nemotron 3.5 Super (NVIDIA API) | Rankify (monoT5 + basic-rag) | 458 | 60.3 | 8.3 | 31.4 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | Vanilla RAG | 458 | 56.5 | 9.6 | 33.8 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | Direct CoT (few-shot, no retrieval) | 458 | 43.2 | 9.6 | 47.2 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG (strict, dense; ablation) | 458 | 62.0 | 10.5 | 27.5 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG answer-and-flag (dense) | 458 | 59.0 | 8.1 | 33.0 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG answer-and-flag + hybrid | 458 | 57.9 | 8.9 | 33.2 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG + hybrid retrieval (strict) | 458 | 63.8 | 9.2 | 27.1 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | Hybrid RAG (BM25 + dense, RRF) | 458 | 55.9 | 8.5 | 35.6 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | LlamaIndex default (top-k 2) | 458 | 52.6 | 8.7 | 38.6 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | LlamaIndex default (top-k 8) | 458 | 61.4 | 10.3 | 28.4 | 0 |
| conflicts | Qwen3.8-27B (local vLLM, thinking off) | Rankify (monoT5 + basic-rag) | 458 | 55.0 | 8.7 | 36.2 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | Vanilla RAG | 500 | 37.4 | 41.2 | 21.4 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | Direct CoT (few-shot, no retrieval) | 500 | 4.0 | 10.2 | 85.8 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG (strict, dense; ablation) | 500 | 35.0 | 43.8 | 21.2 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG answer-and-flag (dense) | 500 | 38.2 | 38.8 | 23.0 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG answer-and-flag + hybrid | 500 | 37.4 | 40.0 | 22.6 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | GOV-RAG + hybrid retrieval (strict) | 500 | 34.4 | 45.0 | 20.6 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | Hybrid RAG (BM25 + dense, RRF) | 500 | 40.6 | 38.4 | 21.0 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | LlamaIndex default (top-k 2) | 500 | 22.8 | 42.0 | 35.2 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | LlamaIndex default (top-k 8) | 500 | 39.6 | 35.2 | 25.2 | 0 |
| ramdocs | Gemma-4-31B-it (local vLLM, thinking off) | Rankify (monoT5 + basic-rag) | 500 | 45.4 | 30.8 | 23.8 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | Vanilla RAG | 500 | 49.4 | 31.6 | 19.0 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | Direct CoT (few-shot, no retrieval) | 500 | 8.2 | 28.2 | 63.6 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG (strict, dense; ablation) | 500 | 45.0 | 37.2 | 17.8 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG answer-and-flag (dense) | 500 | 52.2 | 29.8 | 18.0 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG answer-and-flag + hybrid | 500 | 53.4 | 30.2 | 16.4 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | GOV-RAG + hybrid retrieval (strict) | 500 | 45.2 | 37.8 | 17.0 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | Hybrid RAG (BM25 + dense, RRF) | 500 | 50.0 | 32.0 | 18.0 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | LlamaIndex default (top-k 2) | 500 | 26.0 | 40.4 | 33.6 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | LlamaIndex default (top-k 8) | 500 | 43.0 | 31.0 | 26.0 | 0 |
| ramdocs | Nemotron 3.5 Super (NVIDIA API) | Rankify (monoT5 + basic-rag) | 500 | 49.6 | 26.0 | 24.4 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | Vanilla RAG | 500 | 43.4 | 30.4 | 26.2 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | Direct CoT (few-shot, no retrieval) | 500 | 3.0 | 12.8 | 84.2 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG (strict, dense; ablation) | 500 | 43.4 | 33.4 | 23.2 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG answer-and-flag (dense) | 500 | 44.4 | 32.2 | 23.4 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG answer-and-flag + hybrid | 500 | 45.0 | 32.6 | 22.4 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | GOV-RAG + hybrid retrieval (strict) | 500 | 43.8 | 33.2 | 23.0 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | Hybrid RAG (BM25 + dense, RRF) | 500 | 45.2 | 29.4 | 25.4 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | LlamaIndex default (top-k 2) | 500 | 25.2 | 39.6 | 35.2 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | LlamaIndex default (top-k 8) | 500 | 45.4 | 28.8 | 25.8 | 0 |
| ramdocs | Qwen3.8-27B (local vLLM, thinking off) | Rankify (monoT5 + basic-rag) | 500 | 41.2 | 34.0 | 24.8 | 0 |
