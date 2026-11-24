# LLM-judge results (judge: z-ai/glm-5.3-flash); cells are FC / Inc in %, PC is the remainder

## CONFLICTS

| Method | All (n) | No conf. (n) | Compl. (n) | Outdated (n) | Misinfo (n) | Opinions (n) | With ref. (n) |
|---|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 58.1 / 27.1 (458) | 64.6 / 25.5 (161) | 75.7 / 13.0 (115) | 48.4 / 41.9 (62) | 100.0 / 0.0 (5) | 34.8 / 36.5 (115) | 61.6 / 28.3 (237) |
| Vanilla RAG | 54.0 / 33.9 (458) | 70.2 / 28.6 (161) | 59.1 / 21.7 (115) | 65.6 / 31.1 (62) | 80.0 / 20.0 (5) | 19.1 / 55.6 (115) | 70.3 / 28.0 (237) |
| Hybrid RAG (BM25 + dense, RRF) | 68.6 / 17.2 (458) | 90.1 / 8.7 (161) | 77.4 / 3.5 (115) | 77.4 / 22.6 (62) | 100.0 / 0.0 (5) | 23.5 / 40.9 (115) | 86.5 / 11.8 (237) |
| LlamaIndex default (top-k 2) | 58.5 / 27.9 (458) | 80.8 / 15.5 (161) | 60.9 / 7.8 (115) | 53.2 / 43.5 (62) | 60.0 / 40.0 (5) | 27.8 / 56.5 (115) | 73.0 / 22.8 (237) |
| LlamaIndex default (top-k 8) | 67.7 / 18.1 (458) | 90.1 / 9.3 (161) | 67.0 / 7.8 (115) | 75.8 / 19.4 (62) | 80.0 / 20.0 (5) | 32.2 / 40.0 (115) | 86.1 / 11.8 (237) |
| GOV-RAG | 69.4 / 21.2 (458) | 76.4 / 19.2 (161) | 79.1 / 13.9 (115) | 61.3 / 32.3 (62) | 60.0 / 0.0 (5) | 54.8 / 26.1 (115) | 71.7 / 22.4 (237) |

## RAMDocs

| Method | All (n) | 1 gold (n) | 2 gold (n) | 3 gold (n) | With misinfo (n) | No misinfo (n) |
|---|---|---|---|---|---|---|
| Direct CoT (few-shot, no retrieval) | 9.0 / 57.8 (500) | 17.4 / 63.3 (100) | 10.0 / 52.5 (200) | 4.0 / 60.5 (200) | 9.2 / 61.0 (306) | 8.8 / 52.9 (194) |
| Vanilla RAG | 52.3 / 10.0 (500) | 62.0 / 12.0 (100) | 62.3 / 7.0 (200) | 37.5 / 12.0 (200) | 40.2 / 12.4 (306) | 71.5 / 6.2 (194) |
| Hybrid RAG (BM25 + dense, RRF) | 51.8 / 11.0 (500) | 61.0 / 14.0 (100) | 56.3 / 7.5 (200) | 42.7 / 13.1 (200) | 38.9 / 14.1 (306) | 72.4 / 6.2 (194) |
| LlamaIndex default (top-k 2) | 27.3 / 25.5 (500) | 48.0 / 25.0 (100) | 40.7 / 23.1 (200) | 3.5 / 28.1 (200) | 18.4 / 40.0 (306) | 41.5 / 2.6 (194) |
| LlamaIndex default (top-k 8) | 46.9 / 15.4 (500) | 47.0 / 14.0 (100) | 54.0 / 13.0 (200) | 39.7 / 18.6 (200) | 29.5 / 22.9 (306) | 74.2 / 3.6 (194) |
| GOV-RAG | 46.8 / 9.6 (500) | 45.0 / 11.0 (100) | 55.5 / 8.5 (200) | 38.9 / 10.1 (200) | 28.6 / 12.5 (306) | 75.3 / 5.2 (194) |

## Full FC / PC / Inc

| Dataset | Method | n | FC | PC | Inc | unlabelled |
|---|---|---|---|---|---|---|
| conflicts | Vanilla RAG | 458 | 54.0 | 12.0 | 33.9 | 1 |
| conflicts | Direct CoT (few-shot, no retrieval) | 458 | 58.1 | 14.8 | 27.1 | 0 |
| conflicts | GOV-RAG | 458 | 69.4 | 9.4 | 21.2 | 0 |
| conflicts | Hybrid RAG (BM25 + dense, RRF) | 458 | 68.6 | 14.2 | 17.2 | 0 |
| conflicts | LlamaIndex default (top-k 2) | 458 | 58.5 | 13.5 | 27.9 | 0 |
| conflicts | LlamaIndex default (top-k 8) | 458 | 67.7 | 14.2 | 18.1 | 0 |
| ramdocs | Vanilla RAG | 500 | 52.3 | 37.7 | 10.0 | 1 |
| ramdocs | Direct CoT (few-shot, no retrieval) | 500 | 9.0 | 33.1 | 57.8 | 2 |
| ramdocs | GOV-RAG | 500 | 46.8 | 43.6 | 9.6 | 2 |
| ramdocs | Hybrid RAG (BM25 + dense, RRF) | 500 | 51.8 | 37.1 | 11.0 | 2 |
| ramdocs | LlamaIndex default (top-k 2) | 500 | 27.3 | 47.2 | 25.5 | 2 |
| ramdocs | LlamaIndex default (top-k 8) | 500 | 46.9 | 37.7 | 15.4 | 1 |
