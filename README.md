# airr_ml_2025

Rozwiązanie konkursu [**Adaptive Immune Profiling Challenge 2025**](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025) (Kaggle). Klasyfikacja na podstawie repertuarów TCR oraz odkrywanie istotnych receptorów.

Format notatnika (Jupyter) wybrany został dlatego, że zadanie realizowane było w ramach przedmiotu **uczenie maszynowe** na **Uniwersytecie Warszawskim**.

## Krótki opis rozwiązania

- **Zadanie 1 (immune-state):** Cechy repertuaru (m.in. liczba unikalnych junction, długości CDR3, częstości genów V/J, k-mery) są liczone na poziomie repertuaru; klasyfikacja — pipeline z normalizacją i regresją logistyczną (lub innym klasyfikatorem), osobny model per dataset, walidacja krzyżowa; na zbiorze testowym zwracane są prawdopodobieństwa (`label_positive_probability`).
- **Zadanie 2 (receptor discovery):** Dla każdego zbioru treningowego sekwencje TCR (junction_aa, v_call, j_call) są rangowane statystycznie (test Fishera) pod kątem związku z etykietą; do submissionu wybierane jest top 50 000 sekwencji.

## Konkurs

- **Platforma:** Kaggle  
- **Link:** https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025  

**Zadania:**
1. **Immune-state prediction** — przewidywanie stanu (zdrowy/chory) na podstawie repertuaru TCR; metryka ROC AUC.
2. **Receptor discovery** — wyłonienie top 50 000 istotnych sekwencji (junction_aa, v_call, j_call); metryka Jaccard.

## Wymagania

- Python 3.10+
- Zależności: patrz `requirements.txt`

## Instalacja

```bash
git clone <url-repozytorium>
cd airr_ml_2025
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dane

Dane konkursowe nie są w repozytorium. Należy pobrać zbiór ze [strony konkursu](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/data) i umieścić w katalogu `data/` (na tym samym poziomie co `src/`, czyli w katalogu głównym repozytorium — względem `src/` jest to `../data`).

Struktura po pobraniu:
```
data/
  train/          # dane treningowe według datasetów
  test/           # test (jeśli przewidziany)
  ...
```

## Uruchomienie

1. Aktywuj środowisko i zainstaluj zależności (patrz wyżej).
2. Umieść dane w `data/` (lub wskaż ścieżkę w notatniku).
3. Otwórz i uruchom notatnik:
   - `src/unified-models.ipynb` — EDA, uczenie i inferencja.

Aby odtworzyć submission od zera: uruchom wszystkie komórki notatnika; przy użyciu wytrenowanych modeli — umieść je w `models/` i ładuj w kodzie.

## Struktura repozytorium

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/              # dane konkursowe (poza git)
├── models/            # wagi/checkpointy (poza git)
└── src/
    ├── unified-models.ipynb
    └── utils.py
```

## Licencja

MIT — patrz [LICENSE](LICENSE).
