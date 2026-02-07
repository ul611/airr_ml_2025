import os
import sys
import gc
import glob
import json
import argparse

from collections import defaultdict, Counter
from typing import Iterator, Tuple, Union, List

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    import polars as pl
except ImportError:
    pl = None

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.pipeline import Pipeline

# CatBoost (opcjonalnie)
try:
    from catboost import CatBoostClassifier, cv, Pool as CatBoostPool
except ImportError:
    CatBoostClassifier = None
    cv = None
    CatBoostPool = None

# Wykresy (opcjonalnie, nie włączy magicznych funkcji Jupyter)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (12, 6)
except ImportError:
    plt = None

try:
    import plotly
    import plotly.graph_objects as go
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)
    print("Plotly zainicjalizowany dla interaktywnych wykresów")
except ImportError:
    plotly = None
    go = None

# Statystyki/narzędzia
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from functools import partial
from multiprocessing import Pool, cpu_count

# Zapisywanie/ładowanie modeli
try:
    import joblib  # do zapisywania/ładowania modeli
except ImportError:
    joblib = None

# SHAP (opcjonalnie)
try:
    import shap
except ImportError:
    shap = None


def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    Generator do ładowania danych repertuaru immunologicznego.

    Ta funkcja działa w dwóch trybach:
    1.  Jeśli znaleziono metadane, zwraca dane na podstawie pliku metadanych. (głównie zbiory treningowe)
    2.  Jeśli metadanych NIE znaleziono, używa glob do znalezienia i zwrócenia wszystkich plików '.tsv' (głównie zbiory testowe)
        w katalogu.

    Args:
        data_dir (str): Ścieżka do katalogu zawierającego dane.

    Yields:
        Iterator krotek. Format zależy od trybu:
        - Z metadanymi: (repertoire_id, pd.DataFrame, label_positive)
        - Bez metadanych: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Ostrzeżenie: Plik '{row.filename}' wymieniony w metadanych nie został znaleziony. Pomijanie.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Ostrzeżenie: Nie można odczytać pliku '{file_path}'. Błąd: {e}. Pomijanie.")
                continue


def filter_low_quality_sequences(df: pd.DataFrame, min_length: int = 8, max_length: int = 90, 
                                  min_reads: int = 1, sequence_col: str = 'junction_aa') -> pd.DataFrame:
    """
    Filtrowanie sekwencji niskiej jakości.
    
    Args:
        df: DataFrame z sekwencjami
        min_length: Minimalna długość CDR3
        max_length: Maksymalna długość CDR3
        min_reads: Minimalna liczba odczytów (jeśli istnieje duplicate_count lub templates)
        sequence_col: Nazwa kolumny z sekwencją (junction_aa lub sequence_aa)
    
    Returns:
        Przefiltrowany DataFrame
    """
    initial_count = len(df)
    
    # Określamy nazwę kolumny z sekwencją
    if sequence_col not in df.columns:
        if 'sequence_aa' in df.columns:
            sequence_col = 'sequence_aa'
        elif 'junction_aa' in df.columns:
            sequence_col = 'junction_aa'
        else:
            print("Ostrzeżenie: Nie znaleziono kolumny z sekwencją. Pomijanie filtrowania.")
            return df
    
    # 1. Usunąć sekwencje z niestandardowymi aminokwasami
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    df = df[df[sequence_col].apply(
        lambda x: pd.notna(x) and all(aa in valid_aa for aa in str(x))
    )]
    
    # 2. Usunąć kodony stop (jeśli jest '*')
    df = df[~df[sequence_col].str.contains(r'\*', na=False, regex=True)]
    
    # 3. Usunąć sekwencje out-of-frame
    # CDR3 powinien zaczynać się od C i kończyć na F lub W
    df = df[df[sequence_col].str.match(r'^C.*[FW]$', na=False)]
    
    # 4. Filtr według długości
    df = df[df[sequence_col].str.len().between(min_length, max_length, inclusive='both')]
    
    # 5. Usunąć sekwencje z niską liczbą odczytów (jeśli istnieje)
    if 'duplicate_count' in df.columns:
        df = df[df['duplicate_count'] >= min_reads]
    elif 'templates' in df.columns:
        df = df[df['templates'] >= min_reads]
    
    # 6. Usunąć niejednoznaczne wywołania V/J
    if 'v_call' in df.columns:
        df = df.dropna(subset=['v_call'])
    if 'j_call' in df.columns:
        df = df.dropna(subset=['j_call'])
    
    removed = initial_count - len(df)
    
    return df.reset_index(drop=True)


def filter_sequences_polars(df: pl.DataFrame, min_length: int = 8, max_length: int = 90,
                            min_reads: int = 1, sequence_col: str = 'sequence_aa') -> pl.DataFrame:
    """
    Filtrowanie sekwencji niskiej jakości dla Polars DataFrame.
    
    Args:
        df: Polars DataFrame z sekwencjami
        min_length: Minimalna długość CDR3
        max_length: Maksymalna długość CDR3
        min_reads: Minimalna liczba odczytów (jeśli istnieje templates)
        sequence_col: Nazwa kolumny z sekwencją
    
    Returns:
        Przefiltrowany Polars DataFrame
    """
    if pl is None:
        raise ValueError("Polars jest wymagany")
    
    initial = df.height
    
    # Określamy nazwę kolumny z sekwencją
    if sequence_col not in df.columns:
        if 'sequence_aa' in df.columns:
            sequence_col = 'sequence_aa'
        elif 'junction_aa' in df.columns:
            sequence_col = 'junction_aa'
            df = df.rename({'junction_aa': 'sequence_aa'})
        else:
            print("Ostrzeżenie: Nie znaleziono kolumny z sekwencją. Pomijanie filtrowania.")
            return df
    
    valid_aa_pattern = r'^[ACDEFGHIKLMNPQRSTVWY]+$'
    cdr3_pattern = r'^C.*[FW]$'
    
    filters = [
        pl.col(sequence_col).str.contains(valid_aa_pattern),
        ~pl.col(sequence_col).str.contains(r'\*'),
        pl.col(sequence_col).str.contains(cdr3_pattern),
        pl.col(sequence_col).str.len_chars().is_between(min_length, max_length, closed='both')
    ]
    
    if 'v_call' in df.columns:
        filters.append(pl.col('v_call').is_not_null())
    if 'j_call' in df.columns:
        filters.append(pl.col('j_call').is_not_null())
    
    if 'templates' in df.columns:
        filters.append(pl.col('templates') >= min_reads)
    
    df = df.filter(pl.all_horizontal(filters))
    
    removed = initial - df.height
    
    return df


def load_full_dataset(data_dir: str, filter_sequences: bool = True, 
                     min_length: int = 8, max_length: int = 90, min_reads: int = 1) -> pd.DataFrame:
    """
    Ładuje wszystkie pliki TSV z katalogu i łączy je w jeden DataFrame.

    Ta funkcja obsługuje dwa scenariusze:
    1. Jeśli metadata.csv istnieje, ładuje dane na podstawie metadanych i dodaje
       kolumny 'repertoire_id' i 'label_positive'.
    2. Jeśli metadata.csv nie istnieje, ładuje wszystkie pliki .tsv i dodaje
       kolumnę 'filename' jako identyfikator.

    Args:
        data_dir (str): Ścieżka do katalogu z danymi.
        filter_sequences (bool): Czy filtrować sekwencje niskiej jakości (domyślnie True)
        min_length (int): Minimalna długość CDR3 (domyślnie 8)
        max_length (int): Maksymalna długość CDR3 (domyślnie 90)
        min_reads (int): Minimalna liczba odczytów (domyślnie 1)

    Returns:
        pd.DataFrame: Pojedynczy, połączony DataFrame zawierający wszystkie dane.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)  
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Ładowanie plików"):
            # Stosujemy filtrowanie jeśli potrzeba
            if filter_sequences:
                data_df = filter_low_quality_sequences(data_df, min_length, max_length, min_reads)
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Ładowanie plików"):
            # Stosujemy filtrowanie jeśli potrzeba
            if filter_sequences:
                data_df = filter_low_quality_sequences(data_df, min_length, max_length, min_reads)
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Ostrzeżenie: Nie załadowano żadnych plików danych.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


def load_and_encode_kmers(data_dir: str, k: int = 3, use_templates: bool = True, 
                          include_vj_features: bool = False, min_count: int = 1,
                          filter_sequences: bool = True, min_length: int = 8, 
                          max_length: int = 25, min_reads: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ładowanie i kodowanie k-merów danych repertuaru z opcjonalnymi wagami templates i częstotliwościami v/j/d_call.

    Args:
        data_dir: Ścieżka do katalogu z danymi
        k: Długość k-meru (3 lub 4)
        use_templates: Czy używać templates jako wag (jeśli dostępne)
        include_vj_features: Czy uwzględnić częstotliwości v_call, j_call, d_call
        min_count: Minimalny próg liczby dla cech (domyślnie 1, użyj 3 do filtrowania)
        filter_sequences: Czy filtrować sekwencje niskiej jakości (domyślnie True)
        min_length: Minimalna długość CDR3 (domyślnie 8)
        max_length: Maksymalna długość CDR3 (domyślnie 90)
        min_reads: Minimalna liczba odczytów (domyślnie 1)

    Returns:
        Krotka (encoded_features_df, metadata_df)
        metadata_df zawsze zawiera 'ID', i 'label_positive' jeśli dostępne
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Kodowanie {k}-merów"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None
        
        # Stosujemy filtrowanie jeśli potrzeba
        if filter_sequences:
            data_df = filter_low_quality_sequences(data_df, min_length, max_length, min_reads)

        kmer_counts = Counter()
        v_call_counts = Counter()
        j_call_counts = Counter()
        d_call_counts = Counter()
        
        junction_attr = 'junction_aa'.replace(' ', '_').replace('-', '_')
        templates_attr = 'templates'.replace(' ', '_').replace('-', '_')
        v_call_attr = 'v_call'.replace(' ', '_').replace('-', '_')
        j_call_attr = 'j_call'.replace(' ', '_').replace('-', '_')
        d_call_attr = 'd_call'.replace(' ', '_').replace('-', '_')
        
        if use_templates and 'templates' in data_df.columns:
            for row in data_df.itertuples(index=False):
                seq = getattr(row, junction_attr, None)
                weight = getattr(row, templates_attr, 1)
                if pd.isna(weight) or weight <= 0:
                    weight = 1
                
                if pd.notna(seq):
                    for i in range(len(seq) - k + 1):
                        kmer_counts[seq[i:i + k]] += weight
                
                if include_vj_features:
                    v_call = getattr(row, v_call_attr, None)
                    if pd.notna(v_call) and v_call != '':
                        v_call_counts[str(v_call)] += weight
                    
                    j_call = getattr(row, j_call_attr, None)
                    if pd.notna(j_call) and j_call != '':
                        j_call_counts[str(j_call)] += weight
                    
                    d_call = getattr(row, d_call_attr, None)
                    if pd.notna(d_call) and d_call != '':
                        d_call_counts[str(d_call)] += weight
        else:
            for row in data_df.itertuples(index=False):
                seq = getattr(row, junction_attr, None)
                if pd.notna(seq):
                    for i in range(len(seq) - k + 1):
                        kmer_counts[seq[i:i + k]] += 1
                
                if include_vj_features:
                    v_call = getattr(row, v_call_attr, None)
                    if pd.notna(v_call) and v_call != '':
                        v_call_counts[str(v_call)] += 1
                    
                    j_call = getattr(row, j_call_attr, None)
                    if pd.notna(j_call) and j_call != '':
                        j_call_counts[str(j_call)] += 1
                    
                    d_call = getattr(row, d_call_attr, None)
                    if pd.notna(d_call) and d_call != '':
                        d_call_counts[str(d_call)] += 1
        
        # Filtruj według min_count
        kmer_counts_filtered = {kmer: count for kmer, count in kmer_counts.items() if count >= min_count}
        
        feature_dict = {
            'ID': rep_id,
            **kmer_counts_filtered
        }
        
        if include_vj_features:
            for v_call, count in v_call_counts.items():
                if count >= min_count:
                    feature_dict[f'v_call_{v_call}'] = count
            
            for j_call, count in j_call_counts.items():
                if count >= min_count:
                    feature_dict[f'j_call_{j_call}'] = count
            
            for d_call, count in d_call_counts.items():
                if count >= min_count:
                    feature_dict[f'd_call_{d_call}'] = count
        
        repertoire_features.append(feature_dict)

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts, kmer_counts_filtered, v_call_counts, j_call_counts, d_call_counts

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    features_df.fillna(0)
    
    # Optymalizacja pamięci: konwertujemy float64 na float32
    for col in features_df.columns:
        if features_df[col].dtype == 'float64':
            features_df[col] = features_df[col].astype('float32')
    
    del repertoire_features
    gc.collect()
    metadata_df = pd.DataFrame(metadata_records)
    del metadata_records
    gc.collect()

    return features_df, metadata_df


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Katalog treningowy `{train_dir}` nie istnieje."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"Nie znaleziono plików .tsv w katalogu treningowym `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` nie znaleziono w katalogu treningowym `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Katalog testowy `{test_dir}` nie istnieje."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"Nie znaleziono plików .tsv w katalogu testowym `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Nie udało się utworzyć lub zapisać do katalogu wyjściowego `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Łączy wszystkie pliki TSV z predykcjami testowymi i ważnymi sekwencjami z katalogu wyjściowego.

    Ta funkcja znajduje wszystkie pliki pasujące do wzorców:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    i łączy je, aby dopasować oczekiwany format wyjściowy submissions.csv.

    Args:
        out_dir (str): Ścieżka do katalogu wyjściowego zawierającego pliki TSV.

    Returns:
        pd.DataFrame: Połączony DataFrame z predykcjami, po których następują ważne sekwencje.
                     Kolumny: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Ostrzeżenie: Nie można odczytać pliku predykcji '{pred_file}'. Błąd: {e}. Pomijanie.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Ostrzeżenie: Nie można odczytać pliku sekwencji '{seq_file}'. Błąd: {e}. Pomijanie.")
            continue

    if not df_list:
        print("Ostrzeżenie: Nie znaleziono plików wyjściowych do połączenia.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Połączone dane wyjściowe zapisane do `{submissions_file}`.")


def save_metrics(predictor, dataset_name: str, results_dir: str, 
                 config_params: dict = None) -> None:
    """
    Zapisuje metryki modelu dla zbioru danych.
    
    Args:
        predictor: Wytrenowany predictor (KmerImmuneStatePredictor lub inny)
        dataset_name: Nazwa zbioru danych
        results_dir: Katalog do zapisywania wyników
        config_params: Słownik z parametrami konfiguracji (opcjonalnie)
    """
    metrics = {
        'dataset': dataset_name
    }
    
    # Dodajemy parametry konfiguracji jeśli są
    if config_params:
        metrics.update(config_params)
    
    # Pobieramy metryki z modelu
    if hasattr(predictor, 'model') and predictor.model is not None:
        model = predictor.model
        if hasattr(model, 'cv_roc_auc_'):
            metrics['cv_roc_auc'] = model.cv_roc_auc_
        if hasattr(model, 'cv_roc_auc_std_'):
            metrics['cv_roc_auc_std'] = model.cv_roc_auc_std_
        if hasattr(model, 'val_score_'):
            metrics['validation_roc_auc'] = model.val_score_
        if hasattr(model, 'mcc_score_'):
            metrics['validation_mcc'] = model.mcc_score_
        if hasattr(model, 'best_C_'):
            metrics['best_C'] = model.best_C_
        if hasattr(model, 'best_depth_'):
            metrics['best_depth'] = model.best_depth_
        if hasattr(model, 'best_learning_rate_'):
            metrics['best_learning_rate'] = model.best_learning_rate_
    
    # Zapisujemy do pliku
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    # Jeśli plik istnieje, dodajemy wiersz, w przeciwnym razie tworzymy nowy
    if os.path.exists(metrics_file):
        existing_df = pd.read_csv(metrics_file)
        new_df = pd.DataFrame([metrics])
        # Łączymy, zastępując duplikaty według dataset
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Usuwamy duplikaty, zostawiając ostatni wpis
        combined_df = combined_df.drop_duplicates(subset=['dataset'], keep='last')
        combined_df.to_csv(metrics_file, index=False)
    else:
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    
    print(f"Metryki zapisane do `{metrics_file}`")


class BaseKmerClassifier:
    """Klasa bazowa dla klasyfikatorów k-merów ze wspólną funkcjonalnością."""
    
    def __init__(self, cv_folds=5, opt_metric='roc_auc', random_state=123, n_jobs=1,
                 force_invert_predictions=None, disable_invert_predictions=False):
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None
        self.mcc_score_ = None
        self.cv_roc_auc_ = None
        self.cv_roc_auc_std_ = None
        self.invert_predictions_ = False
        self.X_train_ = None
        self.kmer_shap_importance_ = None
        self.force_invert_predictions = force_invert_predictions
        self.disable_invert_predictions = disable_invert_predictions
    
    def _get_scorer(self):
        """Pobiera funkcję scoringową do optymalizacji."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Nieznana metryka: {self.opt_metric}")
    
    def _handle_auc_inversion(self, force_invert=None, disable_invert=False):
        """
        Obsługuje AUC < 0.5 poprzez wybór najlepszego modelu na podstawie odchylenia od 0.5 lub max AUC.
        
        Args:
            force_invert: Jeśli True, wymusza inwersję. Jeśli False, wymusza brak inwersji. Jeśli None, auto-detekcja.
            disable_invert: Jeśli True, wyłącza automatyczną inwersję (force_invert=False).
        
        Returns:
            Indeks najlepszego modelu
        """
        if disable_invert:
            force_invert = False
        
        if self.opt_metric == 'roc_auc':
            self.cv_results_['adjusted_score'] = self.cv_results_['mean_score'].apply(
                lambda x: max(x, 1 - x)
            )
            self.cv_results_['deviation_from_05'] = self.cv_results_['mean_score'].apply(
                lambda x: abs(x - 0.5)
            )
            best_idx_by_auc = self.cv_results_['adjusted_score'].idxmax()
            best_idx_by_dev = self.cv_results_['deviation_from_05'].idxmax()
            
            # Jeśli std duże (model losowy), używamy odchylenia od 0.5
            if self.cv_results_.loc[best_idx_by_auc, 'std_score'] > 0.1:
                best_idx = best_idx_by_dev
            else:
                best_idx = best_idx_by_auc
            
            # Określamy inwersję
            if force_invert is not None:
                self.invert_predictions_ = force_invert
            else:
                self.invert_predictions_ = self.cv_results_.loc[best_idx, 'mean_score'] < 0.5
            
            return best_idx
        else:
            best_idx = self.cv_results_['mean_score'].idxmax()
            if force_invert is not None:
                self.invert_predictions_ = force_invert
            else:
                self.invert_predictions_ = False
            return best_idx
    
    def predict_proba(self, X):
        """Przewiduje prawdopodobieństwa klas."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        proba = self._predict_proba_internal(X)
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            proba = 1 - proba
        return proba
    
    def _predict_proba_internal(self, X):
        """Wewnętrzna metoda do uzyskania surowych prawdopodobieństw (do nadpisania)."""
        raise NotImplementedError
    
    def predict(self, X):
        """Przewiduje etykiety klas."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = self._predict_internal(X)
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            pred = 1 - pred
        return pred
    
    def _predict_internal(self, X):
        """Wewnętrzna metoda do uzyskania surowych predykcji (do nadpisania)."""
        raise NotImplementedError
    
    def get_feature_importance(self, sample_size=10000, use_shap=True):
        """
        Pobiera ważność cech używając wartości SHAP lub współczynników modelu.
        Do nadpisania przez podklasy.
        """
        raise NotImplementedError
    
    def score_all_sequences(self, sequences_df, sequence_col='junction_aa', 
                           return_kmer_importance=False, use_shap=True, shap_sample_size=10000):
        """
        Ocenia wszystkie sekwencje używając ważności cech.
        Do nadpisania przez podklasy.
        """
        raise NotImplementedError
    
    def select_important_features(self, importance_df, top_n=None, threshold=None):
        """
        Wybiera ważne cechy na podstawie wyników ważności.
        
        Args:
            importance_df: DataFrame z kolumnami ['feature', 'coefficient', 'abs_coefficient']
            top_n: Liczba najważniejszych cech do wyboru
            threshold: Minimalny próg bezwzględnej ważności
        
        Returns:
            Zbiór wybranych nazw cech
        """
        if top_n is not None:
            top_features = importance_df.nlargest(top_n, 'abs_coefficient')
            return set(top_features['feature'].tolist())
        elif threshold is not None:
            important_features = importance_df[importance_df['abs_coefficient'] >= threshold]
            return set(important_features['feature'].tolist())
        else:
            return set(importance_df['feature'].tolist())


class LogisticRegressionKmerClassifier(BaseKmerClassifier):
    """Regresja logistyczna z regularyzacją L1 dla danych liczby k-merów z obsługą AUC < 0.5."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='roc_auc', random_state=123, n_jobs=1,
                 force_invert_predictions=None, disable_invert_predictions=False):
        super().__init__(cv_folds, opt_metric, random_state, n_jobs,
                        force_invert_predictions, disable_invert_predictions)
        if c_values is None:
            c_values = [1, 0.1, 0.05, 0.03]
        self.c_values = c_values
        self.best_C_ = None

    def _make_pipeline(self, C):
        """Tworzy pipeline standaryzacji + regresji logistycznej L1."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])

    def _get_scorer(self):
        """Pobiera funkcję scoringową do optymalizacji."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Nieznana metryka: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.2, selected_features=None):
        """
        Wykonuje tuning CV na zbiorze treningowym i dopasowuje, z obsługą AUC < 0.5.
        
        Args:
            X: Macierz cech
            y: Etykiety
            val_size: Rozmiar zbioru walidacyjnego
            selected_features: Zbiór nazw cech do użycia (do trenowania tylko na ważnych cechach)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Filtruje cechy jeśli określono
        if selected_features is not None and self.feature_names_ is not None:
            feature_indices = [i for i, name in enumerate(self.feature_names_) if name in selected_features]
            X = X[:, feature_indices]
            self.feature_names_ = [name for name in self.feature_names_ if name in selected_features]
            print(f"Trening na {len(self.feature_names_)} wybranych cechach")

        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y, shuffle=True)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                     n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self._handle_auc_inversion(
            force_invert=self.force_invert_predictions,
            disable_invert=self.disable_invert_predictions
        )
        
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']
        print(f"Najlepsze C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f} ± {self.cv_results_.loc[best_idx, 'std_score']:.4f})")
        if self.invert_predictions_:
            print("Uwaga: Predykcje zostaną odwrócone (wykryto AUC < 0.5)")

        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)
        
        if self.feature_names_ is not None:
            self.X_train_ = pd.DataFrame(X_train, columns=self.feature_names_)
        else:
            self.X_train_ = pd.DataFrame(X_train)
        
        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.predict(X_val))
            else:
                self.val_score_ = roc_auc_score(y_val, self.predict_proba(X_val))
            # Wyświetlamy validation score i std z CV dla najlepszego modelu
            best_std = self.cv_results_.loc[best_idx, 'std_score']
            print(f"Walidacja {self.opt_metric}: {self.val_score_:.4f}")
            
            # Obliczamy MCC
            y_val_pred = self.predict(X_val)
            self.mcc_score_ = matthews_corrcoef(y_val, y_val_pred)
            print(f"Walidacja MCC: {self.mcc_score_:.4f}")
        
        # Zapisujemy metryki CV
        if self.opt_metric == 'roc_auc':
            self.cv_roc_auc_ = self.best_score_
            self.cv_roc_auc_std_ = self.cv_results_.loc[best_idx, 'std_score']
        
        return self

    def _predict_proba_internal(self, X):
        """Wewnętrzna metoda do uzyskania surowych prawdopodobieństw."""
        return self.model_.predict_proba(X)[:, 1]
    
    def _predict_internal(self, X):
        """Wewnętrzna metoda do uzyskania surowych predykcji."""
        return self.model_.predict(X)

    def get_feature_importance(self, sample_size=10000, use_shap=True):
        """
        Pobiera ważność cech używając wartości SHAP lub współczynników modelu.

        Parametry:
            sample_size: Maksymalna liczba próbek do użycia w obliczeniach SHAP
            use_shap: Czy używać wartości SHAP. Jeśli False lub SHAP niedostępny, używa współczynników.

        Zwraca:
            pd.DataFrame z kolumnami ['feature', 'coefficient', 'abs_coefficient']
        """
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")

        if use_shap and shap is not None and self.X_train_ is not None:
            try:
                sample_size_actual = min(sample_size, len(self.X_train_))
                X_sample = self.X_train_.iloc[:sample_size_actual]
                
                explainer = shap.LinearExplainer(
                    self.model_.named_steps['classifier'],
                    X_sample
                )
                
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                shap_importance = shap_values.mean(axis=0)
                
                if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
                    shap_importance = -shap_importance
                
                if self.feature_names_ is not None:
                    feature_names = self.feature_names_
                else:
                    feature_names = [f"feature_{i}" for i in range(len(shap_importance))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': shap_importance,
                    'abs_coefficient': np.abs(shap_importance)
                })
                
                importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
                return importance_df
                
            except Exception as e:
                print(f"Ostrzeżenie: Nie udało się obliczyć wartości SHAP: {e}. Powrót do współczynników.")
        
        # Fallback: używamy współczynników modelu
        coef = self.model_.named_steps['classifier'].coef_[0]
        
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            coef = -coef

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df

    def score_all_sequences(self, sequences_df, sequence_col='junction_aa', return_kmer_importance=False, 
                           use_shap=True, shap_sample_size=10000):
        """
        Ocenia wszystkie sekwencje używając ważności opartej na SHAP lub współczynnikach modelu.

        Parametry:
            sequences_df: DataFrame z unikalnymi sekwencjami
            sequence_col: Nazwa kolumny zawierającej sekwencje
            return_kmer_importance: Czy zwrócić surowe ważności k-merów
            use_shap: Czy używać wartości SHAP (domyślnie True)
            shap_sample_size: Rozmiar próbki dla obliczeń SHAP jeśli nie w cache

        Zwraca:
            result_df: DataFrame z dodaną kolumną 'importance_score'
            kmer_importance_df (opcjonalnie): DataFrame z kolumnami ['kmer', 'importance']
        """
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")

        if use_shap and shap is not None and self.X_train_ is not None:
            if self.kmer_shap_importance_ is None:
                importance_df = self.get_feature_importance(sample_size=shap_sample_size, use_shap=use_shap)
                self.kmer_shap_importance_ = dict(zip(importance_df['feature'], importance_df['coefficient']))
            
            kmer_importance_dict = self.kmer_shap_importance_
        else:
            scaler = self.model_.named_steps['scaler']
            coefficients = self.model_.named_steps['classifier'].coef_[0]
            coefficients = coefficients / scaler.scale_

            if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
                coefficients = -coefficients

            kmer_importance_dict = dict(zip(self.feature_names_, coefficients))

        kmer_importance_df = pd.DataFrame({
            "kmer": list(kmer_importance_dict.keys()),
            "importance": list(kmer_importance_dict.values())
        })

        kmer_set = set(kmer_importance_dict.keys())
        k = len(self.feature_names_[0]) if self.feature_names_ else 3

        scores = []
        total_seqs = len(sequences_df)
        for seq in tqdm(sequences_df[sequence_col], total=total_seqs, desc="Ocenianie sekwencji"):
            score = 0.0
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                if kmer in kmer_set:
                    score += kmer_importance_dict[kmer]
            scores.append(score)

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores

        if return_kmer_importance:
            return result_df, kmer_importance_df
        else:
            return result_df


class BurdenClassifier:
    """Regresja logistyczna z regularyzacją L1/L2 dla cech obciążenia."""

    def __init__(self, c_values=None, cv_folds=5, penalty='l2',
                 opt_metric='roc_auc', random_state=42, n_jobs=1,
                 force_invert_predictions=None, disable_invert_predictions=False):
        if c_values is None:
            c_values = [10, 1, 0.1, 0.01]
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.penalty = penalty
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None
        self.mcc_score_ = None
        self.cv_roc_auc_ = None
        self.cv_roc_auc_std_ = None
        self.invert_predictions_ = False
        self.force_invert_predictions = force_invert_predictions
        self.disable_invert_predictions = disable_invert_predictions

    def _make_pipeline(self, C):
        """Tworzy pipeline standaryzacji + regresji logistycznej."""
        solver = 'liblinear' if self.penalty == 'l1' else 'lbfgs'
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty=self.penalty,
                C=C,
                solver=solver,
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])

    def _get_scorer(self):
        """Pobiera funkcję scoringową do optymalizacji."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Nieznana metryka: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.2, X_val=None, y_val=None):
        """Wykonuje tuning CV na zbiorze treningowym i dopasowuje.
        Jeśli podano X_val, y_val (repertoiry nieużyte przy wyborze cech), używane są do walidacji zamiast dzielenia X,y."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if X_val is not None and y_val is not None:
            X_train, y_train = X, y
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
        elif val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                     n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        
        # Obsługuje inwersję AUC jeśli potrzebna
        if self.opt_metric == 'roc_auc':
            self.cv_results_['adjusted_score'] = self.cv_results_['mean_score'].apply(
                lambda x: max(x, 1 - x)
            )
            self.cv_results_['deviation_from_05'] = self.cv_results_['mean_score'].apply(
                lambda x: abs(x - 0.5)
            )
            best_idx_by_auc = self.cv_results_['adjusted_score'].idxmax()
            best_idx_by_dev = self.cv_results_['deviation_from_05'].idxmax()
            
            if self.cv_results_.loc[best_idx_by_auc, 'std_score'] > 0.1:
                best_idx = best_idx_by_dev
            else:
                best_idx = best_idx_by_auc
            
            if self.disable_invert_predictions:
                self.invert_predictions_ = False
            elif self.force_invert_predictions is not None:
                self.invert_predictions_ = self.force_invert_predictions
            else:
                self.invert_predictions_ = self.cv_results_.loc[best_idx, 'mean_score'] < 0.5
        else:
            best_idx = self.cv_results_['mean_score'].idxmax()
            if self.force_invert_predictions is not None:
                self.invert_predictions_ = self.force_invert_predictions
            else:
                self.invert_predictions_ = False
        
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']

        print(f"Najlepsze C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f} ± {self.cv_results_.loc[best_idx, 'std_score']:.4f})")
        if self.invert_predictions_:
            print("Uwaga: Predykcje zostaną odwrócone (wykryto AUC < 0.5)")

        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.predict(X_val))
            else:
                self.val_score_ = roc_auc_score(y_val, self.predict_proba(X_val))
            print(f"Walidacja {self.opt_metric}: {self.val_score_:.4f}")

        return self

    def predict_proba(self, X):
        """Przewiduje prawdopodobieństwa klas."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        proba = self.model_.predict_proba(X)[:, 1]
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            proba = 1 - proba
        return proba

    def predict(self, X):
        """Przewiduje etykiety klas."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = self.model_.predict(X)
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            pred = 1 - pred
        return pred

    def get_feature_importance(self):
        """Pobiera ważność cech ze współczynników."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")

        coef = self.model_.named_steps['classifier'].coef_[0]
        
        # Odwracamy współczynniki, jeśli model przewiduje odwrotnie
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            coef = -coef

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df


def fisher_test_single(tcr_presence: np.ndarray, disease_status: np.ndarray) -> tuple:
    """Test dokładny Fishera dla pojedynczego TCR."""
    disease_pos = disease_status == 1
    disease_neg = disease_status == 0
    
    a = np.sum(tcr_presence & disease_pos)
    b = np.sum((~tcr_presence.astype(bool)) & disease_pos)
    c = np.sum(tcr_presence & disease_neg)
    d = np.sum((~tcr_presence.astype(bool)) & disease_neg)
    
    try:
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
    except:
        odds_ratio, p_value = np.nan, 1.0
    
    return (a, c, odds_ratio, p_value)


def calculate_burden_features(df, associated_tcrs, subject_col='subject_id', 
                             tcr_id_col='tcr_id', sequence_col='sequence_aa',
                             v_call_col='v_call', j_call_col='j_call'):
    """
    Oblicza cechy obciążenia fenotypowego dla każdego osobnika.
    
    Args:
        df: DataFrame z sekwencjami (polars lub pandas)
        associated_tcrs: DataFrame lub zbiór powiązanych ID TCR
        subject_col: Nazwa kolumny dla ID osobnika
        tcr_id_col: Nazwa kolumny dla ID TCR (v_call + sequence + j_call)
        sequence_col: Nazwa kolumny dla sekwencji CDR3
        v_call_col: Nazwa kolumny dla genu V
        j_call_col: Nazwa kolumny dla genu J
    
    Returns:
        DataFrame z cechami obciążenia na osobnika
    """
    if pl is not None and isinstance(df, pl.DataFrame):
        # Wersja Polars
        if isinstance(associated_tcrs, pl.DataFrame):
            associated_set = set(associated_tcrs[tcr_id_col].to_list())
        else:
            associated_set = associated_tcrs
        
        if tcr_id_col not in df.columns:
            df = df.with_columns([
                (pl.col(v_call_col) + '_' + pl.col(sequence_col) + '_' + 
                 pl.col(j_call_col)).alias(tcr_id_col)
            ])
        
        df = df.with_columns([
            pl.col(tcr_id_col).is_in(associated_set).alias('is_associated')
        ])
        
        features = df.group_by(subject_col).agg([
            pl.col(tcr_id_col).n_unique().cast(pl.Float64).alias('n'),
            pl.col('is_associated').sum().cast(pl.Float64).alias('k'),
            pl.col(sequence_col).str.len_chars().mean().cast(pl.Float64).alias('mean_cdr3_length'),
            pl.col(v_call_col).n_unique().cast(pl.Float64).alias('n_v_genes'),
            pl.col(j_call_col).n_unique().cast(pl.Float64).alias('n_j_genes')
        ])
        
        return features
    else:
        # Wersja Pandas
        if isinstance(associated_tcrs, pd.DataFrame):
            associated_set = set(associated_tcrs[tcr_id_col].tolist())
        else:
            associated_set = associated_tcrs
        
        if tcr_id_col not in df.columns:
            df[tcr_id_col] = df[v_call_col] + '_' + df[sequence_col] + '_' + df[j_call_col]
        
        df['is_associated'] = df[tcr_id_col].isin(associated_set)
        
        features = df.groupby(subject_col).agg({
            tcr_id_col: 'nunique',
            'is_associated': 'sum',
            sequence_col: lambda x: x.str.len().mean(),
            v_call_col: 'nunique',
            j_call_col: 'nunique'
        }).rename(columns={
            tcr_id_col: 'n',
            'is_associated': 'k',
            sequence_col: 'mean_cdr3_length',
            v_call_col: 'n_v_genes',
            j_call_col: 'n_j_genes'
        })
        
        return features


class ImmuneStatePredictor:
    """
    Szablon do przewidywania stanów immunologicznych z danych repertuaru TCR.
    """

    def __init__(self, n_jobs: int = -1, device: str = 'cpu', **kwargs):
        """
        Inicjalizuje predyktor.

        Args:
            n_jobs (int): Liczba rdzeni CPU do użycia w przetwarzaniu równoległym.
            device (str): Urządzenie do użycia w obliczeniach (np. 'cpu', 'cuda').
            **kwargs: Dodatkowe hiperparametry dla modelu.
        """
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Ostrzeżenie: Zażądano 'cuda', ale nie jest dostępna. Powrót do 'cpu'.")
            self.device = 'cpu'
        else:
            self.device = device
            
        # --- twój kod zaczyna się tutaj ---
        
        # Konfiguracja
        self.p_threshold = 1e-2
        self.min_subjects = 2
        self.min_cdr3_length = 8
        self.max_cdr3_length = 90
        self.min_reads = 2
        
        # Komponenty modelu
        self.model = None
        self.important_sequences_ = None
        self.associated_tcrs = None
        self.feature_names_ = None
        self.out_dir = kwargs.get('out_dir', None)
        self.train_dir = kwargs.get('train_dir', None)
        
        # --- twój kod kończy się tutaj ---

    def fit(self, train_dir_path: str, results_dir: str):
        """
        Trenuje model na dostarczonych danych treningowych.

        Args:
            train_dir_path (str): Ścieżka do katalogu z plikami TSV treningowymi.

        Returns:
            self: Dopasowana instancja predyktora.
        """

        # --- twój kod zaczyna się tutaj ---
        gc.collect()
        
        print("="*80)
        print("TRENING: Odkrywanie TCR związanych z chorobą (podejście Emerson et al.)")
        print("="*80)
        
        # Krok 1: Ładowanie i przetwarzanie danych z Polars
        print("\n[1/5] Ładowanie danych treningowych...")
        train_df_pl = self._load_repertoires_fast(train_dir_path)
        
        print(f"Załadowano {train_df_pl.height:,} sekwencji od {train_df_pl['subject_id'].n_unique()} osób")
        
        # # Krok 2: Filtrowanie sekwencji
        print("\n[2/5] Filtrowanie sekwencji...")
        train_df_pl = self._filter_sequences_vectorized(train_df_pl)
        gc.collect()
        
        # Krok 3: Ładowanie metadanych i znajdowanie TCR związanych z chorobą
        print("\n[3/5] Znajdowanie TCR związanych z chorobą...")
        metadata_path = os.path.join(train_dir_path, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path)
        
        # Zmiana nazw kolumn na nasz wewnętrzny format
        metadata_df = metadata_df.rename(columns={
            'repertoire_id': 'subject_id',
            'label_positive': 'disease_status'
        })
        
        metadata_pl = pl.from_pandas(metadata_df[['subject_id', 'disease_status']])
        
        # Znajdowanie powiązanych TCR za pomocą testu Fishera
        self.associated_tcrs = self._find_associated_tcrs_parallel(train_df_pl, metadata_pl)
        
        print(f"Znaleziono {len(self.associated_tcrs)} TCR związanych z chorobą")

        # Zapisanie wyników testu Fishera
        fisher_results_path = f"{results_dir}fisher_results_{os.path.basename(train_dir_path)}.parquet"
        self.associated_tcrs.write_parquet(fisher_results_path)
        print(f"Zapisano wyniki testu Fishera do {fisher_results_path}")
        
        # Krok 4: Obliczanie cech obciążenia fenotypowego
        print("\n[4/5] Obliczanie cech obciążenia fenotypowego...")
        burden_features = self._calculate_burden_features(train_df_pl)
        
        # Łączenie z metadanymi
        X_train_df = burden_features.join(
            metadata_pl.select(['subject_id', 'disease_status']), 
            on='subject_id', 
            how='left'
        ).to_pandas().set_index('subject_id')
        
        y_train = X_train_df['disease_status'].values
        X_train = X_train_df.drop('disease_status', axis=1)
        
        self.feature_names_ = X_train.columns.tolist()
        self.train_ids_ = X_train.index.tolist()
        
        print(f"Macierz cech: {X_train.shape}")
        print(f"Częstość występowania choroby: {y_train.mean():.2%}")
        
        print("\n[5/5] Trenowanie modelu...")
        self.model = BurdenClassifier(
            c_values=[10, 1, 0.1, 0.01],
            cv_folds=5,
            penalty='l2',  # lub 'l1' dla selekcji cech
            opt_metric='roc_auc',
            random_state=42,
            n_jobs=self.n_jobs
        )
        
        self.model.tune_and_fit(X_train, y_train, val_size=0.2)
        
        # Identyfikacja ważnych sekwencji
        print("\nIdentyfikacja ważnych sekwencji...")
        self.important_sequences_ = self.identify_associated_sequences(
            train_dir_path=train_dir_path, 
            top_k=50000
        )
        
        print("="*80)
        print("TRENING ZAKOŃCZONY")
        print("="*80)
        
        # --- twój kod kończy się tutaj ---
        
        return self

    def predict_proba(self, test_dir_path: str, results_dir: str) -> pd.DataFrame:
        """Przewiduje prawdopodobieństwa dla przykładów w podanej ścieżce."""
        print(f"Tworzenie predykcji dla danych w {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("Model nie został jeszcze dopasowany. Proszę najpierw wywołać `fit`.")
        
        # --- twój kod zaczyna się tutaj ---
        
        # Ładowanie i przetwarzanie danych testowych
        test_df_pl = self._load_repertoires_fast(test_dir_path)
        # test_df_pl = self._filter_sequences_vectorized(test_df_pl)
        
        # Tworzenie TCR ID
        test_df_pl = test_df_pl.with_columns([
            (pl.col('v_call') + '_' + pl.col('sequence_aa') + '_' + pl.col('j_call')).alias('tcr_id')
        ])
        
        # Oznaczanie powiązanych TCR
        if self.associated_tcrs is not None and len(self.associated_tcrs) > 0:
            associated_set = set(self.associated_tcrs['tcr_id'].to_list())
        else:
            associated_set = set()
        
        test_df_pl = test_df_pl.with_columns([
            pl.col('tcr_id').is_in(associated_set).alias('is_associated')
        ])
        
        # Obliczanie WSZYSTKICH cech (jak w treningu)
        burden_features = test_df_pl.group_by('subject_id').agg([
            pl.col('tcr_id').n_unique().cast(pl.Float64).alias('n'),
            pl.col('is_associated').sum().cast(pl.Float64).alias('k'),
            pl.col('sequence_aa').str.len_chars().mean().cast(pl.Float64).alias('mean_cdr3_length'),
            pl.col('v_call').n_unique().cast(pl.Float64).alias('n_v_genes'),
            pl.col('j_call').n_unique().cast(pl.Float64).alias('n_j_genes'),
            # (pl.col('is_associated').sum() / pl.col('tcr_id').n_unique()).cast(pl.Float64).alias('burden_ratio'),
        ])
        
        del test_df_pl
        gc.collect()
        
        X_test_df = burden_features.to_pandas().set_index('subject_id')
        # Zapisanie cech obciążenia testowego
        test_burden_path = f"{results_dir}/test_burden_features_{os.path.basename(test_dir_path)}.csv"
        X_test_df.to_csv(test_burden_path)
        print(f"Zapisano cechy obciążenia testowego do {test_burden_path}")
        
        # Zapewnienie tych samych cech co w treningu
        X_test = X_test_df[self.feature_names_]  # Używamy feature_names_
        repertoire_ids = X_test.index.tolist()
        
        # Przewidywanie
        if isinstance(self.model, dict):
            # Model beta-dwumianowy (tylko k, n)
            X_burden = X_test[['k', 'n']].values
            probabilities = self._predict_beta_binomial(X_burden)
        else:
            # BurdenClassifier (wszystkie cechy)
            probabilities = self.model.predict_proba(X_test)
        
        # --- twój kod kończy się tutaj ---
        
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 
                                        'junction_aa', 'v_call', 'j_call']]
        print(f"Predykcja zakończona na {len(repertoire_ids)} przykładach.")
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """Identyfikuje top "k" ważnych sekwencji z danych treningowych."""
        dataset_name = os.path.basename(train_dir_path)
    
        # --- twój kod zaczyna się tutaj ---
        
        if self.associated_tcrs is None or len(self.associated_tcrs) == 0:
            print("Ostrzeżenie: Nie znaleziono istotnych powiązanych TCR. Używanie najczęstszych w próbkach chorych+.")
            
            # Ładowanie danych i metadanych
            metadata_path = os.path.join(train_dir_path, 'metadata.csv')
            metadata = pd.read_csv(metadata_path)
            metadata = metadata.rename(columns={'repertoire_id': 'subject_id', 'label_positive': 'disease_status'})
            
            disease_pos_ids = set(metadata[metadata['disease_status'] == 1]['subject_id'])
            
            # Ładowanie sekwencji
            full_df = load_full_dataset(train_dir_path)
            full_df = full_df.rename(columns={'ID': 'subject_id'})
            
            # Filtrowanie tylko do chorych+ i liczenie
            disease_df = full_df[full_df['subject_id'].isin(disease_pos_ids)]
            
            # Liczenie częstości każdej unikalnej sekwencji
            seq_counts = disease_df.groupby(['junction_aa', 'v_call', 'j_call']).size().reset_index(name='count')
            top_sequences_df = seq_counts.nlargest(top_k, 'count')[['junction_aa', 'v_call', 'j_call']]
            
        else:
            # Używanie powiązanych TCR posortowanych według ilorazu szans
            top_tcrs = self.associated_tcrs.to_pandas().nlargest(
                min(top_k, len(self.associated_tcrs)), 
                'odds_ratio'
            )
            
            top_sequences_list = []
            for tcr_id in top_tcrs['tcr_id']:
                parts = tcr_id.split('_', 2)
                if len(parts) == 3:
                    v_call, cdr3, j_call = parts
                    top_sequences_list.append({
                        'junction_aa': cdr3,
                        'v_call': v_call,
                        'j_call': j_call
                    })
            
            top_sequences_df = pd.DataFrame(top_sequences_list)
        
        # --- twój kod kończy się tutaj ---
    
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = (top_sequences_df['dataset'] + '_seq_top_' + 
                                  top_sequences_df['ID'].astype(str))
        top_sequences_df['label_positive_probability'] = -999.0
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 
                                             'junction_aa', 'v_call', 'j_call']]
    
        return top_sequences_df
    
    # --- twój kod zaczyna się tutaj ---
    # Metody pomocnicze (mogą być dodane poza blokami szablonu)
    
    def _load_repertoires_fast(self, data_dir: str) -> pl.DataFrame:
        """Szybkie równoległe ładowanie z Polars"""
        metadata_path = os.path.join(data_dir, 'metadata.csv')

        # Trening: używanie metadanych
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            
            dfs = []
            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Ładowanie"):
                file_path = os.path.join(data_dir, row['filename'])
                try:
                    df = pl.read_csv(file_path, separator='\t')
                    
                    # Zmiana nazw kolumn na standardowy format
                    if 'junction_aa' in df.columns:
                        df = df.rename({'junction_aa': 'sequence_aa'})
                    
                    df = df.with_columns([
                        pl.lit(row['repertoire_id']).alias('subject_id')
                    ])
                    
                    # Wybór odpowiednich kolumn
                    cols_to_keep = ['subject_id', 'sequence_aa', 'v_call', 'j_call']
                    if 'templates' in df.columns:
                        cols_to_keep.append('templates')
                    
                    df = df.select([col for col in cols_to_keep if col in df.columns])
                    
                    # Stosowanie filtrowania
                    df = filter_sequences_polars(df, 
                                                min_length=self.min_cdr3_length,
                                                max_length=self.max_cdr3_length,
                                                min_reads=self.min_reads)
                    dfs.append(df)
                    
                except Exception as e:
                    print(f"Błąd ładowania {row['filename']}: {e}")
                    continue

        # Test: używanie glob (bez metadanych)
        else:
            dfs = []
            tsv_files = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
            for file_path in tqdm(tsv_files, desc="Ładowanie"):
                try:
                    df = pl.read_csv(file_path, separator='\t')
                    if 'junction_aa' in df.columns:
                        df = df.rename({'junction_aa': 'sequence_aa'})
                    subject_id = os.path.basename(file_path).replace('.tsv', '')
                    df = df.with_columns([pl.lit(subject_id).alias('subject_id')])
                    cols_to_keep = ['subject_id', 'sequence_aa', 'v_call', 'j_call']
                    if 'templates' in df.columns:
                        cols_to_keep.append('templates')
                    df = df.select([col for col in cols_to_keep if col in df.columns])
                    
                    # Stosowanie filtrowania
                    df = filter_sequences_polars(df,
                                                min_length=self.min_cdr3_length,
                                                max_length=self.max_cdr3_length,
                                                min_reads=self.min_reads)
                    dfs.append(df)
                except Exception as e:
                    print(f"Błąd ładowania {file_path}: {e}")
                    continue

        for df in dfs:
            # Upewnić się że templates - Float64 jeśli jest
            if 'templates' in df.columns:
                df = df.with_columns([
                    pl.col('templates').cast(pl.Float64)
                ])
        result_df = pl.concat(dfs, how='vertical_relaxed')
        del dfs
        gc.collect()
        return result_df
    
    def _filter_sequences_vectorized(self, df: pl.DataFrame) -> pl.DataFrame:
        """Wektoryzowane filtrowanie"""
        initial = df.height
        
        valid_aa_pattern = r'^[ACDEFGHIKLMNPQRSTVWY]+$'
        cdr3_pattern = r'^C.*[FW]$'
        
        df = df.filter(
            pl.col('sequence_aa').str.contains(valid_aa_pattern) &
            ~pl.col('sequence_aa').str.contains(r'\*') &
            pl.col('sequence_aa').str.contains(cdr3_pattern) &
            pl.col('sequence_aa').str.len_chars().is_between(
                self.min_cdr3_length, self.max_cdr3_length
            ) &
            pl.col('v_call').is_not_null() &
            pl.col('j_call').is_not_null()
        )
        
        return df
    
    def _find_associated_tcrs_parallel(self, df: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
        """Równoległy test Fishera oszczędzający pamięć"""
        
        gc.collect()
        
        # Tworzenie TCR ID TYLKO dla potrzebnych kolumn, usuwając resztę
        df = df.select(['v_call', 'sequence_aa', 'j_call', 'subject_id']).with_columns([
            (pl.col('v_call') + '_' + pl.col('sequence_aa') + '_' + pl.col('j_call')).alias('tcr_id')
        ])
        
        # Od razu usunąć oryginalne kolumny
        df = df.select(['tcr_id', 'subject_id'])
        gc.collect()
        
        # Najpierw policzyć które TCR występują >= min_subjects
        print('Liczenie częstości TCR...')
        tcr_freq = df.group_by('tcr_id').agg(
            pl.col('subject_id').n_unique().alias('n_subjects')
        ).filter(
            pl.col('n_subjects') >= self.min_subjects
        )
        
        frequent_tcrs = set(tcr_freq['tcr_id'].to_list())
        del tcr_freq
        gc.collect()
        print(f'{len(frequent_tcrs):,} TCR przechodzi filtr min_subjects')
        
        # Konwertować na DataFrame dla szybkiego join
        frequent_df = pl.DataFrame({'tcr_id': list(frequent_tcrs)})
        del frequent_tcrs
        gc.collect()
        
        print('Filtrowanie według częstych TCR (używając join)...')
        # Join zamiast is_in - ZNACZNIE szybsze
        df = df.join(frequent_df, on='tcr_id', how='inner')
        del frequent_df
        gc.collect()
        
        print(f'Przefiltrowano do {df.height:,} wierszy')
        
        # Teraz group_by na mniejszym DataFrame
        print('Grupowanie według tcr_id i subject_id...')
        tcr_counts = df.group_by(['tcr_id', 'subject_id']).agg(pl.count().alias('count'))
        
        del df
        gc.collect()
        
        # Pobranie listy TCR do testowania
        tcr_to_test = set(tcr_counts['tcr_id'].unique().to_list())  # POPRAWIONE
        print(f'{len(tcr_to_test):,} unikalnych TCR do testowania')
        gc.collect()
        
        print(f"Testowanie {len(tcr_to_test):,} TCR...")

        # Przeliczenie słownika PRZED partiami (raz zamiast N razy)
        print("Wstępne obliczanie mapowania TCR-osoba...")
        tcr_subject_map = {}
        for row in tqdm(tcr_counts.iter_rows(named=True), total=tcr_counts.height, desc="Budowanie mapy"):
            tcr_id = row['tcr_id']
            subject = row['subject_id']
            if tcr_id not in tcr_subject_map:
                tcr_subject_map[tcr_id] = set()
            tcr_subject_map[tcr_id].add(subject)
        
        del tcr_counts  # Już nie potrzebny
        gc.collect()
        
        tcr_list = list(tcr_to_test)
        batch_size = 50000  # Można zwiększyć
        all_results = []
        
        status_dict = dict(zip(metadata['subject_id'].to_list(), metadata['disease_status'].to_list()))
        all_subjects = sorted(status_dict.keys())
        disease_status_array = np.array([status_dict[s] for s in all_subjects])
        
        for i in range(0, len(tcr_list), batch_size):
            batch_tcrs = tcr_list[i:i+batch_size]
            
            # Pobranie z przeliczonego słownika (szybko!)
            batch_matrix = []
            batch_ids = []
            for tcr_id in batch_tcrs:
                presence = np.array([1 if s in tcr_subject_map.get(tcr_id, set()) else 0 for s in all_subjects])
                batch_matrix.append(presence)
                batch_ids.append(tcr_id)
            
            # Równoległy test Fishera
            test_func = partial(self._fisher_test_single, disease_status=disease_status_array)
            with Pool(self.n_jobs) as pool:
                results = pool.map(test_func, batch_matrix)
            
            for tcr_id, res in zip(batch_ids, results):
                all_results.append({'tcr_id': tcr_id, 'n_disease_pos': res[0], 
                                   'n_disease_neg': res[1], 'odds_ratio': res[2], 'p_value': res[3]})
            
            del batch_matrix, batch_ids, results
            gc.collect()
            
            print(f"Przetworzono {min(i+batch_size, len(tcr_list))}/{len(tcr_list)} TCR")
        
        del tcr_subject_map, tcr_list
        gc.collect()
        
        # Łączenie wyników
        results_df = pl.DataFrame(all_results)
        del all_results
        gc.collect()
        
        # Korekcja FDR
        p_values = results_df['p_value'].to_numpy()
        _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        results_df = results_df.with_columns([pl.Series('p_value_fdr', p_fdr)])

        results_df.sort('p_value').head(50000).to_pandas().to_csv(os.path.join(self.out_dir, f"{os.path.basename(self.train_dir)}_important_sequences.tsv"), sep='\t', index=False)
        
        # Filtrowanie według progu
        associated = results_df.filter(pl.col('p_value') < self.p_threshold).sort('p_value')
        
        # FALLBACK: jeśli puste, wziąć top-50000 według p-value
        if associated.height == 0:
            print(f"Ostrzeżenie: Żaden TCR nie przechodzi p < {self.p_threshold}. Używanie top 50000 według p-value.")
            associated = results_df.sort('p_value').head(50000)
        
        if associated.height > 0:
            print(f"Zakres FDR: {associated['p_value_fdr'].min():.4f} - {associated['p_value_fdr'].max():.4f}")
        
        return associated
    
    @staticmethod
    def _fisher_test_single(tcr_presence: np.ndarray, disease_status: np.ndarray) -> tuple:
        """Test Fishera dla pojedynczego TCR"""
        disease_pos = disease_status == 1
        disease_neg = disease_status == 0
        
        a = np.sum(tcr_presence & disease_pos)
        b = np.sum((~tcr_presence.astype(bool)) & disease_pos)
        c = np.sum(tcr_presence & disease_neg)
        d = np.sum((~tcr_presence.astype(bool)) & disease_neg)
        
        try:
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
        except:
            odds_ratio, p_value = np.nan, 1.0
        
        return (a, c, odds_ratio, p_value)
    
    def _calculate_burden_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Obliczanie obciążenia fenotypowego i dodatkowych cech"""
        
        if 'tcr_id' not in df.columns:
            df = df.with_columns([
                (pl.col('v_call') + '_' + pl.col('sequence_aa') + '_' + 
                 pl.col('j_call')).alias('tcr_id')
            ])
        
        # Tworzenie flagi powiązania
        associated_set = set(self.associated_tcrs['tcr_id'].to_list())
        df = df.with_columns([
            pl.col('tcr_id').is_in(associated_set).alias('is_associated')
        ])
        
        # Obliczanie cech
        features = df.group_by('subject_id').agg([
            pl.col('tcr_id').n_unique().cast(pl.Float64).alias('n'),
            pl.col('is_associated').sum().cast(pl.Float64).alias('k'),
            pl.col('sequence_aa').str.len_chars().mean().cast(pl.Float64).alias('mean_cdr3_length'),
            pl.col('v_call').n_unique().cast(pl.Float64).alias('n_v_genes'),
            pl.col('j_call').n_unique().cast(pl.Float64).alias('n_j_genes')
        ])
        # .with_columns([
        #     (pl.col('k') / pl.col('n')).cast(pl.Float64).alias('burden_ratio')
        # ])
        
        return features

    def _train_beta_binomial_model(self, X: pd.DataFrame, y: np.ndarray):
        """Trenowanie modelu beta-dwumianowego"""
        from scipy.special import betaln
        from scipy.optimize import minimize
        
        X_burden = X[['k', 'n']].values
        k, n = X_burden[:, 0], X_burden[:, 1]
        
        k_neg = k[y == 0]
        n_neg = n[y == 0]
        k_pos = k[y == 1]
        n_pos = n[y == 1]
        
        def neg_log_likelihood(params, k, n):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10
            ll = np.sum(betaln(k + alpha, n - k + beta) - betaln(alpha, beta))
            return -ll
        
        # Dopasowanie dla klasy negatywnej
        result_neg = minimize(
            neg_log_likelihood, x0=[1.0, 10.0], args=(k_neg, n_neg),
            method='L-BFGS-B', bounds=[(0.001, None), (0.001, None)]
        )
        
        # Dopasowanie dla klasy pozytywnej
        result_pos = minimize(
            neg_log_likelihood, x0=[1.0, 10.0], args=(k_pos, n_pos),
            method='L-BFGS-B', bounds=[(0.001, None), (0.001, None)]
        )
        
        # Priorytety
        prior_neg = (np.sum(y == 0) + 1) / (len(y) + 2)
        prior_pos = (np.sum(y == 1) + 1) / (len(y) + 2)
        
        model = {
            'params_neg': result_neg.x,
            'params_pos': result_pos.x,
            'prior_neg': prior_neg,
            'prior_pos': prior_pos
        }
        
        print(f"Parametry beta-dwumianowe: neg={model['params_neg']}, pos={model['params_pos']}")
        
        # Walidacja krzyżowa
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = []
        
        for train_idx, val_idx in skf.split(X_burden, y):
            X_tr, X_val = X_burden[train_idx], X_burden[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Trenowanie modelu dla foldu (uproszczone)
            proba_val = self._predict_beta_binomial_with_params(
                X_val, model['params_neg'], model['params_pos'], 
                model['prior_neg'], model['prior_pos']
            )
            
            auc = roc_auc_score(y_val, proba_val)
            cv_aucs.append(auc)
        
        print(f"CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
        
        return model
    
    def _predict_beta_binomial(self, X_burden: np.ndarray) -> np.ndarray:
        """Przewidywanie z modelem beta-dwumianowym"""
        return self._predict_beta_binomial_with_params(
            X_burden,
            self.model['params_neg'],
            self.model['params_pos'],
            self.model['prior_neg'],
            self.model['prior_pos']
        )
    
    @staticmethod
    def _predict_beta_binomial_with_params(
        X_burden: np.ndarray, 
        params_neg: np.ndarray,
        params_pos: np.ndarray,
        prior_neg: float,
        prior_pos: float
    ) -> np.ndarray:
        """Przewidywanie prawdopodobieństw"""
        from scipy.special import betaln
        
        k, n = X_burden[:, 0], X_burden[:, 1]
        alpha_0, beta_0 = params_neg
        alpha_1, beta_1 = params_pos
        
        log_odds = np.log(prior_pos / prior_neg)
        log_odds += betaln(k + alpha_1, n - k + beta_1) - betaln(alpha_1, beta_1)
        log_odds -= betaln(k + alpha_0, n - k + beta_0) - betaln(alpha_0, beta_0)
        
        proba_pos = 1 / (1 + np.exp(-log_odds))
        return proba_pos
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ImmuneStatePredictor':
        """Ładuje zapisany model z pliku."""
        metadata_path = filepath.replace('.pkl', '.json') if filepath.endswith('.pkl') else f"{filepath}.json"
        model_path = filepath if filepath.endswith('.pkl') else f"{filepath}.pkl"

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Plik metadanych nie znaleziony: {metadata_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Plik modelu nie znaleziony: {model_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Tworzenie instancji
        predictor = cls(
            n_jobs=metadata['n_jobs'],
            device=metadata['device']
        )

        # Ładowanie modelu
        predictor.model = joblib.load(model_path)
        predictor.feature_names_ = metadata['feature_names']
        
        # Ładowanie powiązanych TCR
        if 'associated_tcrs_path' in metadata:
            predictor.associated_tcrs = pl.read_parquet(metadata['associated_tcrs_path'])
        
        print(f"Model załadowany z {model_path}")
        return predictor

    def save_model(self, filepath: str) -> None:
        """Zapisuje wytrenowany model i wszystkie niezbędne dane do pliku."""
        if self.model is None:
            raise RuntimeError("Model nie jest wytrenowany. Proszę najpierw wywołać fit().")

        # Zapisywanie wytrenowanego modelu za pomocą joblib
        model_path = filepath if filepath.endswith('.pkl') else f"{filepath}.pkl"
        
        # Sprawdzamy typ modelu
        if isinstance(self.model, dict):
            # Model beta-dwumianowy (słownik)
            joblib.dump(self.model, model_path)
            model_type = 'beta_binomial'
            feature_names = self.feature_names_
            best_params = None
            val_score = None
        else:
            # KmerClassifier lub BurdenClassifier
            joblib.dump(self.model.model_, model_path)
            model_type = 'classifier'
            feature_names = getattr(self.model, 'feature_names_', None)
            best_params = getattr(self.model, 'best_C_', None)
            val_score = getattr(self.model, 'val_score_', None)
        
        print(f"Model zapisany do {model_path}")

        # Zapisanie metadanych
        metadata = {
            'model_type': model_type,
            'feature_names': feature_names,
            'best_params': best_params,
            'val_score': val_score,
            'n_jobs': self.n_jobs,
            'device': self.device,
        }
        
        # Zapisanie associated_tcrs (ważne dla predykcji!)
        if self.associated_tcrs is not None:
            tcrs_path = filepath.replace('.pkl', '_associated_tcrs.parquet')
            self.associated_tcrs.write_parquet(tcrs_path)
            metadata['associated_tcrs_path'] = tcrs_path
        
        # Konwersja typów numpy
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif type(obj).__module__ == 'numpy':
                return obj.item()
            return obj
        
        metadata = convert_to_native(metadata)

        metadata_path = filepath.replace('.pkl', '.json') if filepath.endswith('.pkl') else f"{filepath}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadane zapisane do {metadata_path}")
    # --- twój kod kończy się tutaj ---


def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """
    Łączy macierz cech z etykietami, zapewniając wyrównanie.

    Parametry:
        X_df: DataFrame z próbkami jako wierszami (indeks zawiera ID)
        labels_df: DataFrame z kolumną ID i kolumną etykiet
        id_col: Nazwa kolumny ID w labels_df
        label_col: Nazwa kolumny etykiet w labels_df

    Zwraca:
        X: Macierz cech wyrównana z etykietami
        y: Etykiety binarne
        common_ids: ID, które zostały zachowane
    """
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("Nie znaleziono wspólnych ID między macierzą cech a etykietami")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Wyrównano {len(common_ids)} próbek z etykietami")

    return X, y, common_ids

## Główny przepływ pracy, który używa twojej implementacji klasy ImmuneStatePredictor do trenowania, identyfikacji ważnych sekwencji i przewidywania etykiet testowych

class CatBoostKmerClassifier(BaseKmerClassifier):
    """Klasyfikator CatBoost dla danych liczby k-merów z obsługą AUC < 0.5."""
    
    def __init__(self, iterations=500, learning_rates=[0.01, 0.1, 0.3], depths=[3, 4, 5],
                 cv_folds=5, opt_metric='roc_auc', random_state=123, n_jobs=-1,
                 verbose=False, plot_cv=False, 
                 force_invert_predictions=None, disable_invert_predictions=False):
        super().__init__(cv_folds, opt_metric, random_state, n_jobs,
                        force_invert_predictions, disable_invert_predictions)
        self.iterations = iterations
        self.learning_rates = learning_rates
        self.depths = depths
        self.verbose = verbose
        self.plot_cv = plot_cv
        self.best_depth_ = None
        self.best_learning_rate_ = None
        
        try:
            from catboost import CatBoostClassifier, Pool, cv
            self.CatBoostClassifier = CatBoostClassifier
            self.Pool = Pool
            self.cv = cv
            self.catboost_available = True
        except ImportError:
            self.catboost_available = False
            raise ImportError("CatBoost nie jest zainstalowany. Zainstaluj go za pomocą: pip install catboost")
    
    def _make_model(self, depth=None, learning_rate=None, use_best_model=False):
        """Tworzy model CatBoost."""
        if depth is None:
            depth = self.depths[0]
        if learning_rate is None:
            learning_rate = self.learning_rates[0]
        
        model = self.CatBoostClassifier(
            iterations=self.iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='Logloss',
            random_seed=self.random_state,
            thread_count=self.n_jobs,
            verbose=self.verbose,
            use_best_model=use_best_model,
            auto_class_weights='Balanced'
        )
        return model
    
    def tune_and_fit(self, X, y, val_size=0.2, selected_features=None):
        """Wykonuje strojenie CV i dopasowanie."""
        if not self.catboost_available:
            raise ValueError("CatBoost nie jest dostępny")
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Filtruj cechy, jeśli określono
        if selected_features is not None and self.feature_names_ is not None:
            feature_indices = [i for i, name in enumerate(self.feature_names_) if name in selected_features]
            X = X[:, feature_indices]
            self.feature_names_ = [name for name in self.feature_names_ if name in selected_features]
            print(f"Trening na {len(self.feature_names_)} wybranych cechach")
        
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y, shuffle=True)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        train_pool = self.Pool(X_train, y_train)
        scorer = self._get_scorer()
        
        # Przeszukiwanie siatki
        depths = self.depths
        learning_rates = self.learning_rates
        
        results = []
        for depth in depths:
            for lr in learning_rates:
                model = self._make_model(depth=depth, learning_rate=lr)
                cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, 
                                          scoring=scorer, n_jobs=self.n_jobs)
                results.append({
                    'depth': depth,
                    'learning_rate': lr,
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'scores': cv_scores
                })
        
        self.cv_results_ = pd.DataFrame(results)
        best_idx = self._handle_auc_inversion(
            force_invert=self.force_invert_predictions,
            disable_invert=self.disable_invert_predictions
        )
        
        self.best_depth_ = self.cv_results_.loc[best_idx, 'depth']
        self.best_learning_rate_ = self.cv_results_.loc[best_idx, 'learning_rate']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']
        
        print(f"Najlepsza głębokość: {self.best_depth_}, współczynnik uczenia: {self.best_learning_rate_} "
              f"(CV {self.opt_metric}: {self.best_score_:.4f} ± {self.cv_results_.loc[best_idx, 'std_score']:.4f})")
        if self.invert_predictions_:
            print("Uwaga: Predykcje zostaną odwrócone (wykryto AUC < 0.5)")
        
        use_best = (X_val is not None and len(X_val) > 0)
        self.model_ = self._make_model(depth=self.best_depth_, 
                                      learning_rate=self.best_learning_rate_,
                                      use_best_model=use_best)
        
        if use_best:
            val_pool = self.Pool(X_val, y_val)
            self.model_.fit(train_pool, eval_set=val_pool, verbose=self.verbose)
        else:
            self.model_.fit(train_pool, verbose=self.verbose)
        
        if self.feature_names_ is not None:
            self.X_train_ = pd.DataFrame(X_train, columns=self.feature_names_)
        else:
            self.X_train_ = pd.DataFrame(X_train)
        
        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.predict(X_val))
            else:
                self.val_score_ = roc_auc_score(y_val, self.predict_proba(X_val))
            # Wyświetlamy wynik walidacji i std z CV dla najlepszego modelu
            best_std = self.cv_results_.loc[best_idx, 'std_score']
            print(f"Walidacja {self.opt_metric}: {self.val_score_:.4f}")
            
            # Obliczamy MCC
            y_val_pred = self.predict(X_val)
            self.mcc_score_ = matthews_corrcoef(y_val, y_val_pred)
            print(f"Walidacja MCC: {self.mcc_score_:.4f}")
        
        # Zapisujemy metryki CV
        if self.opt_metric == 'roc_auc':
            self.cv_roc_auc_ = self.best_score_
            self.cv_roc_auc_std_ = self.cv_results_.loc[best_idx, 'std_score']
        
        return self
    
    def _predict_proba_internal(self, X):
        """Wewnętrzna metoda do uzyskania surowych prawdopodobieństw."""
        return self.model_.predict_proba(X)[:, 1]
    
    def _predict_internal(self, X):
        """Wewnętrzna metoda do uzyskania surowych predykcji."""
        return self.model_.predict(X)
    
    def get_feature_importance(self, sample_size=10000, use_shap=True):
        """Uzyskaj ważność cech za pomocą SHAP lub ważności cech CatBoost."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")
        
        if use_shap and shap is not None and self.X_train_ is not None:
            try:
                sample_size_actual = min(sample_size, len(self.X_train_))
                X_sample = self.X_train_.iloc[:sample_size_actual]
                
                explainer = shap.TreeExplainer(self.model_)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                shap_importance = shap_values.mean(axis=0)
                
                if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
                    shap_importance = -shap_importance
                
                if self.feature_names_ is not None:
                    feature_names = self.feature_names_
                else:
                    feature_names = [f"feature_{i}" for i in range(len(shap_importance))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': shap_importance,
                    'abs_coefficient': np.abs(shap_importance)
                })
                
                importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
                return importance_df
                
            except Exception as e:
                print(f"Ostrzeżenie: Nie udało się obliczyć wartości SHAP: {e}. Powrót do ważności cech.")
        
        # Zapasowe: ważność cech CatBoost
        feature_importance = self.model_.get_feature_importance()
        
        if hasattr(self, 'invert_predictions_') and self.invert_predictions_:
            feature_importance = -feature_importance
        
        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': feature_importance,
            'abs_coefficient': np.abs(feature_importance)
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df
    
    def score_all_sequences(self, sequences_df, sequence_col='junction_aa', 
                           return_kmer_importance=False, use_shap=True, shap_sample_size=10000):
        """Oceniaj wszystkie sekwencje za pomocą ważności cech."""
        if self.model_ is None:
            raise ValueError("Model nie został dopasowany.")
        
        importance_df = self.get_feature_importance(use_shap=use_shap, sample_size=shap_sample_size)
        kmer_importance_dict = dict(zip(importance_df['feature'], importance_df['coefficient']))
        
        kmer_set = set(kmer_importance_dict.keys())
        k = len(self.feature_names_[0]) if self.feature_names_ else 3
        
        scores = []
        for seq in tqdm(sequences_df[sequence_col], desc="Ocenianie sekwencji"):
            score = 0.0
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                if kmer in kmer_set:
                    score += kmer_importance_dict[kmer]
            scores.append(score)
        
        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        
        if return_kmer_importance:
            kmer_importance_df = pd.DataFrame({
                "kmer": list(kmer_importance_dict.keys()),
                "importance": list(kmer_importance_dict.values())
            })
            return result_df, kmer_importance_df
        else:
            return result_df


class BaseImmuneStatePredictor:
    """Klasa bazowa dla predyktorów stanu immunologicznego ze wspólną funkcjonalnością."""
    
    def __init__(self, n_jobs=-1, device='cpu', **kwargs):
        import os
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        self.train_ids_ = None
        self.important_sequences_ = None
        self.out_dir = kwargs.get('out_dir', None)
        self.train_dir = kwargs.get('train_dir', None)
    
    def fit(self, train_dir_path: str, **kwargs):
        """Trenuj model. Do nadpisania przez podklasy."""
        raise NotImplementedError
    
    def predict_proba(self, test_dir_path: str, **kwargs):
        """Przewiduj prawdopodobieństwa. Do nadpisania przez podklasy."""
        raise NotImplementedError
    
    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000):
        """Zidentyfikuj ważne sekwencje. Do nadpisania przez podklasy."""
        raise NotImplementedError


class KmerImmuneStatePredictor(BaseImmuneStatePredictor):
    """Predyktor dla modeli opartych na k-merach."""
    
    def __init__(self, k=3, use_templates=True, include_vj_features=False,
                 model_type='logistic_regression', importance_method='shap', iterations=500,
                 min_count=1, c_values=None, learning_rates=None, depths=None, cv_folds=5, opt_metric='roc_auc',
                 random_state=123, n_jobs=-1, device='cpu',
                 force_invert_predictions=None, disable_invert_predictions=False, **kwargs):
        super().__init__(n_jobs, device, **kwargs)
        self.k = k
        self.use_templates = use_templates
        self.include_vj_features = include_vj_features
        self.model_type = model_type
        self.importance_method = importance_method
        self.iterations = iterations
        self.min_count = min_count
        self.c_values = c_values or [1, 0.1, 0.05, 0.03]
        self.learning_rates = learning_rates or [0.01, 0.1, 0.3]
        self.depths = depths or [3, 4, 5]
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.force_invert_predictions = force_invert_predictions
        self.disable_invert_predictions = disable_invert_predictions
        self.model = None
        self.feature_names_ = None
    
    def fit(self, train_dir_path: str, results_dir: str = None, selected_features=None):
        """Trenuj model."""
        gc.collect()
        
        print(f"Ładowanie i kodowanie {self.k}-merów...")
        X_train_df, y_train_df = load_and_encode_kmers(
            train_dir_path,
            k=self.k,
            use_templates=self.use_templates,
            include_vj_features=self.include_vj_features,
            min_count=self.min_count
        )
        
        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df)
        self.feature_names_ = X_train.columns.tolist()
        self.train_ids_ = train_ids
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegressionKmerClassifier(
                c_values=self.c_values,
                cv_folds=self.cv_folds,
                opt_metric=self.opt_metric,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                force_invert_predictions=self.force_invert_predictions,
                disable_invert_predictions=self.disable_invert_predictions
            )
            self.model.tune_and_fit(X_train, y_train, val_size=0.2, selected_features=selected_features)
        elif self.model_type == 'catboost':
            try:
                self.model = CatBoostKmerClassifier(
                    iterations=self.iterations,
                    learning_rates=self.learning_rates,
                    depths=self.depths,
                    cv_folds=self.cv_folds,
                    opt_metric=self.opt_metric,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    force_invert_predictions=self.force_invert_predictions,
                    disable_invert_predictions=self.disable_invert_predictions
                )
                self.model.tune_and_fit(X_train, y_train, val_size=0.2, selected_features=selected_features)
            except ImportError:
                raise ImportError("CatBoost nie jest zainstalowany. Zainstaluj go za pomocą: pip install catboost")
        else:
            raise ValueError(f"Nieznany typ modelu: {self.model_type}")
        
        print("Identyfikacja ważnych sekwencji...")
        self.important_sequences_ = self.identify_associated_sequences(train_dir_path, top_k=50000)
        
        return self
    
    def predict_proba(self, test_dir_path: str, results_dir: str = None):
        """Przewiduj prawdopodobieństwa."""
        if self.model is None:
            raise RuntimeError("Model nie został dopasowany. Najpierw wywołaj fit().")
        
        X_test_df, _ = load_and_encode_kmers(
            test_dir_path,
            k=self.k,
            use_templates=self.use_templates,
            include_vj_features=self.include_vj_features,
            min_count=self.min_count
        )
        
        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)
        
        repertoire_ids = X_test_df.index.tolist()
        probabilities = self.model.predict_proba(X_test_df)
        
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability',
                                        'junction_aa', 'v_call', 'j_call']]
        
        print(f"Predykcja zakończona na {len(repertoire_ids)} przykładach.")
        return predictions_df
    
    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000):
        """Zidentyfikuj ważne sekwencje."""
        dataset_name = os.path.basename(train_dir_path)
        
        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        
        use_shap = (self.importance_method == 'shap')
        all_sequences_scored = self.model.score_all_sequences(
            unique_seqs,
            sequence_col='junction_aa',
            return_kmer_importance=False,
            use_shap=use_shap
        )
        
        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']].copy()
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df) + 1)
        top_sequences_df['ID'] = (top_sequences_df['dataset'] + '_seq_top_' +
                                 top_sequences_df['ID'].astype(str))
        top_sequences_df['label_positive_probability'] = -999.0
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability',
                                            'junction_aa', 'v_call', 'j_call']]
        
        return top_sequences_df


# Alias dla kompatybilności wstecznej
# Stary ImmuneStatePredictor pozostaje bez zmian, ale zaleca się używanie
# KmerImmuneStatePredictor lub FisherBurdenImmuneStatePredictor


class FisherBurdenImmuneStatePredictor(BaseImmuneStatePredictor):
    """Predyktor dla modelu test Fishera + cechy obciążenia."""
    
    def __init__(self, p_threshold=1e-2, min_subjects=2, min_cdr3_length=8,
                 max_cdr3_length=90, min_reads=2, c_values=None, cv_folds=5,
                 penalty='l2', opt_metric='roc_auc', random_state=42, n_jobs=-1,
                 device='cpu', force_invert_predictions=None, disable_invert_predictions=False, **kwargs):
        super().__init__(n_jobs, device, **kwargs)
        self.p_threshold = p_threshold
        self.min_subjects = min_subjects
        self.min_cdr3_length = min_cdr3_length
        self.max_cdr3_length = max_cdr3_length
        self.min_reads = min_reads
        self.c_values = c_values or [10, 1, 0.1, 0.01]
        self.cv_folds = cv_folds
        self.penalty = penalty
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.force_invert_predictions = force_invert_predictions
        self.disable_invert_predictions = disable_invert_predictions
        self.model = None
        self.important_sequences_ = None
        self.associated_tcrs = None
        self.feature_names_ = None
    
    def fit(self, train_dir_path: str, results_dir: str = None, val_fraction: float = 0.2):
        """Trenuj model. Walidacja na osobnym podzbiorze repertoirów (bez przecieku: Fisher i cechy tylko na train)."""
        if pl is None:
            raise ValueError("Polars jest wymagany dla modelu testu Fishera")
        
        gc.collect()
        
        print("="*80)
        print("TRENING: Odkrywanie TCR związanych z chorobą (podejście Emerson et al.)")
        print("="*80)
        
        metadata_path = os.path.join(train_dir_path, 'metadata.csv')
        metadata_full = pd.read_csv(metadata_path)
        if val_fraction > 0 and val_fraction < 1 and metadata_full['label_positive'].nunique() > 1:
            train_meta, val_meta = train_test_split(
                metadata_full, test_size=val_fraction, random_state=self.random_state,
                stratify=metadata_full['label_positive']
            )
            train_meta = train_meta.reset_index(drop=True)
            val_meta = val_meta.reset_index(drop=True)
            print(f"\nPodział: {len(train_meta)} repertoirów treningowych, {len(val_meta)} walidacyjnych (bez przecieku).")
        else:
            train_meta = metadata_full
            val_meta = None
        
        print("\n[1/5] Ładowanie danych treningowych (tylko train)...")
        train_df_pl = self._load_repertoires_fast(train_dir_path, metadata_df=train_meta)
        print(f"Załadowano {train_df_pl.height:,} sekwencji")
        
        print("\n[2/5] Znajdowanie TCR związanych z chorobą (tylko na train)...")
        train_meta_renamed = train_meta.rename(columns={
            'repertoire_id': 'subject_id',
            'label_positive': 'disease_status'
        })
        metadata_pl = pl.from_pandas(train_meta_renamed[['subject_id', 'disease_status']])
        
        self.associated_tcrs = self._find_associated_tcrs_parallel(train_df_pl, metadata_pl)
        print(f"Znaleziono {len(self.associated_tcrs)} TCR związanych z chorobą")
        
        if results_dir:
            fisher_results_path = os.path.join(results_dir, f"fisher_results_{os.path.basename(train_dir_path)}.parquet")
            self.associated_tcrs.write_parquet(fisher_results_path)
            print(f"Zapisano wyniki testu Fishera do {fisher_results_path}")
        
        print("\n[3/5] Obliczanie cech obciążenia (train)...")
        burden_features = calculate_burden_features(
            train_df_pl, self.associated_tcrs,
            subject_col='subject_id',
            tcr_id_col='tcr_id',
            sequence_col='sequence_aa',
            v_call_col='v_call',
            j_call_col='j_call'
        )
        
        X_train_df = burden_features.join(
            metadata_pl.select(['subject_id', 'disease_status']),
            on='subject_id',
            how='left'
        ).to_pandas().set_index('subject_id')
        
        y_train = X_train_df['disease_status'].values
        X_train = X_train_df.drop('disease_status', axis=1)
        
        self.feature_names_ = X_train.columns.tolist()
        self.train_ids_ = X_train.index.tolist()
        
        print(f"Macierz cech train: {X_train.shape}")
        print(f"Częstość występowania choroby (train): {y_train.mean():.2%}")
        
        X_val, y_val = None, None
        if val_meta is not None and len(val_meta) > 0:
            print("Ładowanie danych walidacyjnych (cechy z TCR wyznaczonych tylko na train)...")
            val_df_pl = self._load_repertoires_fast(train_dir_path, metadata_df=val_meta)
            burden_val = calculate_burden_features(
                val_df_pl, self.associated_tcrs,
                subject_col='subject_id',
                tcr_id_col='tcr_id',
                sequence_col='sequence_aa',
                v_call_col='v_call',
                j_call_col='j_call'
            )
            val_meta_renamed = val_meta.rename(columns={
                'repertoire_id': 'subject_id',
                'label_positive': 'disease_status'
            })
            X_val_df = burden_val.join(
                pl.from_pandas(val_meta_renamed[['subject_id', 'disease_status']]),
                on='subject_id', how='left'
            ).to_pandas().set_index('subject_id')
            y_val = X_val_df['disease_status'].values
            X_val = X_val_df.drop('disease_status', axis=1)
            X_val = X_val.reindex(columns=self.feature_names_, fill_value=0)
            del val_df_pl, burden_val, burden_features
            gc.collect()
        
        print("\n[4/5] Trenowanie modelu...")
        self.model = BurdenClassifier(
            c_values=self.c_values,
            cv_folds=self.cv_folds,
            penalty=self.penalty,
            opt_metric=self.opt_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            force_invert_predictions=self.force_invert_predictions,
            disable_invert_predictions=self.disable_invert_predictions
        )
        if X_val is not None and y_val is not None:
            self.model.tune_and_fit(X_train, y_train, val_size=0, X_val=X_val, y_val=y_val)
        else:
            self.model.tune_and_fit(X_train, y_train, val_size=0.2)
        
        print("\n[5/5] Identyfikacja ważnych sekwencji...")
        self.important_sequences_ = self.identify_associated_sequences(train_dir_path, top_k=50000)
        
        print("="*80)
        print("TRENING ZAKOŃCZONY")
        print("="*80)
        
        return self
    
    def predict_proba(self, test_dir_path: str, results_dir: str = None):
        """Przewiduj prawdopodobieństwa."""
        if pl is None:
            raise ValueError("Polars jest wymagany")
        
        if self.model is None:
            raise RuntimeError("Model nie został dopasowany. Najpierw wywołaj fit().")
        
        print(f"Tworzenie predykcji dla danych w {test_dir_path}...")
        
        test_df_pl = self._load_repertoires_fast(test_dir_path)
        test_df_pl = test_df_pl.with_columns([
            (pl.col('v_call') + '_' + pl.col('sequence_aa') + '_' + pl.col('j_call')).alias('tcr_id')
        ])
        
        if self.associated_tcrs is not None and len(self.associated_tcrs) > 0:
            associated_set = set(self.associated_tcrs['tcr_id'].to_list())
        else:
            associated_set = set()
        
        test_df_pl = test_df_pl.with_columns([
            pl.col('tcr_id').is_in(associated_set).alias('is_associated')
        ])
        
        burden_features = calculate_burden_features(
            test_df_pl, associated_set,
            subject_col='subject_id',
            tcr_id_col='tcr_id',
            sequence_col='sequence_aa',
            v_call_col='v_call',
            j_call_col='j_call'
        )
        
        X_test_df = burden_features.to_pandas().set_index('subject_id')
        X_test = X_test_df[self.feature_names_]
        repertoire_ids = X_test.index.tolist()
        probabilities = self.model.predict_proba(X_test)
        
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability',
                                        'junction_aa', 'v_call', 'j_call']]
        
        print(f"Predykcja zakończona na {len(repertoire_ids)} przykładach.")
        return predictions_df
    
    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000):
        """Zidentyfikuj ważne sekwencje."""
        dataset_name = os.path.basename(train_dir_path)
        
        if self.associated_tcrs is None or len(self.associated_tcrs) == 0:
            print("Ostrzeżenie: Nie znaleziono istotnych związanych TCR. Używanie najczęstszych w próbkach choroba+.")
            
            metadata_path = os.path.join(train_dir_path, 'metadata.csv')
            metadata = pd.read_csv(metadata_path)
            metadata = metadata.rename(columns={'repertoire_id': 'subject_id', 'label_positive': 'disease_status'})
            
            disease_pos_ids = set(metadata[metadata['disease_status'] == 1]['subject_id'])
            full_df = load_full_dataset(train_dir_path)
            full_df = full_df.rename(columns={'ID': 'subject_id'})
            disease_df = full_df[full_df['subject_id'].isin(disease_pos_ids)]
            seq_counts = disease_df.groupby(['junction_aa', 'v_call', 'j_call']).size().reset_index(name='count')
            top_sequences_df = seq_counts.nlargest(top_k, 'count')[['junction_aa', 'v_call', 'j_call']]
        else:
            top_tcrs = self.associated_tcrs.to_pandas().nlargest(
                min(top_k, len(self.associated_tcrs)),
                'odds_ratio'
            )
            
            top_sequences_list = []
            for tcr_id in top_tcrs['tcr_id']:
                parts = tcr_id.split('_', 2)
                if len(parts) == 3:
                    v_call, cdr3, j_call = parts
                    top_sequences_list.append({
                        'junction_aa': cdr3,
                        'v_call': v_call,
                        'j_call': j_call
                    })
            
            top_sequences_df = pd.DataFrame(top_sequences_list)
        
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df) + 1)
        top_sequences_df['ID'] = (top_sequences_df['dataset'] + '_seq_top_' +
                                 top_sequences_df['ID'].astype(str))
        top_sequences_df['label_positive_probability'] = -999.0
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability',
                                            'junction_aa', 'v_call', 'j_call']]
        
        return top_sequences_df
    
    def _load_repertoires_fast(self, data_dir: str, metadata_df=None):
        """Szybkie ładowanie z Polars. Jeśli metadata_df jest podane, ładuje tylko te repertoiry (do poprawnej walidacji bez przecieku)."""
        if pl is None:
            raise ValueError("Polars jest wymagany")
        
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        dfs = []
        
        if metadata_df is not None:
            meta_to_use = metadata_df
        elif os.path.exists(metadata_path):
            meta_to_use = pd.read_csv(metadata_path)
        else:
            meta_to_use = None
        
        if meta_to_use is not None:
            for _, row in tqdm(meta_to_use.iterrows(), total=len(meta_to_use), desc="Ładowanie"):
                file_path = os.path.join(data_dir, row['filename'])
                try:
                    df = pl.read_csv(file_path, separator='\t')
                    if 'junction_aa' in df.columns:
                        df = df.rename({'junction_aa': 'sequence_aa'})
                    df = df.with_columns([
                        pl.lit(row['repertoire_id']).alias('subject_id')
                    ])
                    cols_to_keep = ['subject_id', 'sequence_aa', 'v_call', 'j_call']
                    if 'templates' in df.columns:
                        cols_to_keep.append('templates')
                    df = df.select([col for col in cols_to_keep if col in df.columns])
                    
                    # Stosujemy filtrowanie
                    df = filter_sequences_polars(df,
                                                min_length=self.min_cdr3_length,
                                                max_length=self.max_cdr3_length,
                                                min_reads=self.min_reads)
                    dfs.append(df)
                except Exception as e:
                    print(f"Błąd ładowania {row['filename']}: {e}")
                    continue
        else:
            tsv_files = sorted(glob.glob(os.path.join(data_dir, '*.tsv')))
            for file_path in tqdm(tsv_files, desc="Ładowanie"):
                try:
                    df = pl.read_csv(file_path, separator='\t')
                    if 'junction_aa' in df.columns:
                        df = df.rename({'junction_aa': 'sequence_aa'})
                    subject_id = os.path.basename(file_path).replace('.tsv', '')
                    df = df.with_columns([pl.lit(subject_id).alias('subject_id')])
                    cols_to_keep = ['subject_id', 'sequence_aa', 'v_call', 'j_call']
                    if 'templates' in df.columns:
                        cols_to_keep.append('templates')
                    df = df.select([col for col in cols_to_keep if col in df.columns])
                    
                    # Stosujemy filtrowanie
                    df = filter_sequences_polars(df,
                                                min_length=self.min_cdr3_length,
                                                max_length=self.max_cdr3_length,
                                                min_reads=self.min_reads)
                    dfs.append(df)
                except Exception as e:
                    print(f"Błąd ładowania {file_path}: {e}")
                    continue
        
        result_df = pl.concat(dfs, how='vertical_relaxed')
        del dfs
        gc.collect()
        return result_df
    
    def _find_associated_tcrs_parallel(self, df, metadata):
        """Znajdź związane TCR za pomocą testu Fishera."""
        if pl is None:
            raise ValueError("Polars jest wymagany")
        
        from multiprocessing import Pool
        from functools import partial
        
        df = df.select(['v_call', 'sequence_aa', 'j_call', 'subject_id']).with_columns([
            (pl.col('v_call') + '_' + pl.col('sequence_aa') + '_' + pl.col('j_call')).alias('tcr_id')
        ])
        df = df.select(['tcr_id', 'subject_id'])
        gc.collect()
        
        print('Liczenie częstości TCR...')
        tcr_freq = df.group_by('tcr_id').agg(
            pl.col('subject_id').n_unique().alias('n_subjects')
        ).filter(
            pl.col('n_subjects') >= self.min_subjects
        )
        
        frequent_tcrs = set(tcr_freq['tcr_id'].to_list())
        del tcr_freq
        gc.collect()
        print(f'{len(frequent_tcrs):,} TCR przechodzi filtr min_subjects')
        
        frequent_df = pl.DataFrame({'tcr_id': list(frequent_tcrs)})
        del frequent_tcrs
        gc.collect()
        
        df = df.join(frequent_df, on='tcr_id', how='inner')
        del frequent_df
        gc.collect()
        
        tcr_counts = df.group_by(['tcr_id', 'subject_id']).agg(pl.count().alias('count'))
        del df
        gc.collect()
        
        tcr_to_test = set(tcr_counts['tcr_id'].unique().to_list())
        print(f'{len(tcr_to_test):,} unikalnych TCR do testowania')
        gc.collect()
        
        print("Wstępne obliczanie mapowania TCR-podmiot...")
        tcr_subject_map = {}
        for row in tqdm(tcr_counts.iter_rows(named=True), total=tcr_counts.height, desc="Budowanie mapy"):
            tcr_id = row['tcr_id']
            subject = row['subject_id']
            if tcr_id not in tcr_subject_map:
                tcr_subject_map[tcr_id] = set()
            tcr_subject_map[tcr_id].add(subject)
        
        del tcr_counts
        gc.collect()
        
        tcr_list = list(tcr_to_test)
        batch_size = 50000
        all_results = []
        
        status_dict = dict(zip(metadata['subject_id'].to_list(), metadata['disease_status'].to_list()))
        all_subjects = sorted(status_dict.keys())
        disease_status_array = np.array([status_dict[s] for s in all_subjects])
        
        for i in range(0, len(tcr_list), batch_size):
            batch_tcrs = tcr_list[i:i+batch_size]
            
            batch_matrix = []
            batch_ids = []
            for tcr_id in batch_tcrs:
                presence = np.array([1 if s in tcr_subject_map.get(tcr_id, set()) else 0 for s in all_subjects])
                batch_matrix.append(presence)
                batch_ids.append(tcr_id)
            
            test_func = partial(fisher_test_single, disease_status=disease_status_array)
            with Pool(self.n_jobs) as pool:
                results = pool.map(test_func, batch_matrix)
            
            for tcr_id, res in zip(batch_ids, results):
                all_results.append({
                    'tcr_id': tcr_id,
                    'n_disease_pos': res[0],
                    'n_disease_neg': res[1],
                    'odds_ratio': res[2],
                    'p_value': res[3]
                })
            
            del batch_matrix, batch_ids, results
            gc.collect()
            
            print(f"Przetworzono {min(i+batch_size, len(tcr_list))}/{len(tcr_list)} TCR")
        
        del tcr_subject_map, tcr_list
        gc.collect()
        
        results_df = pl.DataFrame(all_results)
        del all_results
        gc.collect()
        
        p_values = results_df['p_value'].to_numpy()
        _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        results_df = results_df.with_columns([pl.Series('p_value_fdr', p_fdr)])
        
        associated = results_df.filter(pl.col('p_value') < self.p_threshold).sort('p_value')
        
        if associated.height == 0:
            print(f"Ostrzeżenie: Żaden TCR nie przechodzi p < {self.p_threshold}. Używanie top 50000 według wartości p.")
            associated = results_df.sort('p_value').head(50000)
        
        return associated


def compute_repertoire_features(metadata_df, data_dir):
    """
    Oblicza cechy na poziomie repertuaru dla każdej próbki.
    
    Parametry:
    -----------
    metadata_df : pd.DataFrame
        DataFrame z metadanymi, zawierający kolumny 'repertoire_id', 'filename', 'label_positive'
    data_dir : Path
        Ścieżka do katalogu z plikami TCR (.tsv)
    
    Zwraca:
    -----------
    pd.DataFrame
        DataFrame z cechami na poziomie repertuaru dla każdej próbki
    """
    features_list = []
    
    for _, row in metadata_df.iterrows():
        file_path = data_dir / row['filename']
        df = pd.read_csv(file_path, sep='\t')
        
        # Podstawowe cechy: identyfikatory i etykiety
        features = {
            'repertoire_id': row['repertoire_id'],
            'dataset': row.get('dataset', data_dir.name),
            'label_positive': row['label_positive'],
        }
        
        # Cechy oparte na sekwencjach
        features['n_sequences'] = len(df)
        features['n_unique_junctions'] = df['junction_aa'].nunique()
        
        # Statystyki długości junction_aa
        junction_lengths = df['junction_aa'].str.len()
        features['mean_junction_length'] = junction_lengths.mean()
        features['std_junction_length'] = junction_lengths.std()
        features['min_junction_length'] = junction_lengths.min()
        features['max_junction_length'] = junction_lengths.max()
        features['median_junction_length'] = junction_lengths.median()
        
        # Różnorodność genów
        features['n_unique_v_genes'] = df['v_call'].nunique()
        features['n_unique_j_genes'] = df['j_call'].nunique()
        
        # Geny D (opcjonalnie, nie zawsze obecne)
        if 'd_call' in df.columns:
            features['n_unique_d_genes'] = df['d_call'].nunique()
            features['d_call_known_ratio'] = (df['d_call'] != 'unknown').sum() / len(df)
        
        # Templates (klonalność/obfitość sekwencji)
        if 'templates' in df.columns:
            df['templates'] = pd.to_numeric(df['templates'], errors='coerce')
            features['total_templates'] = df['templates'].sum()
            features['mean_templates'] = df['templates'].mean()
            features['median_templates'] = df['templates'].median()
            features['std_templates'] = df['templates'].std()
            # Udział unikalnych sekwencji (1 = wszystkie unikalne, 0 = wszystkie duplikaty)
            features['unique_sequence_ratio'] = features['n_unique_junctions'] / features['n_sequences']
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)
    

def validate_sequences_in_datasets(data_dir: str, valid_aa: set = None, sequence_col: str = 'junction_aa'):
    """
    Sprawdza poprawność sekwencji we wszystkich zestawach danych (train i test).
    
    Args:
        data_dir: Ścieżka do katalogu z danymi (powinna zawierać train_datasets i test_datasets)
        valid_aa: Zbiór poprawnych aminokwasów (domyślnie standardowe 20 aminokwasów)
        sequence_col: Nazwa kolumny z sekwencjami
    
    Returns:
        dict: Słownik z wynikami sprawdzenia dla każdego zestawu danych
    """
    if valid_aa is None:
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    train_dir = os.path.join(data_dir, 'train_datasets')
    test_dir = os.path.join(data_dir, 'test_datasets')
    
    results = {
        'train': {},
        'test': {}
    }
    
    # Sprawdzamy zestawy danych train
    if os.path.exists(train_dir):
        train_datasets = sorted([d for d in os.listdir(train_dir) if d.startswith('train_dataset_')])
        print(f"Sprawdzanie zestawów danych train: {len(train_datasets)} zestawów")
        
        for dataset_name in tqdm(train_datasets, desc="Zestawy train"):
            dataset_path = os.path.join(train_dir, dataset_name)
            dataset_results = _validate_dataset(dataset_path, valid_aa, sequence_col, dataset_name)
            results['train'][dataset_name] = dataset_results
    
    # Sprawdzamy zestawy danych test
    if os.path.exists(test_dir):
        test_datasets = sorted([d for d in os.listdir(test_dir) if d.startswith('test_dataset_')])
        print(f"\nSprawdzanie zestawów danych test: {len(test_datasets)} zestawów")
        
        for dataset_name in tqdm(test_datasets, desc="Zestawy test"):
            dataset_path = os.path.join(test_dir, dataset_name)
            dataset_results = _validate_dataset(dataset_path, valid_aa, sequence_col, dataset_name)
            results['test'][dataset_name] = dataset_results
    
    # Wyświetlamy podsumowanie
    _print_validation_summary(results, valid_aa)
    
    return results


def _validate_dataset(dataset_path: str, valid_aa: set, sequence_col: str, dataset_name: str) -> dict:
    """
    Sprawdza poprawność sekwencji w jednym zestawie danych.
    
    Returns:
        dict: Statystyki dla zestawu danych
    """
    tsv_files = glob.glob(os.path.join(dataset_path, '*.tsv'))
    
    total_sequences = 0
    invalid_sequences = 0
    invalid_files = []
    invalid_examples = []
    
    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            
            if sequence_col not in df.columns:
                continue
            
            file_invalid_count = 0
            file_invalid_seqs = []
            
            for idx, seq in enumerate(df[sequence_col]):
                if pd.isna(seq) or seq == '':
                    continue
                
                total_sequences += 1
                seq_str = str(seq).strip()
                
                # Sprawdzamy, czy wszystkie znaki są poprawne
                invalid_chars = set(seq_str) - valid_aa
                if invalid_chars:
                    invalid_sequences += 1
                    file_invalid_count += 1
                    if len(file_invalid_seqs) < 10:  # Zapisujemy tylko pierwsze 10 przykładów
                        file_invalid_seqs.append({
                            'sequence': seq_str,
                            'invalid_chars': ''.join(sorted(invalid_chars)),
                            'row_index': idx
                        })
            
            if file_invalid_count > 0:
                invalid_files.append({
                    'filename': os.path.basename(tsv_file),
                    'invalid_count': file_invalid_count,
                    'total_count': len(df[df[sequence_col].notna()]),
                    'examples': file_invalid_seqs[:5]  # Pierwsze 5 przykładów
                })
        
        except Exception as e:
            print(f"Błąd podczas odczytu pliku {tsv_file}: {e}")
            continue
    
    return {
        'total_sequences': total_sequences,
        'invalid_sequences': invalid_sequences,
        'invalid_files_count': len(invalid_files),
        'invalid_files': invalid_files[:10],  # Pierwsze 10 plików z błędami
        'validity_rate': (total_sequences - invalid_sequences) / total_sequences if total_sequences > 0 else 1.0
    }


def _print_validation_summary(results: dict, valid_aa: set):
    """Wyświetla podsumowanie sprawdzenia poprawności."""
    print("\n" + "="*80)
    print("PODSUMOWANIE SPRAWDZENIA POPRAWNOŚCI SEKWENCJI")
    print("="*80)
    print(f"Poprawne aminokwasy: {''.join(sorted(valid_aa))}")
    print()
    
    # Inicjalizujemy zmienne dla ogólnego podsumowania
    train_total = 0
    train_invalid = 0
    test_total = 0
    test_invalid = 0
    
    # Zestawy danych train
    if results['train']:
        print("ZESTAWY DANYCH TRAIN:")
        print("-" * 80)
        
        for dataset_name, stats in sorted(results['train'].items()):
            train_total += stats['total_sequences']
            train_invalid += stats['invalid_sequences']
            validity_pct = stats['validity_rate'] * 100
            
            status = "✓" if stats['invalid_sequences'] == 0 else "✗"
            print(f"{status} {dataset_name:25s} | "
                  f"Razem: {stats['total_sequences']:>10,} | "
                  f"Niepoprawnych: {stats['invalid_sequences']:>8,} | "
                  f"Poprawność: {validity_pct:>6.2f}% | "
                  f"Plików z błędami: {stats['invalid_files_count']:>3}")
            
            # Pokazujemy przykłady niepoprawnych sekwencji
            if stats['invalid_files']:
                for file_info in stats['invalid_files'][:3]:  # Pierwsze 3 pliki
                    print(f"    Plik: {file_info['filename']}")
                    for example in file_info['examples'][:2]:  # Pierwsze 2 przykłady
                        print(f"      Sekwencja: {example['sequence'][:50]}... "
                              f"(niepoprawne znaki: {example['invalid_chars']})")
        
        train_validity = (train_total - train_invalid) / train_total if train_total > 0 else 1.0
        print(f"\nRazem train: {train_total:,} sekwencji, "
              f"{train_invalid:,} niepoprawnych ({100*(1-train_validity):.2f}%)")
        print()
    
    # Zestawy danych test
    if results['test']:
        print("ZESTAWY DANYCH TEST:")
        print("-" * 80)
        
        for dataset_name, stats in sorted(results['test'].items()):
            test_total += stats['total_sequences']
            test_invalid += stats['invalid_sequences']
            validity_pct = stats['validity_rate'] * 100
            
            status = "✓" if stats['invalid_sequences'] == 0 else "✗"
            print(f"{status} {dataset_name:25s} | "
                  f"Razem: {stats['total_sequences']:>10,} | "
                  f"Niepoprawnych: {stats['invalid_sequences']:>8,} | "
                  f"Poprawność: {validity_pct:>6.2f}% | "
                  f"Plików z błędami: {stats['invalid_files_count']:>3}")
            
            # Pokazujemy przykłady niepoprawnych sekwencji
            if stats['invalid_files']:
                for file_info in stats['invalid_files'][:3]:  # Pierwsze 3 pliki
                    print(f"    Plik: {file_info['filename']}")
                    for example in file_info['examples'][:2]:  # Pierwsze 2 przykłady
                        print(f"      Sekwencja: {example['sequence'][:50]}... "
                              f"(niepoprawne znaki: {example['invalid_chars']})")
        
        test_validity = (test_total - test_invalid) / test_total if test_total > 0 else 1.0
        print(f"\nRazem test: {test_total:,} sekwencji, "
              f"{test_invalid:,} niepoprawnych ({100*(1-test_validity):.2f}%)")
        print()
    
    # Ogólne podsumowanie
    total_seqs = train_total + test_total
    total_invalid = train_invalid + test_invalid
    overall_validity = (total_seqs - total_invalid) / total_seqs if total_seqs > 0 else 1.0
    
    print("="*80)
    print(f"OGÓLNE PODSUMOWANIE:")
    print(f"  Razem sekwencji: {total_seqs:,}")
    print(f"  Niepoprawnych sekwencji: {total_invalid:,} ({100*(1-overall_validity):.2f}%)")
    print(f"  Poprawnych sekwencji: {total_seqs - total_invalid:,} ({100*overall_validity:.2f}%)")
    print("="*80)
    print()


def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str, out_dir: str):
    """
    Trenuje predyktor na danych treningowych lub ładuje istniejący model, jeśli jest dostępny.
    
    Args:
        predictor: Instancja ImmuneStatePredictor
        train_dir: Ścieżka do katalogu danych treningowych
        out_dir: Ścieżka do katalogu wyjściowego, gdzie zapisywane są modele
    """
    # Sprawdzamy obecność zapisanego modelu
    model_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_model")
    model_pkl_path = model_path if model_path.endswith('.pkl') else f"{model_path}.pkl"
    model_json_path = model_path.replace('.pkl', '.json') if model_path.endswith('.pkl') else f"{model_path}.json"
    
    if os.path.exists(model_pkl_path) and os.path.exists(model_json_path):
        print(f"Znaleziono istniejący model w {model_path}. Ładowanie...")
        try:
            loaded_predictor = ImmuneStatePredictor.load_model(model_path, out_dir=out_dir, train_dir=train_dir)
            # Kopiujemy załadowany model do bieżącego predyktora
            predictor.model = loaded_predictor.model
            predictor.train_ids_ = loaded_predictor.train_ids_
            
            # Ładujemy important_sequences z pliku, jeśli istnieje
            important_seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
            if os.path.exists(important_seqs_path):
                try:
                    predictor.important_sequences_ = pd.read_csv(important_seqs_path, sep='\t')
                    print(f"Załadowano ważne sekwencje z {important_seqs_path}")
                except Exception as e:
                    print(f"Ostrzeżenie: Nie można załadować ważnych sekwencji: {e}")
            
            # Ładujemy kmer_importance z pliku, jeśli istnieje
            kmer_importance_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_kmer_importance.tsv")
            if os.path.exists(kmer_importance_path):
                try:
                    predictor.kmer_importance_df_ = pd.read_csv(kmer_importance_path, sep='\t')
                    print(f"Załadowano ważność k-merów z {kmer_importance_path}")
                except Exception as e:
                    print(f"Ostrzeżenie: Nie można załadować ważności k-merów: {e}")
            
            # Jeśli important_sequences nie zostały załadowane, obliczamy je
            if predictor.important_sequences_ is None or predictor.important_sequences_.empty:
                print("Obliczanie ważnych sekwencji...")
                important_seqs_result = predictor.identify_associated_sequences(
                    train_dir_path=train_dir, top_k=50000
                )
                if predictor.return_kmer_importance_:
                    predictor.important_sequences_, predictor.kmer_importance_df_ = important_seqs_result
                else:
                    predictor.important_sequences_ = important_seqs_result
                    predictor.kmer_importance_df_ = None
            
            print(f"Model załadowany pomyślnie. Najlepsze C: {predictor.model.best_C_}")
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}. Trenowanie nowego modelu...")
            print(f"Dopasowywanie modelu na przykładach w ` {train_dir} `...")
            predictor.fit(train_dir, out_dir)
    else:
        print(f"Dopasowywanie modelu na przykładach w ` {train_dir} `...")
        predictor.fit(train_dir, out_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str], results_dir: str) -> pd.DataFrame:
    """Generuje predykcje dla wszystkich katalogów testowych i łączy je."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"Przewidywanie na przykładach w ` {test_dir} `...")
        preds = predictor.predict_proba(test_dir, results_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"Ostrzeżenie: Brak zwróconych predykcji dla {test_dir}")
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    """Zapisuje predykcje do pliku TSV."""
    if predictions.empty:
        raise ValueError("Brak predykcji do zapisania - DataFrame predykcji jest pusty")

    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predykcje zapisane do `{preds_path}`.")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Zapisuje ważne sekwencje do pliku TSV."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("Brak dostępnych ważnych sekwencji do zapisania")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Ważne sekwencje zapisane do `{seqs_path}`.")


def _save_model(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Zapisuje model do pliku."""
    model_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_model")
    predictor.save_model(model_path)
    print(f"Model zapisany do `{model_path}.pkl`.")


def _save_kmer_importance(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Zapisuje ważność k-merów do pliku TSV."""
    kmer_importance_df = predictor.kmer_importance_df_
    if kmer_importance_df is None or kmer_importance_df.empty:
        raise ValueError("Brak dostępnej ważności k-merów do zapisania")
    kmer_importance_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_kmer_importance.tsv")
    save_tsv(kmer_importance_df, kmer_importance_path)
    print(f"Ważność k-merów zapisana do `{kmer_importance_path}`.")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str, return_kmer_importance: bool = False, k: int = 3) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(n_jobs=n_jobs,
                                     device=device,
                                     return_kmer_importance=return_kmer_importance,
                                     out_dir=out_dir,
                                     train_dir=train_dir,
                                     k=k)  # instancja z dowolnymi innymi parametrami zdefiniowanymi przez Ciebie w klasie
    _train_predictor(predictor, train_dir, out_dir)
    
    # Sprawdzamy obecność pliku z predykcjami
    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    if os.path.exists(preds_path):
        print(f"Znaleziono istniejące predykcje w `{preds_path}`. Ładowanie...")
        try:
            predictions = pd.read_csv(preds_path, sep='\t')
            print(f"Predykcje załadowane: {len(predictions)} wierszy")
        except Exception as e:
            print(f"Błąd ładowania predykcji: {e}. Generowanie nowych predykcji...")
            predictions = _generate_predictions(predictor, test_dirs, out_dir)
            _save_predictions(predictions, out_dir, train_dir)
    else:
        predictions = _generate_predictions(predictor, test_dirs, out_dir)
        _save_predictions(predictions, out_dir, train_dir)
    
    # Zapisujemy important_sequences tylko jeśli zostały obliczone i plik jeszcze nie istnieje
    important_seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    if not os.path.exists(important_seqs_path):
        if predictor.important_sequences_ is not None and not predictor.important_sequences_.empty:
            _save_important_sequences(predictor, out_dir, train_dir)
        else:
            print(f"Ostrzeżenie: Ważne sekwencje nie zostały obliczone i plik nie istnieje. Pomijanie zapisu...")
    else:
        print(f"Ważne sekwencje już istnieją w `{important_seqs_path}`. Pomijanie...")
    
    # Zapisujemy model tylko jeśli jeszcze go nie ma (lub jeśli został wytrenowany nowy)
    model_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_model")
    model_pkl_path = model_path if model_path.endswith('.pkl') else f"{model_path}.pkl"
    if not os.path.exists(model_pkl_path):
        _save_model(predictor, out_dir, train_dir)
    else:
        print(f"Model już istnieje w `{model_pkl_path}`. Pomijanie zapisu...")


def run():
    parser = argparse.ArgumentParser(description="Interfejs CLI predyktora stanu immunologicznego")
    parser.add_argument("--train_dir", required=True, help="Ścieżka do katalogu danych treningowych")
    parser.add_argument("--test_dirs", required=True, nargs="+", help="Ścieżka(i) do katalogu(ów) danych testowych")
    parser.add_argument("--out_dir", required=True, help="Ścieżka do katalogu wyjściowego")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Liczba rdzeni CPU do użycia. Użyj -1 dla wszystkich dostępnych rdzeni.")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Urządzenie do użycia w obliczeniach ('cpu' lub 'cuda').")
    parser.add_argument("--return_kmer_importance", action="store_true",
                        help="Zwróć ważność k-merów.")
    parser.add_argument("--k", type=int, default=3,
                        help="Długość k-meru.")
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device, args.return_kmer_importance, args.k)