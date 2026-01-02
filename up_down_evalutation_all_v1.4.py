import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from lightgbm import LGBMClassifier # ML-Modell
from sklearn.metrics import ( # ML-Modell Metriken
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt #Visualisierungen
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============ NEUE IMPORTS FÜR MODELL-VERGLEICH ============
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#Transformer
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten

#TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# 1D-CNN Training
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dropout

# Hybrid LSTM + Tranformer Training
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, MultiHeadAttention, Add, GlobalAveragePooling1D, LayerNormalization

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from sklearn.inspection import permutation_importance

# ============================================================


###############################################################################
# Globals
###############################################################################
DEBUG = 1
# Unterscheidung zwischen Mini-Backtest Hold&Sell oder Hold&Sell(bei Signal = Down)
VARIANTE = 2
#from IPython.display import display, HTML

# Pfad zu den Ordnern mit den Daten
BASE_DIR = Path(r"data")
PRICE_DIR     = BASE_DIR / "prices_v3"          # Pfad zu den Aktienkursen
FUND_DIR      = BASE_DIR / "fundamentals_v1"    # Pfad zu den Quartalsberichten und Unternehmenskennzahlen

TARGET_HORIZON_DAYS = 5                # für wie viele Tage in die Zukunft die Vorhersage gelten soll
TEST_SPLIT_RATIO = 0.05                 # Aufteilung der Daten in Trainings- und Testdaten (z.b. 80% Training/20% Test)
PROBA_THRESHOLD = 0.55                 # Schwellwert der Wahrscheinlichkeit, ab wann die Vorhersage "UP" ausgibt und ab wann "DOWN"
VOLATILITY = 20                        # Anzahl der Tage für die Berechnung der Volatilität
MOMENTUM = 5                           # Anzahl der Tage für die Berechnung des Momentums
VOLUME = 5                             # Anzahl der Tage für die Berechnung der Volume-Dynamik

REPORT_FILE_NAME = f"Klassifikationsreport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Mapping zwischen Aktienkurs-Dateinamen und Unternehmenskennzahlen-Dateinamen
# Aktienkurs-Dateinamen ohne Suffix (_US.csv etc.)
# Unternehmenskennzahlen-Dateiname ohne _balance_sheet.csv etc.

TICKER_MAP = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "NVDA": "NVDA",
    "AMZN": "AMZN",
    "META": "META",
    "TSM": "TSM",
    "BRK-A": "BRK-A",
    "TSLA": "TSLA",
    "LLY": "LLY",
    "V": "V",
    "MA": "MA",
    "XOM": "XOM",
    "JNJ": "JNJ",
    "HD": "HD",
    "ASML": "ASML",
    "BABA": "BABA",
    "BAC": "BAC",
    "AMD": "AMD",
    "PG": "PG",
    "UNH": "UNH",
    "SAP": "SAP.DE",
    "CVX": "CVX",
    "KO": "KO",
    "CSCO": "CSCO",
    "TM": "TM",
    "WFC": "WFC",
    "AZN": "AZN",
    "TMUS": "TMUS",
    "NVS": "NVS",
    "MS": "MS",
    "CRM": "CRM",
    "PM": "PM",
    "CAT": "CAT",
    "RTX": "RTX",
    "NVO": "NVO",

}


###############################################################################
# Funktionen
###############################################################################
def build_price_features(price_df: pd.DataFrame, horizon_days: int, volatility: int, momentum: int, volume : int) -> pd.DataFrame: #Rückgabehinweis -> pd.DataFrame, aber keine Überprüfung/Zwang

    # Kopieren des Dataset
    df = price_df.copy()
    # Sortieren des Dataset nach der Spalte "Date" und Entfernen des zuvorigen Index 
    df = df.sort_values("Date").reset_index(drop=True)

    # Berechnet die tägliche Rendite (Differenz zwischen den Spalten "close" und "close vom Vortag") und speichert die Ergebnisse als eigene Spalte "Return_1d" ab
    df["Return_1d"] = df["Close"].pct_change()

    # Close des Tages, welcher x Tage in der Zukunft liegt(x=Anzahl der Tage, die in die Zukunft geschaut wird [horizon_days])
    future_close = df["Close"].shift(-horizon_days)
    # (Close von x Tage in der Zukunft durch das Close heute -1) und Speicherung des Ergebnisses in der "Return_fwd"-Spalte
    df["Return_fwd"] = future_close / df["Close"] - 1.0

    # Ist der Kurs nach x Tagen höher oder niedriger? -> Speicherung des Ergebnisses in der "Target"-Spalte
    df["Target"] = (df["Return_fwd"] > 0).astype(int)

    # Berechnung des gleitende Durchschnitt für 10 Tage 
    df["SMA10"] = df["Close"].rolling(10).mean() # nimmt die nächsten 10 Werte und ermittelt den Mittelwert davon
    # Berechnung des gleitende Durchschnitt für 50 Tage
    df["SMA50"] = df["Close"].rolling(50).mean()
    # Berechnung des Verhältnis um Trends abzuleiten 
    df["SMA_ratio"] = df["SMA10"] / df["SMA50"]

    # Volatilität - Standardabweichung der täglichen Returns über x Tage
    df["Volatility"] = df["Return_1d"].rolling(volatility).std()

    # Momentum der letzten x Tage (Close von heute durch das Close x Tage in der Vergangenheit -1)
    df["Momentum"] = df["Close"] / df["Close"].shift(momentum) - 1.0

    # Volumen-Dynamik über x Tage (Differenz des Volumens zwischen Volumen des Tages und des Volumens x Tage davor)
    df["VolumeChange"] = df["Volume"].pct_change(volume)

    return df

def load_fundamentals_for_ticker(fund_base: str, fund_dir: Path) -> pd.DataFrame:

    # Pfad zum Balance Sheet
    bal_path = fund_dir / f"{fund_base}_balance_sheet.csv"
    # Pfad zum Income Statement
    inc_path = fund_dir / f"{fund_base}_income_statement.csv"

    # Falls kein Balance Sheet oder Income Statement vorhanden ist -> Return
    if (not bal_path.exists()) or (not inc_path.exists()):
        # Falls eins fehlt -> kein Fundamentals-Join möglich
        return None

    # Auslesen der CSV
    bal = pd.read_csv(bal_path, parse_dates=["fiscalDateEnding"])
    inc = pd.read_csv(inc_path, parse_dates=["fiscalDateEnding"])

    # Alle Spalten, die Zahlen enthalten könnten, werden in float oder int umgewandelt (Fehler werden zu NaN)
    for df in [bal, inc]:
        for col in df.columns:
            if col not in ["fiscalDateEnding", "reportedCurrency", "period"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Überprüfen, ob die Spalten exisiteren, sonst NaN
    def safe_col(df, col):
        return df[col] if col in df.columns else np.nan

    # Erstellen eines neuen Balance_Sheet-Dataset mit den wichtigsten Spalten/Informationen
    bal_feat = pd.DataFrame({
        "fiscalDateEnding": safe_col(bal, "fiscalDateEnding"),
        "totalLiabilities": safe_col(bal, "totalLiabilities"),
        "totalShareholderEquity": safe_col(bal, "totalShareholderEquity"),
        "totalAssets": safe_col(bal, "totalAssets"),
    })

    # Berechnung des Verschuldungsgrades (Verbindlichkeiten / Eigenkapital) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    bal_feat["DebtEquity"] = (
        bal_feat["totalLiabilities"] /
        bal_feat["totalShareholderEquity"].replace({0: np.nan})
    )

    # Berechnung der Eigenkapitalquote (Gesamtvermögen / Eigenkapital) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    bal_feat["Leverage"] = (
        bal_feat["totalAssets"] /
        bal_feat["totalShareholderEquity"].replace({0: np.nan})
    )

    # Erstellen eines neuen Income_Statement-Dataset mit den wichtigsten Spalten/Informationen
    inc_feat = pd.DataFrame({
        "fiscalDateEnding": safe_col(inc, "fiscalDateEnding"),
        "totalRevenue": safe_col(inc, "totalRevenue"),
        "netIncome": safe_col(inc, "netIncome"),
        "operatingIncome": safe_col(inc, "operatingIncome"),
    })

    # Berechnung des Nettogewinn am Umsatz (Nettogewinn / Gesamtumsatz) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    inc_feat["ProfitMargin"] = (
        inc_feat["netIncome"] /
        inc_feat["totalRevenue"].replace({0: np.nan})
    )

    # Berechnung der Betriebsergebnis-Quote (EBIT / Gesamtumsatz) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    inc_feat["OperatingMargin"] = (
        inc_feat["operatingIncome"] /
        inc_feat["totalRevenue"].replace({0: np.nan})
    )

    # Zusammenfügen der beiden Datasets zu einem 
    fundamentals = pd.merge(
        bal_feat,
        inc_feat,
        on="fiscalDateEnding",
        how="outer"
    ).sort_values("fiscalDateEnding")

    # Neu-Benennung der Reporting-Spalte
    fundamentals = fundamentals.rename(columns={"fiscalDateEnding": "ReportDate"})
    # Neusetzen des Index
    fundamentals = fundamentals.reset_index(drop=True)
    
    return fundamentals

def merge_price_and_fundamentals(price_df: pd.DataFrame,
                                 fund_df: pd.DataFrame) -> pd.DataFrame:

    # Überprüfen, ob beide Dataset vorhanden sind
    # Falls kein Fundamentals Dataset vorhanden ist -> Aktienkurs-Dataset bleibt bestehen und Fundamentals Spalten werden mit NaN aufgefüllt
    if fund_df is None or fund_df.empty:
        # Kopieren des Aktienkurs-Datasets
        merged = price_df.copy()
        # Iteration über Fundamentals Spalten
        for col in ["DebtEquity", "Leverage", "ProfitMargin", "OperatingMargin"]:
            # Falls die Spalte nicht im merged-Dataset vorhanden ist -> dazuhängen und mit NaN auffüllen
            if col not in merged.columns:
                merged[col] = np.nan
        return merged

    # Sortieren der Datasets nach Datum und Reset des Index
    price_sorted = price_df.sort_values("Date").reset_index(drop=True)
    fund_sorted = fund_df.sort_values("ReportDate").reset_index(drop=True)

    # Zusammenführen der beiden Dataset, wobei das Mapping Unternehmensbericht-Datum <= (älter) Aktienkurs-Datum ist - letzter verfügbare Bericht vor dem Aktienkursdatum
    merged = pd.merge_asof(
        price_sorted,
        fund_sorted,
        left_on="Date",
        right_on="ReportDate",
        direction="backward"
    )

    return merged

def time_based_train_test_split(df: pd.DataFrame, test_ratio: float):
    # Sortieren des Datasets nach Datum + Index wird neu gesetzt
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    # Berechnung Anzahl der Einträge im Dataset
    n = len(df_sorted)
    # Berechnung Anzahl der Zeilen für das Training-Dataset

    split_idx = int(np.floor((1 - test_ratio) * n))
    # Verwendung als Index für das Splitten des Datasets in ein Training- und ein Test-Datasets
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df

def evaluate_model(
    clf,
    X_train, y_train,
    X_test, y_test,
    df_test_full,
    proba_threshold=0.55,
    horizon_days=5,
    model_type="sklearn"  # "sklearn" oder "lstm"
    ):

    if model_type == "tabnet":
        # TabNet: numpy arrays verwenden
        X_test_np = np.asarray(X_test)
        y_pred = clf.predict(X_test_np)
        y_pred_proba = clf.predict_proba(X_test_np)[:, 1]
    elif model_type == "lstm" or model_type == "cnn" or model_type == "hybrid":
        # Für LSTM: y_pred und y_pred_proba aus 3D-Array berechnen

        y_pred_proba = clf.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        # Für sklearn-Modelle: wie bisher
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Testen des Modell mit Daten des Test-Datasets
    #y_pred = clf.predict(X_test)
    # Berechnung der Wahrscheinlichkeit, ob die Aktien steigen (1) oder nicht (0) 
    # -> clf.predict_proba(X_test) gibt ein 2D-Array zurück mit 1.Spalte, Wahrscheinlichkeit, dass der Kurs fällt und 2. Spalte, Wahrscheinlichkeit, dass der Kurs steigt
    # [:, 1] -> nehme nur die zweite Spalte
    #y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Ausgabe des Klassifikationsreports (wie gut passt das Modell zu den Test Daten (wie gut ist die Modell-Leistung))
    print("=== Klassifikationsreport (Test) ===")
    # 0 = nicht steigen; 1 = steigen
    # precision = Wie viele der positiven Vorhersagen waren tatsächlich richtig? Formel: TP / (TP + FP)
    # recall = Wie viele der tatsächlich positiven Fälle hat das Modell erkannt? Formel: TP / (TP + FN)
    # f1-score = Harmonic mean aus Precision & Recall → Balance zwischen „Genauigkeit der positiven Vorhersagen“ und „Empfindlichkeit“
    #support = Anzahl der tatsächlichen Beispiele dieser Klasse in y_test
    # accuracy = Gesamtanteil korrekt klassifizierter Samples. Formel: (TP + TN) / Gesamtzahl
    # macro avg = Durchschnitt von Precision, Recall und F1 über alle Klassen, ohne Gewichtung (jede Klasse zählt gleich stark)
    # weighted avg = Durchschnitt von Precision, Recall und F1, gewichtet nach support (Klassen mit mehr Beispielen zählen stärker)
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    # Print to txt
    with open("Klassifikationsreport/"+REPORT_FILE_NAME, "w", encoding="utf-8") as f:
        print("=== Klassifikationsreport (Test) ===", file=f)
        print(classification_report(y_test, y_pred, digits=3, zero_division=0), file=f)

    # Versuch, ob auch ein ROC-AUC-Score erstellt werden kann
    # Wie gut unterscheidet Modell „wird steigen“ vs. „wird nicht steigen“
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC: {auc:.3f}")

        with open("Klassifikationsreport/"+REPORT_FILE_NAME, "a", encoding="utf-8") as f:
            print(f"ROC-AUC: {auc:.3f}", file=f)
    except ValueError:
        print("ROC-AUC nicht berechenbar (nur eine Klasse im Test-Set).")

    # Berechnugn des Accuracy Score - misst, wie viele Vorhersagen insgesamt richtig waren
    # acc = (TP + TN) / (TP + TN + FP + FN)
    acc = accuracy_score(y_test, y_pred)
    # Berechnung der Präzision - misst, wie viele der vorhergesagten positiven Fälle tatsächlich positiv waren.
    # prec = TP / (TP + FP)
    prec = precision_score(y_test, y_pred, zero_division=0)
    # Berechnung des Recalls - misst, wie viele der tatsächlich positiven Fälle erkannt wurden.
    # rec = TP / (TP + FN)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Ausgabe der Metriken
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")

    with open("Klassifikationsreport/"+REPORT_FILE_NAME, "a", encoding="utf-8") as f:
        print(f"Accuracy : {acc:.3f}", file=f)
        print(f"Precision: {prec:.3f}", file=f)
        print(f"Recall   : {rec:.3f}", file=f)

    if VARIANTE == 1: #30 Tage halten und dann verkaufen
        # Mini-Backtest
        strat_df = df_test_full.copy()

        # Datum und Sortierung sicherstellen
        strat_df["Date"] = pd.to_datetime(strat_df["Date"])
        strat_df = strat_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        if len(y_pred_proba) < len(strat_df):
            strat_df = strat_df.iloc[-len(y_pred_proba):].copy()

        # Wahrscheinlichkeit und Signal
        strat_df["proba_up"] = y_pred_proba
        strat_df["signal_long"] = (strat_df["proba_up"] > proba_threshold).astype(int)

        # Pro Ticker Positions-Flag über <horizon_days> Tage erzeugen
        def build_position_per_ticker(g):
            sig = g["signal_long"].to_numpy()
            n = len(g)
            pos = np.zeros(n, dtype=int)

            remaining = 0  # wie viele Tage die aktuelle Position noch läuft

            for i in range(n):
                if remaining > 0:
                    # aktueller Trade aktiv
                    pos[i] = 1
                    remaining -= 1
                elif sig[i] == 1:
                    # neues Einstiegssignal, d.h. neue Position eröffnen
                    pos[i] = 1
                    remaining = horizon_days - 1  # heute inklusive, daher -1

            g["position"] = pos
            return g

        # Anwenden der oben definierten Funktion
        strat_df = strat_df.groupby("Ticker", group_keys=False).apply(build_position_per_ticker)

        # Tages-Strategie-Return mit Return_1d
        strat_df["strategy_return_1d"] = strat_df["position"] * strat_df["Return_1d"]

        # Gleichgewichtete tägliche Portfolio-Rendite
        def equal_weight_daily(d):
            active = d["position"].sum()
            if active == 0:
                return 0.0  # kein Trade aktiv - 0% Tagesreturn
            # Durchschnitt der Returns der aktiven Positionen
            return d.loc[d["position"] == 1, "strategy_return_1d"].mean()

        # Anwendung der oben genannten Funktion
        daily_return = strat_df.groupby("Date").apply(equal_weight_daily)

        # Equity-Kurve und Gesamtrendite
        equity_curve_daily = (1 + daily_return).cumprod()
        total_return_daily = equity_curve_daily.iloc[-1] - 1 if len(equity_curve_daily) else np.nan

        print(f"\n=== Up/Down-Schwellwert für potenzielle Long-Strategie: {proba_threshold} ===")
        print(f"Variante: Mehrere Aktien, {horizon_days}-Tage-Hold, tägliche Gleichgewichtung")
        print(f"Mögliche Gesamtrendite bei Verwendung des Up/Down-Schwellwerts für den Testzeitraum {strat_df['Date'].dt.date.min()} bis {strat_df['Date'].dt.date.max()}: {total_return_daily:.2%}")

        # Plot Strategie-Kapitalverlauf über Testzeitraum
        plt.figure(figsize=(8,4))
        plt.plot(equity_curve_daily.index, equity_curve_daily.values, label="Strategie-Kapitalverlauf")
        plt.title("Backtest (Test-Periode, alle Ticker)")
        plt.xlabel("Test-Index (zeitlich sortiert)")
        plt.ylabel("Kumulierte Rendite")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    elif VARIANTE ==2: #Position so lange halten, wie Signal Long bleibt, aber mindestens horizon_days
        # Mini Backtest Variante
        strat_df = df_test_full.copy()

        # Datum und Sortierung
        strat_df["Date"] = pd.to_datetime(strat_df["Date"])
        strat_df = strat_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        if len(y_pred_proba) < len(strat_df):
            strat_df = strat_df.iloc[-len(y_pred_proba):].copy()

        # Wahrscheinlichkeit und Signal
        strat_df["proba_up"] = y_pred_proba
        strat_df["signal_long"] = (strat_df["proba_up"] > proba_threshold).astype(int)

        def build_position_var2(g):
            sig = g["signal_long"].to_numpy()
            n = len(g)
            pos = np.zeros(n, dtype=int)

            in_pos = False
            hold_days_remaining = 0  # Resttage der Mindesthaltedauer

            for i in range(n):
                if in_pos:
                    # aktiver Trade vorhanden
                    pos[i] = 1
                    if hold_days_remaining > 0:
                        hold_days_remaining -= 1
                    else:
                        # Mindesthaltedauer ist vorbei - Entscheidung über das Signal
                        if sig[i] == 0:
                            in_pos = False
                            pos[i] = 0 
                else:
                    # aktuell kein aktiver Trade - neues Signal kann Position öffnen
                    if sig[i] == 1:
                        in_pos = True
                        pos[i] = 1
                        hold_days_remaining = horizon_days - 1

            g["position_D"] = pos
            return g

        strat_df = strat_df.groupby("Ticker", group_keys=False).apply(build_position_var2, include_groups=False)

        # Tages Strategie Return aus Return_1d
        strat_df["strategy_return_1d_D"] = strat_df["position_D"] * strat_df["Return_1d"]

        # Gleichgewichtete tägliche Portfolio Rendite
        def equal_weight_daily_2(d):
            active = d["position_D"].sum()
            if active == 0:
                return 0.0
            return d.loc[d["position_D"] == 1, "strategy_return_1d_D"].mean()

        daily_return_D = strat_df.groupby("Date").apply(equal_weight_daily_2, include_groups=False)

        # Equity Kurve und Gesamtrendite
        equity_curve_D = (1 + daily_return_D).cumprod()
        total_return_D = equity_curve_D.iloc[-1] - 1 if len(equity_curve_D) else np.nan

        print(f"\n=== Variante D: min {horizon_days} Tage Hold, dann solange Signal Long bleibt ===")
        print(f"Schwellwert: {proba_threshold}")
        print(f"Mögliche Gesamtrendite bei Verwendung des Up/Down-Schwellwerts für den Testzeitraum {strat_df['Date'].dt.date.min()} bis {strat_df['Date'].dt.date.max()}: {total_return_D:.2%}")

        # Plot Strategie-Kapitalverlauf über Testzeitraum
        plt.figure(figsize=(8,4))
        plt.plot(equity_curve_D.index, equity_curve_D.values, label="Strategie-Kapitalverlauf")
        plt.title("Backtest (Test-Periode, alle Ticker)")
        plt.xlabel("Test-Index (zeitlich sortiert)")
        plt.ylabel("Kumulierte Rendite")
        plt.legend()
        plt.tight_layout()
        plt.show()


def compare_models_simple(X_train, y_train, X_test, y_test, test_df, columns, proba_threshold=0.55, horizon_days=5):
    """Trainiere und vergleiche 5 top Modelle"""
    results_list = []
    models_dict = {}
    
    print("\n" + "="*70)
    print("MODEL COMPARISON - TRAINING ALLE MODELLE")
    print("="*70)
    
    # Daten bereinigen: Inf/NaN entfernen
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    y_train_clean = y_train[X_train_clean.index]
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    y_test_clean = y_test[X_test_clean.index]

    # Überprüfung Class Distribution
    #print("Y-Verteilung:")
    #print(y_train_clean.value_counts())
    #print(f"\nClass Balance: {y_train_clean.value_counts(normalize=True)}")
    
    # Standardisierung
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # DataFrame mit Spaltennamen wiederherstellen - TODO:Gute Idee zu Standardisieren?
    #X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
    #X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)
    X_train_scaled = pd.DataFrame(X_train_clean, columns=columns)
    X_test_scaled = pd.DataFrame(X_test_clean, columns=columns)
 
    # 1. LightGBM
    print("\n[1/10] Training LightGBM...")
    try:
        lgb_clf = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='auc', early_stopping_rounds=20, verbosity=-1
        )
        lgb_clf.fit(X_train_scaled, y_train_clean, eval_set=[(X_test_scaled, y_test_clean)], feature_name=columns)
        lgb_pred = lgb_clf.predict(X_test_scaled)
        lgb_proba = lgb_clf.predict_proba(X_test_scaled)[:, 1]
        lgb_f1 = f1_score(y_test_clean, lgb_pred, zero_division=0)
        lgb_acc = accuracy_score(y_test_clean, lgb_pred)
        lgb_auc = roc_auc_score(y_test_clean, lgb_proba) if len(set(y_test_clean)) > 1 else 0.0
        results_list.append({'Modell': 'LightGBM', 'F1': lgb_f1, 'AUC': lgb_auc, 'Acc': lgb_acc})
        models_dict['lgb'] = lgb_clf
        print(f"   ✓ F1={lgb_f1:.3f}, AUC={lgb_auc:.3f}, Acc={lgb_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 2. XGBoost
    print("[2/10] Training XGBoost...")
    try:
        xgb_clf = XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='auc', early_stopping_rounds=20, verbosity=0
        )
        xgb_clf.fit(X_train_scaled, y_train_clean, eval_set=[(X_test_scaled, y_test_clean)], verbose=False)
        xgb_pred = xgb_clf.predict(X_test_scaled)
        xgb_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]
        xgb_f1 = f1_score(y_test_clean, xgb_pred, zero_division=0)
        xgb_acc = accuracy_score(y_test_clean, xgb_pred)
        xgb_auc = roc_auc_score(y_test_clean, xgb_proba) if len(set(y_test_clean)) > 1 else 0.0
        results_list.append({'Modell': 'XGBoost', 'F1': xgb_f1, 'AUC': xgb_auc, 'Acc': xgb_acc})
        models_dict['xgb'] = xgb_clf
        print(f"   ✓ F1={xgb_f1:.3f}, AUC={xgb_auc:.3f}, Acc={xgb_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 3. CatBoost
    print("[3/10] Training CatBoost...")
    try:
        cat_clf = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6,
            subsample=0.8, random_state=42, verbose=0, early_stopping_rounds=20
        )
        cat_clf.fit(X_train_scaled, y_train_clean, eval_set=[(X_test_scaled, y_test_clean)], verbose=False)
        cat_pred = cat_clf.predict(X_test_scaled)
        cat_proba = cat_clf.predict_proba(X_test_scaled)[:, 1]
        cat_f1 = f1_score(y_test_clean, cat_pred, zero_division=0)
        cat_acc = accuracy_score(y_test_clean, cat_pred)
        cat_auc = roc_auc_score(y_test_clean, cat_proba) if len(set(y_test_clean)) > 1 else 0.0
        results_list.append({'Modell': 'CatBoost', 'F1': cat_f1, 'AUC': cat_auc, 'Acc': cat_acc})
        models_dict['cat'] = cat_clf
        print(f"   ✓ F1={cat_f1:.3f}, AUC={cat_auc:.3f}, Acc={cat_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 4. Random Forest
    print("[4/10] Training Random Forest...")
    try:
        rf_clf = RandomForestClassifier(
            n_estimators=300, max_depth=15, random_state=42, n_jobs=-1, verbose=0
        )
        rf_clf.fit(X_train_scaled, y_train_clean)
        rf_pred = rf_clf.predict(X_test_scaled)
        rf_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]
        rf_f1 = f1_score(y_test_clean, rf_pred, zero_division=0)
        rf_acc = accuracy_score(y_test_clean, rf_pred)
        rf_auc = roc_auc_score(y_test_clean, rf_proba) if len(set(y_test_clean)) > 1 else 0.0
        results_list.append({'Modell': 'Random Forest', 'F1': rf_f1, 'AUC': rf_auc, 'Acc': rf_acc})
        models_dict['rf'] = rf_clf
        print(f"   ✓ F1={rf_f1:.3f}, AUC={rf_auc:.3f}, Acc={rf_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 5. Stacking
    print("[5/10] Training Stacking Ensemble...")
    try:
        base_learners = [
            ('lgb', LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)),
            ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0)),
            ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, random_state=42, verbose=0))
        ]
        stack_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5
        )
        stack_clf.fit(X_train_scaled, y_train_clean)
        stack_pred = stack_clf.predict(X_test_scaled)
        stack_proba = stack_clf.predict_proba(X_test_scaled)[:, 1]
        stack_f1 = f1_score(y_test_clean, stack_pred)
        stack_acc = accuracy_score(y_test_clean, stack_pred)
        stack_auc = roc_auc_score(y_test_clean, stack_proba) if len(set(y_test_clean)) > 1 else 0.0
        results_list.append({'Modell': 'Stacking', 'F1': stack_f1, 'AUC': stack_auc, 'Acc': stack_acc})
        models_dict['stack'] = stack_clf
        print(f"   ✓ F1={stack_f1:.3f}, AUC={stack_auc:.3f}, Acc={stack_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 6. LSTM
    print("[6/10] Training LSTM...")
    try:
        lstm_model, X_test_lstm, y_test_lstm = train_lstm(X_train_scaled.values, y_train_clean.values, X_test_scaled.values, y_test_clean.values, TARGET_HORIZON_DAYS)
        lstm_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
        lstm_proba = lstm_model.predict(X_test_lstm).flatten()
        lstm_f1 = f1_score(y_test_lstm, lstm_pred, zero_division=0)
        lstm_acc = accuracy_score(y_test_lstm, lstm_pred)
        lstm_auc = roc_auc_score(y_test_lstm, lstm_proba) if len(set(y_test_lstm)) > 1 else 0.0
        results_list.append({'Modell': 'LSTM', 'F1': lstm_f1, 'AUC': lstm_auc, 'Acc': lstm_acc})
        models_dict['lstm'] = lstm_model
        print(f"   ✓ F1={lstm_f1:.3f}, AUC={lstm_auc:.3f}, Acc={lstm_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 7. Transformer
    print("[7/10] Training Transformer...")
    try:
        transformer_model, X_test_transformer, y_test_transformer = train_transformer(X_train_scaled.values, y_train_clean.values, X_test_scaled.values, y_test_clean.values)
        transformer_pred = (transformer_model.predict(X_test_transformer) > 0.5).astype(int).flatten()
        transformer_proba = transformer_model.predict(X_test_transformer).flatten()
        transformer_f1 = f1_score(y_test_transformer, transformer_pred, zero_division=0)
        transformer_acc = accuracy_score(y_test_transformer, transformer_pred)
        transformer_auc = roc_auc_score(y_test_transformer, transformer_proba) if len(set(y_test_transformer)) > 1 else 0.0
        results_list.append({'Modell': 'Transformer', 'F1': transformer_f1, 'AUC': transformer_auc, 'Acc': transformer_acc})
        models_dict['transformer'] = transformer_model
        print(f"   ✓ F1={transformer_f1:.3f}, AUC={transformer_auc:.3f}, Acc={transformer_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # 8. TabNet
    print("[8/10] Training TabNet...")
    try:
        tab_clf = train_tabnet(
            X_train_scaled.values, y_train_clean.values,
            X_test_scaled.values, y_test_clean.values
        )
        tab_pred = tab_clf.predict(X_test_scaled.values)
        tab_proba = tab_clf.predict_proba(X_test_scaled.values)[:, 1]
        tab_f1 = f1_score(y_test_clean, tab_pred, zero_division=0)
        tab_acc = accuracy_score(y_test_clean, tab_pred)
        tab_auc = roc_auc_score(y_test_clean, tab_proba) if len(set(y_test_clean)) > 1 else 0.0
        results_list.append({'Modell': 'TabNet', 'F1': tab_f1, 'AUC': tab_auc, 'Acc': tab_acc})
        models_dict['tabnet'] = tab_clf
        print(f"   ✓ F1={tab_f1:.3f}, AUC={tab_auc:.3f}, Acc={tab_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")


    # 9. 1D-CNN
    print("[9/10] Training 1D-CNN...")
    try:
        cnn_model, X_test_cnn, y_test_cnn = train_cnn1d(X_train_scaled.values, y_train_clean.values, X_test_scaled.values, y_test_clean.values)
        cnn_pred = (cnn_model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()
        cnn_proba = cnn_model.predict(X_test_cnn, verbose=0).flatten()
        cnn_f1 = f1_score(y_test_cnn, cnn_pred, zero_division=0)
        cnn_acc = accuracy_score(y_test_cnn, cnn_pred)
        cnn_auc = roc_auc_score(y_test_cnn, cnn_proba) if len(set(y_test_cnn)) > 1 else 0.0
        results_list.append({'Modell': '1D-CNN', 'F1': cnn_f1, 'AUC': cnn_auc, 'Acc': cnn_acc})
        models_dict['cnn'] = cnn_model
        print(f"   ✓ F1={cnn_f1:.3f}, AUC={cnn_auc:.3f}, Acc={cnn_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")


    # 10. Hybrid LSTM+Transformer
    print("[10/10] Training Hybrid LSTM+Transformer...")
    try:
        hybrid_model, X_test_hybrid, y_test_hybrid = train_hybrid_lstm_transformer(X_train_scaled.values, y_train_clean.values, X_test_scaled.values, y_test_clean.values)
        hybrid_pred = (hybrid_model.predict(X_test_hybrid, verbose=0) > 0.5).astype(int).flatten()
        hybrid_proba = hybrid_model.predict(X_test_hybrid, verbose=0).flatten()
        hybrid_f1 = f1_score(y_test_hybrid, hybrid_pred, zero_division=0)
        hybrid_acc = accuracy_score(y_test_hybrid, hybrid_pred)
        hybrid_auc = roc_auc_score(y_test_hybrid, hybrid_proba) if len(set(y_test_hybrid)) > 1 else 0.0
        results_list.append({'Modell': 'Hybrid LSTM+Transformer', 'F1': hybrid_f1, 'AUC': hybrid_auc, 'Acc': hybrid_acc})
        models_dict['hybrid'] = hybrid_model
        print(f"   ✓ F1={hybrid_f1:.3f}, AUC={hybrid_auc:.3f}, Acc={hybrid_acc:.3f}")
    except Exception as e:
        print(f"   ✗ Fehler: {str(e)[:80]}")
    
    # Zusammenfassung
    results_df = pd.DataFrame(results_list).sort_values('F1', ascending=False)
    print("\n" + "="*70)
    print("ERGEBNISSE:")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    results_df.to_csv(f"Model_comparison/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    best_model_name = results_df.iloc[0]['Modell']
    print(f"\n🏆 BESTES MODELL: {best_model_name} (F1={results_df.iloc[0]['F1']:.3f})")
    
    if best_model_name == "LSTM":
        return results_df, models_dict, best_model_name, X_test_lstm, y_test_lstm
    elif best_model_name == "Transformer":
        return results_df, models_dict, best_model_name, X_test_transformer, y_test_transformer
    elif best_model_name == "1D-CNN":
        return results_df, models_dict, best_model_name, X_test_cnn, y_test_cnn
    elif best_model_name == "Hybrid LSTM+Transformer":
        return results_df, models_dict, best_model_name, X_test_hybrid, y_test_hybrid
    else:
        return results_df, models_dict, best_model_name, X_test_scaled, y_test_clean

def plot_model_comparison(results_df):
    """Visualisiere Modell-Ergebnisse"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    results_df.sort_values('F1').plot(x='Modell', y='F1', kind='barh', ax=axes[0], legend=False, color='steelblue')
    axes[0].set_title('F1-Score')
    axes[0].set_xlabel('F1')
    
    results_df.sort_values('AUC').plot(x='Modell', y='AUC', kind='barh', ax=axes[1], legend=False, color='coral')
    axes[1].set_title('AUC-Score')
    axes[1].set_xlabel('AUC')
    
    results_df.sort_values('Acc').plot(x='Modell', y='Acc', kind='barh', ax=axes[2], legend=False, color='seagreen')
    axes[2].set_title('Accuracy')
    axes[2].set_xlabel('Acc')
    
    plt.tight_layout()
    plt.savefig(f'Model_comparison/model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
    print("✓ Plot gespeichert")
    plt.show()

def train_lstm(X_train, y_train, X_test, y_test, sequence_length=5):
    """Trainiere LSTM-Modell"""
    # Daten für LSTM vorbereiten
    X_train_lstm = np.zeros((X_train.shape[0] - sequence_length, sequence_length, X_train.shape[1]))
    X_test_lstm = np.zeros((X_test.shape[0] - sequence_length, sequence_length, X_test.shape[1]))
    y_train_lstm = np.zeros(X_train.shape[0] - sequence_length)
    y_test_lstm = np.zeros(X_test.shape[0] - sequence_length)
    
    for i in range(X_train.shape[0] - sequence_length):
        X_train_lstm[i] = X_train[i:i+sequence_length]
        y_train_lstm[i] = y_train[i+sequence_length]
    
    for i in range(X_test.shape[0] - sequence_length):
        X_test_lstm[i] = X_test[i:i+sequence_length]
        y_test_lstm[i] = y_test[i+sequence_length]
    
    # LSTM-Modell definieren
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Training
    model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_test_lstm, y_test_lstm),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, X_test_lstm, y_test_lstm

def train_transformer(X_train, y_train, X_test, y_test, sequence_length=5):
    """Trainiere Transformer-Modell"""
    
    # Daten für Transformer vorbereiten
    X_train_transformer = np.zeros((X_train.shape[0] - sequence_length, sequence_length, X_train.shape[1]))
    X_test_transformer = np.zeros((X_test.shape[0] - sequence_length, sequence_length, X_test.shape[1]))
    y_train_transformer = np.zeros(X_train.shape[0] - sequence_length)
    y_test_transformer = np.zeros(X_test.shape[0] - sequence_length)
    
    for i in range(X_train.shape[0] - sequence_length):
        X_train_transformer[i] = X_train[i:i+sequence_length]
        y_train_transformer[i] = y_train[i+sequence_length]
    
    for i in range(X_test.shape[0] - sequence_length):
        X_test_transformer[i] = X_test[i:i+sequence_length]
        y_test_transformer[i] = y_test[i+sequence_length]
    
    # Transformer-Modell mit Functional API definieren
    inputs = Input(shape=(sequence_length, X_train.shape[1]))
    
    # Attention Layer
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Dense Layers
    x = Dense(128, activation='relu')(attention)
    x = Dense(64, activation='relu')(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Training
    model.fit(
        X_train_transformer, y_train_transformer,
        validation_data=(X_test_transformer, y_test_transformer),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, X_test_transformer, y_test_transformer

def train_tabnet(X_train, y_train, X_test, y_test):

    tab = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5, gamma=1.5,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax"
    )

    tab.fit(
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train),
        eval_set=[(np.asarray(X_test), np.asarray(y_test))],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    return tab

#TODO: kann man noch integrieren - hat anfänglich nicht geklappt
def calculate_lstm_importance(model, X_test, y_test, feature_names):
    """Berechne Feature Importance für LSTM mit Permutation Importance pro Zeitschritt"""
    # Vorhersagen machen
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Feature Importance pro Zeitschritt berechnen
    importances = np.zeros((X_test.shape[1], X_test.shape[2]))
    for i in range(X_test.shape[1]):
        for j in range(X_test.shape[2]):
            X_test_perm = X_test.copy()
            X_test_perm[:, i, j] = np.random.permutation(X_test[:, i, j])
            y_pred_perm = (model.predict(X_test_perm) > 0.5).astype(int).flatten()
            importances[i, j] = accuracy_score(y_test, y_pred) - accuracy_score(y_test, y_pred_perm)
    
    # Ergebnis als Series zurückgeben
    importance_series = pd.Series(
        importances.mean(axis=0), index=feature_names
    ).sort_values(ascending=False)
    
    return importance_series

def predict_tabnet_single(model, X_single_row, n_dummy_rows=31):
    """
    Vorhersage für eine einzelne Zeile mit TabNet.
    Fügt Dummy-Zeilen hinzu, um die interne Batch-Size zu erfüllen.
    """
    # X_single_row zu numpy konvertieren
    X_single = np.asarray(X_single_row)
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    # Padding: Wiederhole die Zeile n_dummy_rows mal
    X_padded = np.vstack([X_single] * (n_dummy_rows + 1))
    
    # Vorhersage
    try:
        proba = model.predict_proba(X_padded)
        # Nur die erste Zeile zurückgeben
        return proba[0, 1]  # Wahrscheinlichkeit für Klasse 1
    except Exception as e:
        print(f"[ERROR] TabNet Vorhersage fehlgeschlagen: {str(e)[:80]}")
        return 0.5  # Fallback: neutral

def make_sequences(X_2d, y_1d, sequence_length=5):
    """Erstelle 3D-Sequenzen aus 2D-Daten für LSTM/CNN/Hybrid"""
    X_2d = np.asarray(X_2d)
    y_1d = np.asarray(y_1d)
    
    n = X_2d.shape[0]
    X_seq = np.zeros((n - sequence_length, sequence_length, X_2d.shape[1]), dtype=np.float32)
    y_seq = np.zeros((n - sequence_length,), dtype=np.int64)
    
    for i in range(n - sequence_length):
        X_seq[i] = X_2d[i:i+sequence_length]
        y_seq[i] = y_1d[i+sequence_length]
    
    return X_seq, y_seq

def train_cnn1d(X_train_2d, y_train, X_test_2d, y_test, sequence_length=5):
    """Trainiere 1D-CNN Modell für Zeitreihen"""
    
    # Sequenzen erstellen
    X_train_seq, y_train_seq = make_sequences(X_train_2d, y_train, sequence_length)
    X_test_seq, y_test_seq = make_sequences(X_test_2d, y_test, sequence_length)
    
    # Modell definieren
    inp = Input(shape=(sequence_length, X_train_2d.shape[1]))
    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = Flatten()(x)
    out = Dense(1, activation="sigmoid")(x)
    
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    
    # Training mit Early Stopping
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq),
              epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
    
    return model, X_test_seq, y_test_seq

def train_hybrid_lstm_transformer(X_train_2d, y_train, X_test_2d, y_test, sequence_length=5):
    """Trainiere Hybrid LSTM+Transformer Modell"""
    
    # Sequenzen erstellen
    X_train_seq, y_train_seq = make_sequences(X_train_2d, y_train, sequence_length)
    X_test_seq, y_test_seq = make_sequences(X_test_2d, y_test, sequence_length)
    
    # Modell definieren
    inp = Input(shape=(sequence_length, X_train_2d.shape[1]))
    
    # LSTM als Encoder
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    
    # Self-Attention (Transformer-Teil)
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn])  # Residual connection
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    
    # Training mit Early Stopping
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq),
              epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
    
    return model, X_test_seq, y_test_seq



###############################################################################
# Main
###############################################################################
def main():
    # Alle verfügbaren Aktienkurs-Dateien finden (optional mit _xx oder _xx Endungen; z.b. _US.csv)
    price_files = glob(str(PRICE_DIR / "*_??.csv")) + glob(str(PRICE_DIR / "*_???.csv"))

    # Initialisierung der panel_rows Liste
    panel_rows = []

    # Iteration durch alle gefundenen Aktienkurs-Dateien
    for pf in price_files:
        pf_path = Path(pf)

        # Dateienname ohne Suffix z.B. "AAPL_US"
        basename = pf_path.stem

        # Aufteilung des Dateiennamen und entfernen der Länderkennung z.B. APPL_US -> APPL
        if "_" in basename:
            ticker_candidate = "_".join(basename.split("_")[:-1])
        else:
            ticker_candidate = basename

        # Check ob Dateienname im Mapping (Aktienkurs-Unternehmenskennzahlen exisitiert)
        if ticker_candidate not in TICKER_MAP:
            # Falls nicht -> Warnung in der Commandozeile ausgeben
            print(f"[WARN] Kein Mapping für {ticker_candidate}, überspringe.")
            continue

        # Aus dem Mapping den richtigen Dateinamen der Unternehmenskennzahlen-Dateinamen finden
        fund_base = TICKER_MAP[ticker_candidate]

        # Aktienkursdatei (.csv) einlesen
        price_raw = pd.read_csv(pf_path, parse_dates=["Date"])
        # Groß-/Kleinschreibung anpassen
        price_raw = price_raw.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        # Liste aller benötigten Spalten aus den Aktienkurs-CSVs
        needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        # Überprüfen, ob alle Spalten verfügbar sind -> Falls nicht: Warnung auf der Kommandozeile ausgeben
        missing_cols = [c for c in needed_cols if c not in price_raw.columns]
        if missing_cols:
            print(f"[WARN] {pf_path.name} fehlt Spalten {missing_cols}, überspringe.")
            continue

        # Aufrufen der build_price_features-Funktion - laden und aufbereitung der Aktienkurs-Daten
        price_feat = build_price_features(price_raw, TARGET_HORIZON_DAYS, VOLATILITY, MOMENTUM, VOLUME)
        # Hinzufügen einer Spalte um die Daten einer Aktie zuordnen zu können
        price_feat["Ticker"] = ticker_candidate

        # Aufrufen der load_fundamentals_for_ticker-Funktion - laden und aufbereitung der Fundamentals-Daten
        fund_df = load_fundamentals_for_ticker(fund_base, FUND_DIR)
        # Aufrufen der merge_price_and_fundamentals-Funktion - Zusammenführen der beiden Datasets
        merged = merge_price_and_fundamentals(price_feat, fund_df)

        # Hinzufügen des Datasets zur Liste fürs spätere trainieren 
        panel_rows.append(merged)

    # Überprüfen. ob die Liste nicht leer ist
    if len(panel_rows) == 0:
        raise RuntimeError("Kein Eintrag in der Trainingsliste gefunden! - Mapping prüfen!")

    # Zusammenführen aller Datasets zu einem (untereinander-merge) inkl. Neuindexierung
    full_panel = pd.concat(panel_rows, ignore_index=True)

    # Featureliste fürs Modell
    FEATURE_COLUMNS = [
        # Aktienkurs Features
        "SMA_ratio",
        "Volatility",
        "Momentum",
        "VolumeChange",

        # Fundamentale Features
        "DebtEquity",
        "Leverage",
        "ProfitMargin",
        "OperatingMargin",
    ]


    # Umwandeln der Akitennamen in Kateogiern für die Verarbeitung mit LightGBM
    full_panel["Ticker_cat"] = full_panel["Ticker"].astype("category").cat.codes
    # Erstellen einer Liste mit allen wichtigen Informationen für das Modell
    FEATURE_COLUMNS_WITH_TICKER = FEATURE_COLUMNS + ["Ticker_cat"]

    # Erstellen eines neuen Datasets mit nur den wichtigen Spalten + alle NaN werden bereinigt - 
    # in der TARGET_HORIZON_DAYS-Zeitspanne kann kein Return_fwd und Target berechnet werden - daher nicht brauchbar für Training und Test
    # aber später für die Vorhersage braucht man keines von beiden!!! -> daher am Schluss wieder full_panel verwenden
    model_df = full_panel.dropna(
        subset=FEATURE_COLUMNS_WITH_TICKER + ["Target", "Return_fwd", "Date"]
    ).copy()

    # Aufrufen der time_based_train_test_split-Funktion - Aufteilen des Datasets in ein Training- und ein Test-Datasets
    train_df, test_df = time_based_train_test_split(model_df, TEST_SPLIT_RATIO)

    # Selektieren der wichtigsten Daten fürs Trainieren ohne Datum, Return_fwd und Target Spalte
    X_train = train_df[FEATURE_COLUMNS_WITH_TICKER]
    # Verwenden der Target Spalte als Ergebnis
    y_train = train_df["Target"]

    # Selektieren der wichtigsten Daten fürs Testen ohne Datum, Return_fwd und Target Spalte
    X_test = test_df[FEATURE_COLUMNS_WITH_TICKER]
    # Verwenden der Target Spalte als Ergebnis
    y_test = test_df["Target"]

    # LightGBM Klassifier konfigurieren
    '''
    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    # Modell mit Trainingsdaten füttern
    clf.fit(X_train, y_train)

    # Aufruf der evaluate_model-Funktion - Modell-Metriken, Mini-Backtest
    evaluate_model(
        clf,
        X_train, y_train,
        X_test, y_test,
        df_test_full=test_df,
        proba_threshold=PROBA_THRESHOLD,
        horizon_days=TARGET_HORIZON_DAYS,
    )
    '''

        # ===== MODELL-VERGLEICH STARTEN =====
    print("\n" + "="*80)
    print("🚀 STARTE MODELL-VERGLEICH")
    print("="*80)
    
    comparison_results, trained_models, best_model_name, X_test_lstm, y_test_lstm  = compare_models_simple(
        X_train, y_train,
        X_test, y_test,
        test_df,
        proba_threshold=PROBA_THRESHOLD,
        horizon_days=TARGET_HORIZON_DAYS,
        columns=FEATURE_COLUMNS_WITH_TICKER
    )
    
    # Visualisiere Ergebnisse
    plot_model_comparison(comparison_results)
    
    # Mit BESTEM MODELL evaluieren & Backtest
    print(f"\n📊 Evaluiere nun mit bestem Modell: {best_model_name}")
    
    # Modell aus Dictionary holen
    model_name_key = best_model_name.lower().replace(' ', '_').replace('lightgbm', 'lgb').replace('xgboost', 'xgb').replace('catboost', 'cat').replace('random_forest', 'rf').replace('stacking_ensemble', 'stack').replace('hybrid_lstm+transformer', 'hybrid').replace('1d-cnn', 'cnn')
    clf = trained_models.get(model_name_key)

    '''
    clf = trained_models[
        best_model_name.lower()
        .replace(' ', '_')
        .replace('lightgbm', 'lgb')
        .replace('xgboost', 'xgb')
        .replace('catboost', 'cat')
        .replace('random_forest', 'rf')
        .replace('stacking_ensemble', 'stack')
    ]  
    '''
    if clf is None:
        print(f"[WARNING] Modell '{best_model_name}' nicht in trained_models gefunden.")
        print(f"[INFO] Verfügbare Modelle: {list(trained_models.keys())}")
        # Fallback: Nutze das erste verfügbare Modell
        #clf = list(trained_models.values())[0]
        #best_model_name = list(trained_models.keys())[0]
        #print(f"[INFO] Nutze Fallback: {best_model_name}")
    elif best_model_name == "TabNet":
        evaluate_model(
            clf,
            X_train, y_train,
            X_test, y_test,
            test_df,
            proba_threshold=PROBA_THRESHOLD,
            horizon_days=TARGET_HORIZON_DAYS,
            model_type="tabnet"
        )
    elif best_model_name == "LSTM" or best_model_name == "Transformer":
        evaluate_model(
            clf,
            X_train, y_train,
            X_test_lstm, y_test_lstm,
            test_df,
            proba_threshold=PROBA_THRESHOLD,
            horizon_days=TARGET_HORIZON_DAYS,
            model_type="lstm"
        )
    elif best_model_name == "Hybrid LSTM+Transformer":
        evaluate_model(
            clf,
            X_train, y_train,
            X_test_lstm, y_test_lstm,
            test_df,
            proba_threshold=PROBA_THRESHOLD,
            horizon_days=TARGET_HORIZON_DAYS,
            model_type="hybrid"
        )
    elif best_model_name == "1D-CNN":
        evaluate_model(
            clf,
            X_train, y_train,
            X_test_lstm, y_test_lstm,
            test_df,
            proba_threshold=PROBA_THRESHOLD,
            horizon_days=TARGET_HORIZON_DAYS,
            model_type="cnn"
        )
    else:
        evaluate_model(
            clf,
            X_train, y_train,
            X_test, y_test,
            test_df,
            proba_threshold=PROBA_THRESHOLD,
            horizon_days=TARGET_HORIZON_DAYS,
            model_type="sklearn"
        )     

    # Anzeigen, wie wichtig ein Feature/Datenspalte für das Modell war 
    # und wieviel es zur Verbesserung des Modells beigetragen hat
    if best_model_name == "LSTM" or best_model_name == "Transformer" or best_model_name == "1D-CNN" or best_model_name == "Hybrid LSTM+Transformer":
        print("\n=== Feature Importances ===")
        print("Feature Importance ist für Deep Learning Modelle nicht verfügbar.")
    else:
        importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS_WITH_TICKER)
        print("\n=== Feature Importances (globales Modell) ===")
        print(importances.sort_values(ascending=False))
    
        print("\n=== Feature Importances (globales Modell) ===")
        print(importances.sort_values(ascending=False))

        with open("Klassifikationsreport/"+REPORT_FILE_NAME, "a", encoding="utf-8") as f:
            print("\n=== Feature Importances (globales Modell) ===", file=f)
            print(importances.sort_values(ascending=False), file=f)

        # Plot, welches Feature wie wichtig für das Trainieren des Modells war
        plt.figure(figsize=(7,4))
        importances.sort_values().plot(kind="barh")
        plt.title("LightGBM - Feature Importance (alle Aktien)")
        plt.tight_layout()
        plt.show()

    # Erstellen einer Liste, wo alle Empfehlungen und Wahrscheinlichkeiten pro Aktie gespeichert werden 
    latest_signals_tst = []
    # Iteration durch alle Aktien einzeln
    for ticker_name, grp in full_panel.groupby("Ticker"):
        # Sortieren der vorhandenen Daten nach Datum + Abspeichern der letzten Zeile mit dem aktuellsten Datum
        #last_row = grp.sort_values("Date").iloc[[-1]]

        grp = grp.sort_values("Date").reset_index(drop=True)

        if best_model_name == "LSTM" or best_model_name == "Transformer" or best_model_name == "1D-CNN" or best_model_name == "Hybrid LSTM+Transformer":
            # Für Deep Learning Modelle: Rolling Window erstellen
            sequence_length = TARGET_HORIZON_DAYS
            
            # Überprüfen, ob genug Daten vorhanden sind
            if len(grp) < sequence_length:
                print(f"[INFO] {ticker_name}: Nicht genug Daten für Rolling Window ({len(grp)} < {sequence_length})")
                continue
            
            # Letzte `sequence_length` Zeilen extrahieren
            last_window = grp[FEATURE_COLUMNS_WITH_TICKER].iloc[-sequence_length:].values
            
            # Zu 3D-Array umformen: (1, sequence_length, num_features)
            last_window_3d = np.expand_dims(last_window, axis=0)
            
            # Vorhersage machen
            proba_up = clf.predict(last_window_3d, verbose=0)[0][0]
        
        elif best_model_name == "TabNet":
            # TabNet: Single-Row Vorhersage mit Padding
            last_row = grp[FEATURE_COLUMNS_WITH_TICKER].iloc[-1:].values
            proba_up = predict_tabnet_single(clf, last_row)
        
        else:
            # Für sklearn-Modelle: letzte Zeile nehmen
            last_row = grp[FEATURE_COLUMNS_WITH_TICKER].iloc[[-1]]
            proba_up = clf.predict_proba(last_row)[:, 1][0]


        # Berechnung der Wahrscheinlichkeit, ob die Aktie steigt (1) oder nicht (0) - 2D-Array hat nur einen Eintrag
        #proba_up = clf.predict_proba(last_row[FEATURE_COLUMNS_WITH_TICKER])[:, 1][0]
        # Vorhersage ob es steigen wird (1) oder nicht (0) 
        #pred_up = clf.predict(last_row[FEATURE_COLUMNS_WITH_TICKER])[0]
        # Speichern, der Emfpehlungen in die Liste
        latest_signals_tst.append({
            "Ticker": ticker_name,
            "Date": grp["Date"].iloc[-1],
            "ProbUp": proba_up,
            #"Signal": "UP" if pred_up == 1 else "DOWN"
            "Signal": "UP" if proba_up > PROBA_THRESHOLD else "DOWN"
        })        
   
    # Speichern der gesamte List in einem Dataset + Sotierung nach Wahrscheinlichkeit
    latest_signals_tst_df = pd.DataFrame(latest_signals_tst).sort_values("ProbUp", ascending=False)
    # Ausgabe des Ergebnis
    print("\n=== Aktuelle Einschätzung TST(pro Ticker, letztes Datum) ===")
    print(latest_signals_tst_df.to_string(index=False))
    # Speichern als JSON
    latest_signals_tst_df.to_json("results.json", orient="records", indent=4)
           
        
# Call der Main-Funktion zum Starten des Skripts
if __name__ == "__main__":
    main()
