# LLM Tabular Synthesis

## Projektziel

Dieses Projekt vergleicht synthetische tabellarische Datengenerierung unter Low-Data-Bedingungen. Der Kernvergleich umfasst CTGAN, Direct-ICL und Code-LLM-v2 auf Adult und Pima mit 100 bzw. 500 Trainingszeilen über drei Seeds.

Die gepatchte Version ergänzt gezielte Diagnose- und Ablationsauswertungen, damit finale Pipeline-Leistung, primäre Generatorleistung, Refill-/Fallback-Anteile, Utility, Fidelity, Laufzeit, einfache Privacy-Indikatoren und Fairness-Diagnostik klarer getrennt werden können.

## Experimentdesign der gepatchten Version

Das Hauptdesign bleibt der Vergleich der drei ursprünglichen Methoden:

- `ctgan`
- `direct_icl`
- `code_llm`

Dieses Hauptdesign umfasst weiterhin 36 Kernexperimente:

```text
2 Datensätze × 2 Trainingsgrößen × 3 Seeds × 3 Hauptmethoden = 36
```

Zusätzlich enthält die gepatchte Version zwei Diagnose-/Robustheitsmethoden:

- `fallback_only`: Baseline, die ausschließlich die reproduzierbare Ersatzlogik nutzt.
- `ctgan_strong`: CTGAN-Robustheitscheck mit erhöhter Epochzahl.

Damit umfasst die vollständige gepatchte Experimentmatrix 60 Ergebniszeilen:

```text
2 Datensätze × 2 Trainingsgrößen × 3 Seeds × 5 Methoden = 60
```

Die `direct_icl`-Runs enthalten zusätzlich Primary-only-Diagnostiken. Diese prüfen die validen primären LLM-Zeilen vor Refill/Fallback, sind aber keine eigene Run-Methode.

## Wichtige Interpretationsregel

Die finale Direct-ICL-Tabelle ist als Ergebnis der abgesicherten Gesamtpipeline zu interpretieren: LLM-Rohoutput, Parsing, Normalisierung, Validierung, Deduplizierung und gegebenenfalls Refill/Fallback wirken zusammen.

Die Primary-only-Diagnostik und die Fallback-only-Baseline sind deshalb besonders wichtig. Sie helfen zu prüfen, wie stark der finale Direct-ICL-Befund von primären LLM-Zeilen bzw. von der Ersatzlogik getragen wird.

Die gepatchte Auswertung sollte im Paper nicht als bloße Erweiterung der Methodenliste erzählt werden. Sinnvoll ist die Trennung in:

1. Hauptvergleich: CTGAN, Direct-ICL, Code-LLM-v2.
2. Zusatzdiagnostik: Direct-ICL Primary-only, Fallback-only, CTGAN-strong, Utility-Gap, Real-DPD-Referenzen und target-conditioned Fidelity.

## Finaler Ergebnisstand

Nach einem vollständigen finalen Run werden die Ergebnisse unter folgendem Pfad archiviert:

```text
results/runs/<RUN_ID>/
```

Die Top-Level-Tabellen unter `results/tables/` und die Plots unter `results/plots/` beziehen sich auf den zuletzt erfolgreich ausgeführten vollständigen Run. Vor der finalen Abgabe sollte `<RUN_ID>` im Paper und in begleitenden Hinweisen durch die konkrete Run-ID ersetzt werden.

## Methoden

- `ctgan`: klassische tabellarische CTGAN-Baseline mit kompakter Konfiguration.
- `ctgan_strong`: zusätzlicher CTGAN-Robustheitscheck mit erhöhter Epochzahl.
- `direct_icl`: lokale LLM-Erzeugung CSV-ähnlicher Tabellenzeilen. Der generierte Text wird geparst, normalisiert, validiert und bei Bedarf durch Refill-/Fallback-Mechanismen vervollständigt.
- `code_llm`: lokale LLM-Erzeugung von Sampling-Code im `Code-LLM-v2`-Scaffold. Die Programme unter `data/generated_code/` sind auditable Code-LLM-v2-Sampler aus dem finalen Run und nicht als gescheiterte Fallback-Programme zu interpretieren.
- `fallback_only`: Diagnosebaseline, die nur die reproduzierbare Ersatzlogik verwendet.

Ollama dient ausschließlich als lokale Inferenzumgebung für `llama3.2:3b` und wird nicht als eigenständige wissenschaftliche Methode bewertet.

## Lokale Voraussetzungen

- Python 3.11
- Ollama lokal erreichbar
- Modell `llama3.2:3b`
- `README.md` und `requirements.txt` im Projektroot
- verarbeitete Eingabedaten unter `data/processed/`

## Installation

```bash
cd grundprojekt_llm_tabular_synthesis
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name grundprojekt --display-name "Python (Grundprojekt)"
```

`requirements.txt` enthält die direkten, für den finalen Run relevanten Abhängigkeiten. Das Notebook prüft diese Datei nur und erzeugt oder überschreibt sie nicht. Der vollständige aufgelöste Environment-Snapshot des finalen Runs wird unter `results/runs/<RUN_ID>/pip_freeze.txt` archiviert.

## Ollama-Setup

```bash
ollama serve
ollama pull llama3.2:3b
curl http://localhost:11434/api/tags
```

Der letzte Befehl dient als schneller Erreichbarkeitstest. Für `direct_icl` und `code_llm` muss der Ollama-Server während des Experimentlaufs laufen.

## Notebook starten

```bash
cd grundprojekt_llm_tabular_synthesis
source .venv/bin/activate
jupyter lab
```

Dann `llm_tabular_synthesis_abgabe_final.ipynb` mit dem Kernel `Python (Grundprojekt)` öffnen.

## Reproduction entry points

- `main()` bzw. der Preflight-Einstieg startet standardmäßig keinen vollständigen Experimentlauf und keinen Smoke-Test.
- Ein vollständiger Lauf wird nur explizit mit `run_full=True` gestartet.
- Die archivierten Ergebnisse befinden sich unter `results/runs/<RUN_ID>/`; der Environment-Snapshot liegt dort als `environment_snapshot.json` und `pip_freeze.txt`.
- Die Reproduktion startet bei den processed CSVs unter `data/processed/`; Rohdaten-Preprocessing ist nicht Teil dieses Pakets.
- Die semantische Plausibilitätsdiagnostik in `results/tables/semantic_plausibility.csv` ist post-hoc aus den finalen synthetischen CSVs berechnet.

## Sichere Vorprüfung ohne vollständigen Experimentlauf

Die gepatchte Version soll beim normalen Ausführen nicht automatisch den kompletten LLM/CTGAN-Lauf starten. Für eine reine Input-, Split- und Konfigurationsprüfung:

```python
main()
```

Dieser Aufruf validiert die Projektstruktur und gibt Hinweise zum Preflight aus, führt aber keine Experimente aus.

## Vollständiger finaler Run

Vor dem Full-Run muss Ollama laufen und das Modell verfügbar sein.

```python
main(run_full=True, run_smoke=True, clean_outputs=True)
```

Dieser Aufruf führt die gepatchte Experimentmatrix aus, erzeugt die Ergebnistabellen und Plots, validiert die Ausgabe und erstellt anschließend die Abgabe-ZIP.

Falls nur ein Smoke-Test gewünscht ist:

```python
main(run_full=False, run_smoke=True)
```

## Erwartete Ergebnisdateien

Zentrale Tabellen:

- `results/tables/full_experiment_results.csv`
- `results/tables/main_results.csv`
- `results/tables/summary_by_method_dataset_train_size.csv`
- `results/tables/provenance_ablation_summary.csv`

Zentrale Plots:

- `results/plots/tstr_f1.png`
- `results/plots/fidelity_distance.png`
- `results/plots/constraint_violation_rate.png`
- `results/plots/runtime_seconds.png`
- `results/plots/adult_demographic_parity_difference.png`

Audit- und Provenienzartefakte pro finalem Run:

- `results/runs/<RUN_ID>/environment_snapshot.json`
- `results/runs/<RUN_ID>/config_snapshot.json`
- `results/runs/<RUN_ID>/pip_freeze.txt`
- `results/runs/<RUN_ID>/artifact_manifest.json`
- `results/runs/<RUN_ID>/synthetic_final/`
- `results/runs/<RUN_ID>/row_provenance/`
- `results/runs/<RUN_ID>/run_results/`
- `results/runs/<RUN_ID>/raw_llm/`
- `results/runs/<RUN_ID>/code_llm_traces/`

## Hinweise zur Ergebnisauswertung

Besonders relevante Diagnosefelder sind unter anderem:

- `fallback_fraction`
- `repair_fraction`
- `raw_valid_rate`
- `primary_generator_rows`
- `primary_only_rows`
- `primary_only_tstr_f1`
- `primary_only_distribution_distance_mean`
- `tstr_f1_gap_to_real_train`
- `tstr_f1_relative_to_real_train`
- `target_conditioned_distribution_distance`
- `feature_target_mean_dependency_distance`
- `real_train_demographic_parity_difference_abs`
- `real_test_demographic_parity_difference_abs`

Diese Felder sollten im Paper vorsichtig interpretiert werden. Sie erlauben eine bessere Diagnose, ersetzen aber keine umfassende Unsicherheitsanalyse, keine vollständige Fairnessanalyse und keine formale Privacy-Garantie.

## Hinweise für die Paper-Aktualisierung

Nach dem vollständigen finalen Run sollten Paper und Tabellen konsistent aktualisiert werden:

- Nicht mehr nur von 36 vollständigen Experimenten sprechen, wenn die gepatchte 60-Zeilen-Ergebnistabelle berichtet wird.
- Die ursprünglichen 36 Experimente als Hauptvergleich darstellen.
- `fallback_only` und `ctgan_strong` als Zusatzdiagnostik bzw. Robustheitscheck einordnen.
- Direct-ICL weiterhin als finale Pipeline-Leistung formulieren, nicht als isolierte Rohgeneratorleistung.
- Primary-only-Ergebnisse und Fallback-only-Ergebnisse nur berichten, wenn sie aus dem finalen Run stammen.
- Privacy und Fairness weiterhin nur diagnostisch und vorsichtig formulieren.

## Hinweise zur Abgabe

Der ZIP-Root ist bewusst auf `Grundprojekt_Code/` gesetzt, damit nach dem Entpacken kein lokaler Kopie- oder Versionsordner entsteht. `README.md` und `requirements.txt` werden dabei nur übernommen, wenn sie im Projektroot vorhanden sind.

Die virtuelle Umgebung `.venv` ist nicht Teil der Abgabe. Das Modell `llama3.2:3b` wird lokal über Ollama benötigt und ist nicht in der Abgabe enthalten.
