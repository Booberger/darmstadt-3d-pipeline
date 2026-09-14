"""
run_pipeline.py
---------------
Ausfuehren der gesamten Pipeline in einem Schritt.
Stoppt bei Fehler in einem Schritt und gibt eine Fehlermeldung aus.

Verwendung:
    python run_pipeline.py
    python run_pipeline.py --skip-load      # 01_load.py ueberspringen
    python run_pipeline.py --skip-preprocess # 02_preprocess.py ueberspringen
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# ── Konfiguration ────────────────────────────────────────────────────────────
SCRIPTS = [
    ("01_load.py",        "Datenextraktion (LAZ → Ausschnitt)"),
    ("02_preprocess.py",  "Vorverarbeitung (Filterung, Downsampling, Normalen)"),
    ("03_segment_rf.py",  "Semantische Segmentierung (Random Forest)"),
    ("04_reconstruct.py", "Mesh-Rekonstruktion und Export (LOD0/1/2, glTF)"),
]

def ausfuehren(script_name, beschreibung):
    """Fuehrt ein Script aus und gibt Fehler aus wenn es fehlschlaegt."""
    script_path = Path("src") / script_name

    if not script_path.exists():
        print(f"\n[FEHLER] Script nicht gefunden: {script_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  {beschreibung}")
    print(f"  Script: {script_path}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False  # Ausgabe direkt anzeigen
    )
    dauer = time.time() - start

    if result.returncode != 0:
        print(f"\n[FEHLER] {script_name} ist fehlgeschlagen (Exit Code {result.returncode})")
        print(f"Bitte Fehler beheben und Pipeline neu starten.")
        sys.exit(result.returncode)

    print(f"\n[OK] {script_name} abgeschlossen in {dauer:.1f}s")
    return dauer

def main():
    parser = argparse.ArgumentParser(description="Darmstadt 3D Pipeline")
    parser.add_argument("--skip-load",       action="store_true", help="01_load.py ueberspringen")
    parser.add_argument("--skip-preprocess", action="store_true", help="02_preprocess.py ueberspringen")
    parser.add_argument("--skip-segment",    action="store_true", help="03_segment_rf.py ueberspringen")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  KI-gestuetzte 3D-Stadtmodell-Pipeline")
    print("  Fallbeispiel: Stadt Darmstadt")
    print("="*60)

    gesamt_start = time.time()
    dauern = {}

    for script_name, beschreibung in SCRIPTS:
        # Optionale Schritte ueberspringen
        if script_name == "01_load.py"       and args.skip_load:
            print(f"\n[SKIP] {script_name} wird uebersprungen.")
            continue
        if script_name == "02_preprocess.py" and args.skip_preprocess:
            print(f"\n[SKIP] {script_name} wird uebersprungen.")
            continue
        if script_name == "03_segment_rf.py" and args.skip_segment:
            print(f"\n[SKIP] {script_name} wird uebersprungen.")
            continue

        dauer = ausfuehren(script_name, beschreibung)
        dauern[script_name] = dauer

    gesamt = time.time() - gesamt_start

    print("\n" + "="*60)
    print("  PIPELINE ABGESCHLOSSEN")
    print("="*60)
    print(f"\n  Laufzeiten:")
    for script, dauer in dauern.items():
        print(f"    {script:25s}: {dauer:>8.1f}s")
    print(f"\n  Gesamt: {gesamt:.1f}s ({gesamt/60:.1f} min)")
    print(f"\n  Ausgaben in: data/output/")
    print("="*60)

if __name__ == "__main__":
    main()