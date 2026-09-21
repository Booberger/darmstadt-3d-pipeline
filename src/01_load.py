"""
01_load.py
----------
Schritt 1 der Pipeline: LAZ-Datei erkunden und einen räumlichen Ausschnitt extrahieren.

Eingabe:  data/raw/fullLiWi_pointcloud.laz  (vollstaendige Punktwolke, ~50 GB)
Ausgabe:  data/processed/ausschnitt_100m.laz (100m x 100m Ausschnitt)

Besonderheit: Die Datei wird chunkweise verarbeitet (100.000 Punkte pro Iteration)
um Memory Errors bei der grossen Eingabedatei zu vermeiden. Gefilterte Punkte
werden direkt in die Ausgabedatei geschrieben ohne sie im RAM zu akkumulieren.
"""

import laspy
import numpy as np
import os

# Konfiguration 
INPUT_FILE       = "data/raw/fullLiWi_pointcloud.laz"  # Eingabedatei
OUTPUT_FILE      = "data/processed/ausschnitt_100m.laz" # Ausgabedatei
AUSSCHNITT_METER = 100    # Seitenlänge des quadratischen Ausschnitts in Metern
CHUNK_SIZE       = 100_000 # Anzahl Punkte pro Lesevorgang. War zuerst 500_000, aber das war zu viel für den RAM.

# Schritt 1: Metadaten lesen 
# Die Metadaten (header) enthalten wichtige Informationen über den Datensatz und können ohne vollständiges Laden der Datei gelesen werden.
print("SCHRITT 1: Datei-Informationen")
print("-" * 60)

with laspy.open(INPUT_FILE) as f:
    header = f.header

    print(f"Dateigröße:  {os.path.getsize(INPUT_FILE) / 1e9:.2f} GB")
    print(f"Punkte:      {header.point_count:,}")
    print(f"X: {header.x_min:.2f} - {header.x_max:.2f}  ({header.x_max - header.x_min:.0f} m)")
    print(f"Y: {header.y_min:.2f} - {header.y_max:.2f}  ({header.y_max - header.y_min:.0f} m)")
    print(f"Z: {header.z_min:.2f} - {header.z_max:.2f}  ({header.z_max - header.z_min:.0f} m)")

    # Ausschnitt aus der Mitte der Punktwolke berechnen
    x_mid = (header.x_min + header.x_max) / 2
    y_mid = (header.y_min + header.y_max) / 2
    half  = AUSSCHNITT_METER / 2

    x_min = x_mid - half
    x_max = x_mid + half
    y_min = y_mid - half
    y_max = y_mid + half

print(f"\nAusschnitt ({AUSSCHNITT_METER}m x {AUSSCHNITT_METER}m):")
print(f"  X: {x_min:.0f} - {x_max:.0f}")
print(f"  Y: {y_min:.0f} - {y_max:.0f}")

# ── Schritt 2: Ausschnitt extrahieren ─────────────────────────────────────────
# Die Eingabedatei wird in Blöcken von CHUNK_SIZE Punkten gelesen
# Für jeden Block wird geprüft ob die Punkte im gewünschten Ausschnitt liegen
# Passende Punkte werden direkt in die Ausgabedatei geschrieben
print("\nSCHRITT 2: Ausschnitt extrahieren")
print("-" * 60)

os.makedirs("data/processed", exist_ok=True)

punkte_gesamt = 0
chunk_nr      = 0

with laspy.open(INPUT_FILE) as reader:
    with open(OUTPUT_FILE, "wb") as out_file:

        # LasWriter schreibt direkt in den Datei-Stream
        # do_compress=True erzeugt eine komprimierte LAZ-Datei
        writer = laspy.LasWriter(out_file, header=reader.header, do_compress=True)

        for chunk in reader.chunk_iterator(CHUNK_SIZE):
            chunk_nr += 1

            # Boolsche Maske: True für Punkte innerhalb des Ausschnitts
            maske = (
                (chunk.x >= x_min) & (chunk.x <= x_max) &
                (chunk.y >= y_min) & (chunk.y <= y_max)
            )

            # Nur gefilterte Punkte in die Ausgabedatei schreiben
            anzahl = int(np.sum(maske))
            if anzahl > 0:
                writer.write_points(chunk[maske])
                punkte_gesamt += anzahl

            # Fortschritt alle 500 Chunks ausgeben
            if chunk_nr % 500 == 0:
                prozent = (chunk_nr * CHUNK_SIZE / reader.header.point_count) * 100
                print(f"  {prozent:.1f}% verarbeitet | {punkte_gesamt:,} Punkte gefunden")

        writer.close()

# Ergebnis ausgeben 
print(f"\nPunkte im Ausschnitt: {punkte_gesamt:,}")

if punkte_gesamt > 0:
    dichte = punkte_gesamt / (AUSSCHNITT_METER ** 2)
    print(f"Punktdichte:          {dichte:.1f} Punkte/m²")
    print(f"Gespeichert:          {OUTPUT_FILE}")
    print(f"Dateigröße:           {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
else:
    print("FEHLER: Keine Punkte im Ausschnitt gefunden.")
    print("Prüfe ob die Eingabedatei korrekt ist und der Ausschnitt im Koordinatenbereich liegt.")

print("\nFERTIG - Weiter mit 02_preprocess.py")
print("-" * 60)