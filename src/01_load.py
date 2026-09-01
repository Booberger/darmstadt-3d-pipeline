"""
01_load.py
----------
Schritt 1 der Pipeline: LAZ-Datei erkunden und einen kleinen Ausschnitt extrahieren.
Memory-effiziente Version: schreibt direkt in die Ausgabedatei ohne RAM anzusammeln.
"""

import laspy
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_FILE  = "data/raw/fullLiWi_pointcloud.laz"
OUTPUT_FILE = "data/processed/ausschnitt_100m.laz"

AUSSCHNITT_METER = 100   # 100m x 100m
CHUNK_SIZE       = 100_000  # 100k Punkte pro Chunk - sehr konservativ

# ── Schritt 1: Metadaten lesen ───────────────────────────────────────────────
print("=" * 60)
print("SCHRITT 1: Datei-Informationen")
print("=" * 60)

with laspy.open(INPUT_FILE) as f:
    header = f.header
    print(f"Dateigröße:  {os.path.getsize(INPUT_FILE) / 1e9:.2f} GB")
    print(f"Punkte:      {header.point_count:,}")
    print(f"X: {header.x_min:.2f} - {header.x_max:.2f}  ({header.x_max - header.x_min:.0f} m)")
    print(f"Y: {header.y_min:.2f} - {header.y_max:.2f}  ({header.y_max - header.y_min:.0f} m)")
    print(f"Z: {header.z_min:.2f} - {header.z_max:.2f}  ({header.z_max - header.z_min:.0f} m)")

    x_mid = (header.x_min + header.x_max) / 2
    y_mid = (header.y_min + header.y_max) / 2
    half  = AUSSCHNITT_METER / 2

    x_min = x_mid - half
    x_max = x_mid + half
    y_min = y_mid - half
    y_max = y_mid + half

    point_format = header.point_format
    file_version = header.version

print(f"\nAusschnitt: X {x_min:.0f}-{x_max:.0f}, Y {y_min:.0f}-{y_max:.0f}")

# ── Schritt 2: Ausschnitt direkt in Datei schreiben ──────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 2: Ausschnitt extrahieren (direkt schreiben)")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

punkte_gesamt = 0
chunk_nr = 0

with laspy.open(INPUT_FILE) as reader:
    with open(OUTPUT_FILE, "wb") as out_file:
        writer = laspy.LasWriter(
            out_file,
            header=reader.header,
            do_compress=True
        )
        for chunk in reader.chunk_iterator(CHUNK_SIZE):
            chunk_nr += 1

            # Debug: erste Chunk anzeigen
            if chunk_nr == 1:
                print(f"DEBUG - Erste Chunk X: {float(chunk.x.min()):.2f} - {float(chunk.x.max()):.2f}")
                print(f"DEBUG - Erste Chunk Y: {float(chunk.y.min()):.2f} - {float(chunk.y.max()):.2f}")

            maske = (
                (chunk.x >= x_min) & (chunk.x <= x_max) &
                (chunk.y >= y_min) & (chunk.y <= y_max)
            )
            anzahl = int(np.sum(maske))
            if anzahl > 0:
                writer.write_points(chunk[maske])
                punkte_gesamt += anzahl
            if chunk_nr % 500 == 0:
                prozent = (chunk_nr * CHUNK_SIZE / reader.header.point_count) * 100
                print(f"  {prozent:.1f}% | {punkte_gesamt:,} Punkte")
        writer.close()

print(f"\nFertig!")
print(f"Punkte im Ausschnitt: {punkte_gesamt:,}")
if punkte_gesamt > 0:
    dichte = punkte_gesamt / (AUSSCHNITT_METER ** 2)
    print(f"Punktdichte: {dichte:.1f} Punkte/m²")
    print(f"Gespeichert: {OUTPUT_FILE}")
    print(f"Dateigröße:  {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
else:
    print("FEHLER: Keine Punkte im Ausschnitt gefunden.")

print("\n" + "=" * 60)
print("FERTIG - Weiter mit 02_preprocess.py")
print("=" * 60)