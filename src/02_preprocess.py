"""
02_preprocess.py
----------------
Schritt 2 der Pipeline: Vorverarbeitung der Punktwolke.

Eingabe:  data/processed/ausschnitt_100m.laz
Ausgabe:  data/processed/preprocessed.npy  (Punkte, Normalen, Farben, Features)
          data/processed/preprocessed.ply  (für Visualisierung in MeshLab)

Verarbeitungsschrittte:
  1. LAZ chunkweise einlesen und RGB extrahieren
  2. Koordinaten normalisieren (Schwerpunkt auf Ursprung verschieben)
  3. Rauschfilterung via Statistical Outlier Removal
  4. Downsampling via Voxel-Grid-Filter
  5. Normalen schätzen und orientierenn
  6. Als NPY und PLY speichern
"""

import laspy
import open3d as o3d
import numpy as np
import os

# Konfiguration 
INPUT_FILE  = "data/processed/ausschnitt_100m.laz"
OUTPUT_FILE = "data/processed/preprocessed.npy"

VOXEL_SIZE        = 0.25  # Voxelgröße für Downsampling in Metern
OUTLIER_NEIGHBORS = 20    # Anzahl Nachbarn für Statistical Outlier Removal
OUTLIER_STD_RATIO = 2.0   # Standardabweichungs-Schwellenwert für Ausreißer

# Schritt 1: LAZ einlesen 
# Die LAZ-Datei wird chunkweise eingelesen (je 5 Mio Punkte)
# um Memory Errors bei der großsen Eingabedatei zu vermeiden
# RGB wird von 16-Bit (0-65535) auf den Bereich 0-1 normiert
# Intensitätswerte werden mitgeführt aber nicht als Feature verwendeet
print("SCHRITT 1: Punktwolke einlesen")
print("-" * 60)
print("Lese LAZ-Datei..")

xyz_list       = []
rgb_list       = []
intensity_list = []

CHUNK_SIZE = 5_000_000

with laspy.open(INPUT_FILE) as f:
    total   = f.header.point_count
    gelesen = 0
    hat_rgb = all(hasattr(f.header.point_format, dim)
                  for dim in ['red', 'green', 'blue'])

    for chunk in f.chunk_iterator(CHUNK_SIZE):

        # XYZ-Koordinaten extrahieren
        xyz_list.append(np.column_stack([
            chunk.x.copy(),
            chunk.y.copy(),
            chunk.z.copy()
        ]))

        # RGB-Farbwerte extrahieren und auf 0-1 norrmieren
        # Falls kein RGB vorhanden: Nullvektor Platzhalter
        if hasattr(chunk, 'red'):
            rgb_list.append(np.column_stack([
                np.array(chunk.red,   dtype=np.float32) / 65535.0,
                np.array(chunk.green, dtype=np.float32) / 65535.0,
                np.array(chunk.blue,  dtype=np.float32) / 65535.0,
            ]))
        else:
            rgb_list.append(np.zeros((len(chunk.x), 3), dtype=np.float32))

        # Intensitätswerte speichern (werden gespeichert aber nicht als Feature genutzt)
        intensity_list.append(np.array(chunk.intensity, dtype=np.float32))

        gelesen += len(chunk.x)
        print(f"  {gelesen/total*100:.1f}% gelesen ({gelesen:,} Punkte)")

# Alle Chunks zu einem Array. Zwischenlisten löscshen
xyz       = np.vstack(xyz_list);            del xyz_list
rgb       = np.vstack(rgb_list);            del rgb_list
intensity = np.concatenate(intensity_list); del intensity_list

print(f"\nPunkte geladen: {len(xyz):,}")
print(f"RAM XYZ: {xyz.nbytes/1e6:.1f} MB")
print(f"RAM RGB: {rgb.nbytes/1e6:.1f} MB")

# Schritt 2: Koordinaten normalisieren 
# Der geometrische Schwerpunkt der Punktwolke wird berechnet und alle Koordinaten werden so verschoben dass der Mittelpunkt im Ursprung liegt
# Dies ist notwendig da der Random Forest auf lokalen Koordinaten operiert und nicht auf absoluten Georeferenzierungskoordinaten
# Der ursprüngliche Mittelpunkt wird gespeihcert für spätere Georeferenzierung
print("\nSCHRITT 2: Koordinaten normalisieren")
print("-" * 60)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

center = pcd.get_center()
pcd.translate(-center)
print(f"Koordinaten zentriert. Urspruenglicher Mittelpunkt: {center}")
print(f"Neuer Mittelpunkt: {pcd.get_center()}")

# Schritt 3: Rauschfilterung 
# Statistical Outlier Removal berechnet für jeden Punkt den mittleren Abstand
# zu seinen k nächsten Nachbarn. Punkte deren mittlerer Abstand mehr als OUTLIER_STD_RATIO Standardabweichungen vom globalen Mittelwert abweicht werden als Ausreisser entfernt.
# Der Indexvector ind wird gespeichert um das Intensitäts-Array entsprechend zu reduzieren und Konsistenz der Arrays zu erhalten.
print("\nSCHRITT 3: Rauschfilterung (Statistical Outlier Removal)")
print("-" * 60)

vorher = len(pcd.points)
pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=OUTLIER_NEIGHBORS,
    std_ratio=OUTLIER_STD_RATIO
)
nachher = len(pcd.points)

# Intensitäts-Array auf gefilterte Punkte reduzieren
intensity = intensity[ind]

print(f"Vor Filterung:  {vorher:,} Punkte")
print(f"Nach Filterung: {nachher:,} Punkte")
print(f"Entfernt:       {vorher - nachher:,} Ausreisser ({(vorher-nachher)/vorher*100:.2f}%)")

# Schritt 4: Downsampling 
# Der Voxel-Grid-Filter teilt den 3D-Raum in Wuerfel der Kantenlänge VOXEL_SIZE und ersetzt alle Punkte innerhalb eines Voxels durch den Schwerpunkt
# Dies reduziert die Punktanzahl stark
print(f"\nSCHRITT 4: Voxel-Grid-Downsampling (Voxelgroesse: {VOXEL_SIZE}m)")
print("-" * 60)

vorher = len(pcd.points)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
nachher = len(pcd.points)

print(f"Vor Downsampling:  {vorher:,} Punkte")
print(f"Nach Downsampling: {nachher:,} Punkte")
print(f"Reduktion: {(1 - nachher/vorher)*100:.1f}%")

# Schritt 5: Normalen schätzen
# Für jeden Punkt werden Oberflaechennormalen aus der lokalen Punktnachbarschaft geschäötzt 
# (Radius 1.0m, max. 30 Nachbarn)
# orient_normals_consistent_tangent_plane orientiert alle Normalen konsistent
# Normalen werden als Feature für die Mesh-Rekonstruktion benötigt
print("\nSCHRITT 5: Normalen schaetzen")
print("-" * 60)

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
)
# Normalen konsistent orientieren entlang der Tangentialebene
pcd.orient_normals_consistent_tangent_plane(30)
print("Normalen geschaetzt und nach oben orientiert.")

# Schritt 6: Ergebnis speichern 
# Die vorverarbeitete Punktwolke wird als NPY-Dictionary gespeichert mit:
#   points:   XYZ-Koordinaten (normalisiert)
#   normals:  Oberflächennormalen
#   colors:   RGB-Farbwerte (0-1)
#   features: 7 Features fuer Random Forest (XYZ + RGB + VDVI)
#   center:   Ursprüpnglicher Schwerpunkt für spätere Georeferenzierung
# Zusätzlich wird eine PLY-Datei fr die Visualisierung in MeshLab gespeichert.
print("\nSCHRITT 6: Ergebnis speichern")
print("-" * 60)

os.makedirs("data/processed", exist_ok=True)

points  = np.asarray(pcd.points,  dtype=np.float32)
normals = np.asarray(pcd.normals, dtype=np.float32)
colors  = np.asarray(pcd.colors,  dtype=np.float32)

# VDVI (Visible-band Difference Vegetation Index) berechnen
# Formel nach Oniga et al. (2022): (2G - R - B) / (2G + R + B)
# Unterscheidet Vegetation (hoher VDVI) von Gebäuden und Strasßn (niedriger VDVI)
r = colors[:, 0:1]
g = colors[:, 1:2]
b = colors[:, 2:3]
nenner = 2*g + r + b + 1e-8  # 1e-8 verhindert Division durch Null
vdvi   = (2*g - r - b) / nenner

# Feature-Vektor: 7 Features pro Punkt (XYZ + RGB + VDVI)
features = np.hstack([points, colors, vdvi])

daten = {
    "points":   points,   # (N, 3) normalisierte XYZ-Koordinaten
    "normals":  normals,  # (N, 3) Oberflaechennormalen
    "colors":   colors,   # (N, 3) RGB-Farbwerte
    "features": features, # (N, 7) Feature-Vektor fuer Random Forest
    "center":   center    # (3,)   Urspruenglicher Schwerpunkt
}

np.save(OUTPUT_FILE, daten, allow_pickle=True)
print(f"Gespeichert: {OUTPUT_FILE}")
print(f"Finale Punktanzahl: {len(points):,}")
print(f"Features pro Punkt: {features.shape[1]}")
print(f"RGB gespeichert: {colors.shape}")

# PLY für Visualisierung in MeshLab
ply_file = "data/processed/preprocessed.ply"
o3d.io.write_point_cloud(ply_file, pcd)
print(f"PLY gespeichert: {ply_file}")

print("\nFERTIG - Weiter mit 03_segment_rf.py")
print("-" * 60)