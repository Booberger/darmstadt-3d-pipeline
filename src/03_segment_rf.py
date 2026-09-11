"""
03_segment_rf.py
----------------
Schritt 3 der Pipeline: Semantische Segmentierung mittels Random Forest.
- 4 Klassen: Boden, Vegetation, Gebaeude, Strasse (analog zu Ballouch 2024)
- Training auf annotierten SensatUrban-Bloecken
- Evaluation auf Test-Block (mIoU, F1, OA)
- Inferenz auf Darmstaedter Punktwolke
"""

import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Konfiguration
TRAIN_FILES = [
    "data/sensaturban/train/birmingham_block_1.ply",
    "data/sensaturban/train/birmingham_block_3.ply",
    "data/sensaturban/train/cambridge_block_2.ply",
    "data/sensaturban/train/cambridge_block_10.ply",
    "data/sensaturban/train/cambridge_block_26.ply",
]
#TEST_FILE      = "data/sensaturban/test/birmingham_block_4.ply"
TEST_FILES = [
    "data/sensaturban/test/birmingham_block_4.ply",
    "data/sensaturban/test/cambridge_block_20.ply",
]
DARMSTADT_FILE = "data/processed/preprocessed.npy"
MODEL_FILE     = "data/processed/rf_model.joblib"
OUTPUT_DIR     = "data/output"

# Mapping SensatUrban (13 Klassen) -> 4 Hauptklassen
KLASSEN_MAPPING = {
    0:  0,  # ground           -> boden
    1:  1,  # high vegetation  -> vegetation
    2:  2,  # buildings        -> gebaeude
    3:  2,  # walls            -> gebaeude
    4:  3,  # bridge           -> strasse
    5:  3,  # parking          -> strasse
    6:  3,  # rail             -> strasse
    7:  3,  # traffic roads    -> strasse
    8:  0,  # street furniture -> boden
    9:  0,  # cars             -> boden
    10: 3,  # footpath         -> strasse
    11: 3,  # bikes            -> strasse
    12: 0,  # water            -> boden
}

KLASSEN = {0: "boden", 1: "vegetation", 2: "gebaeude", 3: "strasse"}

KLASSEN_FARBEN = {
    0: [0.6, 0.5, 0.3],
    1: [0.2, 0.7, 0.2],
    2: [0.8, 0.8, 0.8],
    3: [0.4, 0.4, 0.4],
}

MAX_PUNKTE_TRAINING = 1_000_000
MAX_PUNKTE_TEST     = 500_000


def lese_ply_mit_labels(filepath, max_punkte=None):
    print(f"  Lese: {filepath}")
    with open(filepath, 'rb') as f:
        properties = []
        num_vertices = 0
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == 'end_header':
                break

        print(f"  Punkte gesamt: {num_vertices:,}")
        print(f"  Properties: {[p[1] for p in properties]}")

        dtype_map = {
            'float32': np.float32, 'float64': np.float64,
            'uint8': np.uint8, 'int32': np.int32, 'uint32': np.uint32,
        }
        dt = np.dtype([(p[1], dtype_map.get(p[0], np.float32)) for p in properties])

        if max_punkte and num_vertices > max_punkte:
            daten = np.frombuffer(f.read(num_vertices * dt.itemsize), dtype=dt)
            idx = np.random.choice(len(daten), max_punkte, replace=False)
            daten = daten[idx]
            print(f"  Subsampled auf: {max_punkte:,} Punkte")
        else:
            daten = np.frombuffer(f.read(), dtype=dt)

    xyz = np.column_stack([daten['x'], daten['y'], daten['z']]).astype(np.float32)
    rgb = np.column_stack([daten['red'], daten['green'], daten['blue']]).astype(np.float32) / 255.0

    labels = None
    if 'class' in [p[1] for p in properties]:
        labels_roh = daten['class'].astype(np.int32)
        labels = np.array([KLASSEN_MAPPING.get(int(l), 0) for l in labels_roh], dtype=np.int32)
        print(f"  Klassen nach Mapping: {np.unique(labels)} ({[KLASSEN[k] for k in np.unique(labels)]})")

    return xyz, rgb, labels


def extrahiere_features(xyz, rgb):
    xyz_norm = xyz - xyz.mean(axis=0)
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    nenner = 2*g + r + b + 1e-8
    vdvi   = (2*g - r - b) / nenner
    features = np.hstack([xyz_norm, rgb, vdvi])
    return features.astype(np.float32)


def berechne_metriken(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)
    iou_werte = {}
    for k in sorted(KLASSEN.keys()):
        if np.sum(y_true == k) == 0:
            continue
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))
        denom = tp + fp + fn
        iou_werte[k] = tp / denom if denom > 0 else 0.0
    miou = np.mean(list(iou_werte.values()))
    return oa, miou, iou_werte


# SCHRITT 1: Training
print("=" * 60)
print("SCHRITT 1: Training (SensatUrban, 4 Klassen)")
print("=" * 60)

np.random.seed(42)
X_liste, y_liste = [], []

for filepath in TRAIN_FILES:
    xyz, rgb, labels = lese_ply_mit_labels(filepath, MAX_PUNKTE_TRAINING)
    if labels is None:
        print(f"  WARNUNG: Keine Labels in {filepath}")
        continue
    X_liste.append(extrahiere_features(xyz, rgb))
    y_liste.append(labels)

X_train = np.vstack(X_liste)
y_train = np.concatenate(y_liste)

print(f"\nTrainingsdaten: {X_train.shape}")
print(f"Klassenverteilung:")
for k, name in KLASSEN.items():
    n = np.sum(y_train == k)
    print(f"  {k} {name:12s}: {n:>8,} ({n/len(y_train)*100:.1f}%)")

print("\nTrainiere Random Forest (100 Baeume, max_depth=20)...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',  # NEU: Klassenimbalance ausgleichen
    verbose=1
)
rf.fit(X_train, y_train)
print(f"Training abgeschlossen in {time.time()-start:.1f}s")

os.makedirs("data/processed", exist_ok=True)
joblib.dump(rf, MODEL_FILE)
print(f"Modell gespeichert: {MODEL_FILE}")

print("\nFeature Importance:")
namen = ["x_norm","y_norm","z_norm","r","g","b","vdvi"]
for name, imp in sorted(zip(namen, rf.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:8s}: {imp:.4f}")

# SCHRITT 2: Evaluation
print("\n" + "=" * 60)
print("SCHRITT 2: Evaluation (SensatUrban Test-Bloecke)")
print("=" * 60)

alle_y_true = []
alle_y_pred = []

for test_file in TEST_FILES:
    print(f"\nTest-Block: {test_file}")
    xyz_t, rgb_t, labels_t = lese_ply_mit_labels(test_file, MAX_PUNKTE_TEST)

    if labels_t is None:
        print("Keine Labels - ueberspringe.")
        continue

    y_pred_t = rf.predict(extrahiere_features(xyz_t, rgb_t))
    oa_t, miou_t, iou_t = berechne_metriken(labels_t, y_pred_t)

    print(f"  OA:   {oa_t*100:.2f}%")
    print(f"  mIoU: {miou_t*100:.2f}%")
    for k, iou in iou_t.items():
        print(f"    {k} {KLASSEN[k]:12s}: {iou*100:.2f}%")

    alle_y_true.append(labels_t)
    alle_y_pred.append(y_pred_t)

# Gesamt-Metriken ueber alle Testbloecke
if alle_y_true:
    y_true_all = np.concatenate(alle_y_true)
    y_pred_all = np.concatenate(alle_y_pred)
    oa_all, miou_all, iou_all = berechne_metriken(y_true_all, y_pred_all)
    print(f"\nGesamt ueber alle Test-Bloecke:")
    print(f"  OA:   {oa_all*100:.2f}%")
    print(f"  mIoU: {miou_all*100:.2f}%")
    for k, iou in iou_all.items():
        print(f"    {k} {KLASSEN[k]:12s}: {iou*100:.2f}%")

np.save("data/processed/test_pred.npy",
        {"labels_pred": y_pred_all, "labels_true": y_true_all},
        allow_pickle=True)
print(f"\nTest-Ergebnis gespeichert: data/processed/test_pred.npy")

# SCHRITT 3: Inferenz Darmstadt
print("\n" + "=" * 60)
print("SCHRITT 3: Inferenz auf Darmstaedter Daten")
print("=" * 60)

darmstadt = np.load(DARMSTADT_FILE, allow_pickle=True).item()
xyz_da  = darmstadt['points'].astype(np.float32)
normals = darmstadt['normals'].astype(np.float32)
rgb_da  = darmstadt.get('colors', np.zeros((len(xyz_da), 3), dtype=np.float32)).astype(np.float32)

print(f"Darmstadt Punkte: {len(xyz_da):,}")
y_da = rf.predict(extrahiere_features(xyz_da, rgb_da))

print(f"Klassenverteilung Darmstadt:")
for k, name in KLASSEN.items():
    n = np.sum(y_da == k)
    print(f"  {k} {name:12s}: {n:>6,} ({n/len(y_da)*100:.1f}%)")

np.save("data/processed/darmstadt_pred.npy",
        {"xyz": xyz_da, "normals": normals, "labels": y_da, "colors": rgb_da},
        allow_pickle=True)
print(f"\nDarmstadt-Segmentierung gespeichert: data/processed/darmstadt_pred.npy")

print("\n" + "=" * 60)
print("FERTIG - Weiter mit 04_reconstruct.py")
print("=" * 60)
