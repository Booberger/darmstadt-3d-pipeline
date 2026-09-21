"""
04_reconstruct.py
-----------------
Schritt 4 der Pipeline: Mesh-Rekonstruktion aus segmentierter Punktwolke.

Eingabe:  data/processed/darmstadt_pred.npy  (Segmentierungsergebnis)
          data/processed/preprocessed.npy    (RGB-Farben der Originalpunktwolke)
Ausgabe:  data/output/*_LOD0/LOD1/LOD2.ply      (Klassenfarben, fuer MeshLab)
          data/output/*_LOD0/LOD1/LOD2.glb      (Klassenfarben, fuer Game Engine)
          data/output/*_LOD0/LOD1/LOD2_rgb.glb  (echte RGB, fuer Game Engine)

Verarbeitungsschritte:
  1. Segmentierungsergebnis und RGB-Farben laden
  2. Mesh-Rekonstruktion pro Klasse:
     a. DBSCAN-Artefaktfilterung (entfernt isolierte Fehlklassifikationen)
     b. Screened Poisson Surface Reconstruction
     c. Mesh-Bereinigung (degenerierte Dreiecke, nicht-manifolde Kanten)
  3. 3-stufige LOD-Generierung via Quadric Edge Collapse Decimation
     LOD0: 100% (Original) | LOD1: 10% | LOD2: 1%
  4. glTF-Export mit echten RGB-Farben (KD-Tree Farbzuweisung)

Hinweis zur LOD-Konvention:
  Es wird die Rendering-Konvention verwendet (LOD0 = hoechste Detailstufe),
  nicht die CityGML-Konvention (LOD0 = niedrigste Detailstufe).
"""

import open3d as o3d
import numpy as np
import os

# ── Konfiguration ─────────────────────────────────────────────────────────────
INPUT_PRED    = "data/processed/darmstadt_pred.npy"  # Segmentierungsergebnis
INPUT_PREPROC = "data/processed/preprocessed.npy"   # Originalpunktwolke (RGB)
OUTPUT_DIR    = "data/output"

POISSON_DEPTH      = 9     # Octree-Tiefe fuer Poisson Reconstruction (hoeher = mehr Detail)
LOD1_REDUCTION     = 0.10  # LOD1: 10% der Dreiecke von LOD0
LOD2_REDUCTION     = 0.01  # LOD2:  1% der Dreiecke von LOD0
DICHTE_PERCENTIL   = 10    # Unterste X% der Poisson-Dichte werden als Artefakte entfernt
DBSCAN_EPS         = 2.0   # Maximaler Nachbarschaftsradius fuer DBSCAN in Metern
DBSCAN_MIN_POINTS  = 10    # Mindestpunkte pro DBSCAN-Cluster
MIN_CLUSTER_PUNKTE = 50    # Cluster mit weniger Punkten gelten als Artefakte

KLASSEN = {
    0: "boden",
    1: "vegetation",
    2: "gebaeude",
    3: "strasse"
}

# Einheitliche Klassenfarben fuer Visualisierung (RGB 0-1)
KLASSEN_FARBEN = {
    0: [0.6, 0.5, 0.3],  # Braun  - Boden
    1: [0.2, 0.7, 0.2],  # Gruen  - Vegetation
    2: [0.8, 0.8, 0.8],  # Grau   - Gebaeude
    3: [0.3, 0.3, 0.3],  # Dunkel - Strasse
}

# Hilfsfunktionen 

def filtere_artefakte(pcd):
    """
    Entfernt kleine isolierte Punktgruppen via DBSCAN-Clustering.
    Cluster mit weniger als MIN_CLUSTER_PUNKTE Punkten gelten als
    Segmentierungsartefakte und werden entfernt.
    Echte urbane Objekte (Gebäude, Strassensegmente) haben typischerweise
    deutlich mehr als 50 zusammenhaengende Punkte.
    Gibt die bereinigte Punktwolke zurueck.
    """
    if len(pcd.points) < MIN_CLUSTER_PUNKTE:
        return pcd

    # DBSCAN: Punkte innerhalb DBSCAN_EPS Meter gehörn zum selben Cluster
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=False
    ))

    if len(labels) == 0 or labels.max() < 0:
        return pcd  # Kein Cluster gefunden - unverändert zurückgeben

    # Nur große cluster behalten. klein = artefakt
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    grosse_cluster = unique[counts >= MIN_CLUSTER_PUNKTE]

    if len(grosse_cluster) == 0:
        return pcd

    maske = np.isin(labels, grosse_cluster)
    pcd_f = pcd.select_by_index(np.where(maske)[0].tolist())

    entfernt = len(pcd.points) - len(pcd_f.points)
    print(f"    Artefakte entfernt: {entfernt:,} ({entfernt/len(pcd.points)*100:.1f}%)")
    return pcd_f


def rekonstruiere_klasse(xyz_all, labels_all, normals_all, colors_all, klasse_id, name):
    """
    Rekonstruiert ein Mesh für eine semantische Klasse,
    Gibt (mesh, pcd_rgb) zurück oder (None, None) bei zu wenigen Punkten.

    Ablauf:
      1. Punkte der Klasse extrahieren
      2. Normalen setzen oder neu schaetzen
      3. DBSCAN-Artefaktfilterung
      4. Screened Poisson Surface Reconstruction
      5. Randatefakte via Dichtefilter entfernen
      6. Mesh-Bereinigung
    """
    maske    = labels_all == klasse_id
    n_punkte = int(np.sum(maske))

    if n_punkte < 100:
        print(f"  {name}: Zu wenige Punkte ({n_punkte}) - ueberspringe.")
        return None, None

    print(f"\n  {name}: {n_punkte:,} Punkte")

    # Punktwolke für Mesh-Rekonstruktion ((mit Normalen)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_all[maske])

    if normals_all is not None:
        # Vorberechnete Normalen aus Vorverarbeitung verwenden
        pcd.normals = o3d.utility.Vector3dVector(normals_all[maske])
    else:
        # Normalen neu schätzen falls nicht vorhanden
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)

    # Separate Punktwolke mit RGB-Farben für spätere Farbzuweisung auf Mesh
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(xyz_all[maske])
    if colors_all is not None:
        pcd_rgb.colors  = o3d.utility.Vector3dVector(colors_all[maske])
        pcd_rgb.normals = pcd.normals

    # Artefaktfilterung via DBSCAN
    pcd = filtere_artefakte(pcd)
    if len(pcd.points) < 100:
        return None, None

    # Screened Poisson Surface Reconstruction
    # Erzeugt ein wasserdichtes, geschlossenes Mesh aus der Punktwolke
    print(f"    Poisson Reconstruction (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )

    # Randatefakte entfernen: Vertices im untersten Dichte-Perzentil
    # haben wenig Eingabepunkte in ihrer Nahe und sind meist Artefakte
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.percentile(densities, DICHTE_PERCENTIL))

    # Mesh-Bereinigung fuer robuste Weiterverarbeitung
    mesh.remove_degenerate_triangles()   # Dreiecke mit Fläche = 0
    mesh.remove_duplicated_triangles()   # Mehrfach vorhandene Dreiecke
    mesh.remove_duplicated_vertices()    # Mehrfach vorhandene Vertices
    mesh.remove_non_manifold_edges()     # Kanten die mehr als 2 Dreiecken gehören
    mesh.orient_triangles()              # Konsistente Normalenrichtung
    mesh.compute_vertex_normals()        # Vertex-Normalen für Smooth Shading

    print(f"    Dreiecke: {len(mesh.triangles):,}")
    print(f"    Vertices: {len(mesh.vertices):,}")

    return mesh, pcd_rgb


def erzeuge_lod(mesh, farbe, n, reduktion, suffix, output_dir, name):
    """
    Erzeugt eine LOD-Stufe via Quadric Edge Collapse Decimation.
    Reduziert die Dreiecksanzahl auf n * reduktion durch iteratives
    Kollabieren der Kanten mit den geringsten geometrischen Fehlerkosten.
    Speichert als PLY (Klassenfarben) und GLB (für Game Engine).
    Gibt (mesh_lod, dreiecksanzahl, dateigroesse_mb) zurück.
    """
    ziel     = max(10, int(n * reduktion))
    mesh_lod = mesh.simplify_quadric_decimation(ziel)
    mesh_lod.paint_uniform_color(farbe)

    ply_path = f"{output_dir}/{name}_{suffix}.ply"
    glb_path = f"{output_dir}/{name}_{suffix}.glb"
    o3d.io.write_triangle_mesh(ply_path, mesh_lod)
    o3d.io.write_triangle_mesh(glb_path, mesh_lod)

    return mesh_lod, len(mesh_lod.triangles), os.path.getsize(ply_path)/1e6

# Schritt 1: Daten laden
# Segmentierungsergebnis (XYZ, Labels, Normalen) aus Schritt 3 laden.
# RGB-Farben werden separat aus der Vorverarbeitung geladen da sie nicht
# in darmstadt_pred.npy gespeichert sind.
# Normalen wreden auf korrekte Orientierung geprüft: Nach ALS-Aufnahme
# von oben sollten Normalen eine positive Z-Komponente haben.
print("SCHRITT 1: Segmentierungsergebnis laden")
print("-" * 60)

daten   = np.load(INPUT_PRED,    allow_pickle=True).item()
preproc = np.load(INPUT_PREPROC, allow_pickle=True).item()

xyz     = daten['xyz']    if 'xyz'    in daten else daten['points']
labels  = daten['labels']
normals = daten.get('normals', None)
colors  = preproc.get('colors', None)

# Normalen-Korrektur: Falls Normalen nach unten zeigen (negative Z-Komponente) werden sie umgekehrt
if normals is not None and np.mean(normals[:, 2]) < 0:
    normals = -normals
    print("Normalen umgekehrt (zeigten nach unten)")

print(f"Punkte: {len(xyz):,}")
print(f"RGB verfügbar: {colors is not None}")
print(f"Klassenverteilung:")
for k, name in KLASSEN.items():
    n = int(np.sum(labels == k))
    print(f"  {k} {name:12s}: {n:>6,} ({n/len(labels)*100:.1f}%)")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Schritt 2: Mesh-Rekonstruktion 
# Für jede der 4 semantischen Klassen wird separat ein Mesh rekonstruiert.
# Dadrurch können die Klassen in der Game Engine unabhängig voneinander
# geladen, eingefaerbt und manipuliert werden.
print("\nSCHRITT 2: Mesh-Rekonstruktion mit Artefaktfilterung")
print("-" * 60)

meshes = {}
for klasse_id, name in KLASSEN.items():
    mesh, pcd_rgb = rekonstruiere_klasse(
        xyz, labels, normals, colors, klasse_id, name
    )
    if mesh is not None:
        meshes[name] = (klasse_id, mesh, pcd_rgb)

# Schritt 3: LOD-Generierung 
# Drei LOD-Stufen werden erzeugt (Rendering-Konvention: LOD0 = höcxchste Detailstufe):
#   LOD0: Vollstandiges Poisson-Mesh
#   LOD1: 10% der Dreiecke       
#   LOD2:  1% der Dreiecke         
# Alle Stufen werden als PLY (Klassenfarben, MeshLab) und GLB (Game Engine) gespeichert.
print("\nSCHRITT 3: 3-stufige LOD-Generierung")
print("-" * 60)
print(f"{'Klasse':12s} | {'LOD0':>10s} | {'LOD1 (10%)':>12s} | {'LOD2 (1%)':>11s}")
print("-" * 55)

for name, (klasse_id, mesh, _) in meshes.items():
    farbe = KLASSEN_FARBEN[klasse_id]
    n     = len(mesh.triangles)

    # LOD0 — Vollständiges Mesh mit Klassenfarbe
    mesh_lod0 = o3d.geometry.TriangleMesh(mesh)
    mesh_lod0.paint_uniform_color(farbe)
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0.ply", mesh_lod0)
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0.glb", mesh_lod0)
    g0 = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0.ply") / 1e6

    # LOD1 und LOD2 via Quadric Decimation
    _, n1, g1 = erzeuge_lod(mesh, farbe, n, LOD1_REDUCTION, "LOD1", OUTPUT_DIR, name)
    _, n2, g2 = erzeuge_lod(mesh, farbe, n, LOD2_REDUCTION, "LOD2", OUTPUT_DIR, name)

    print(f"{name:12s} | {n:>7,} ({g0:.1f}MB) | {n1:>7,} ({g1:.1f}MB) | {n2:>6,} ({g2:.1f}MB)")

# Schritt 4: glTF-Export mit echten RGB-Farben 
# Zusätzlich zu den Klassenfarben werden Versionen mit echten RGB-Farben erzeugt
# Da das Mesh nach Poisson andere Vertices hat als die Eingabepunktwolke,
# wird die Farbe via KD-Tree übertragen: Für jeden Mesh-Vertex wird der
# nächste Punkt in der RGB-Punktwolke gesucht und dessen Farbe übernommen
# Diese Versionen eignen sich für realistische Visualisierungen in Game Engines
print("\nSCHRITT 4: glTF Export (Echte RGB-Farben)")
print("-" * 60)

if colors is not None:
    for name, (klasse_id, mesh, pcd_rgb) in meshes.items():
        if pcd_rgb is None or not pcd_rgb.has_colors():
            continue

        # KD-Tree auf RGB-Punktwolke aufbauen für schnelle Nachbarschaftssuche
        mesh_rgb  = o3d.geometry.TriangleMesh(mesh)
        vertices  = np.asarray(mesh_rgb.vertices)
        pcd_tree  = o3d.geometry.KDTreeFlann(pcd_rgb)
        vertex_colors = np.zeros((len(vertices), 3))

        # Für jeden Mesh-Vertex:n ächsten Punkt in RGB-Punktwolke suchen
        for i, v in enumerate(vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            vertex_colors[i] = np.asarray(pcd_rgb.colors)[idx[0]]

        mesh_rgb.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # Alle drei LOD-Stufen mit echten RGB-Farben exportieren
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0_rgb.glb", mesh_rgb)

        mesh_lod1_rgb = mesh_rgb.simplify_quadric_decimation(
            max(10, int(len(mesh_rgb.triangles) * LOD1_REDUCTION))
        )
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD1_rgb.glb", mesh_lod1_rgb)

        mesh_lod2_rgb = mesh_rgb.simplify_quadric_decimation(
            max(10, int(len(mesh_rgb.triangles) * LOD2_REDUCTION))
        )
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD2_rgb.glb", mesh_lod2_rgb)

        g = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0_rgb.glb") / 1e6
        print(f"{name:12s}: LOD0/LOD1/LOD2 RGB exportiert ({g:.1f}MB)")
else:
    print("Keine RGB-Daten verfuegbar.")

# Zusammenfassung
print("\nZUSAMMENFASSUNG")
print("-" * 60)
print("Ausgabedateien pro Klasse:")
print("  *_LOD0/LOD1/LOD2.ply     - Klassenfarben (MeshLab)")
print("  *_LOD0/LOD1/LOD2.glb     - Klassenfarben (Game Engine)")
print("  *_LOD0/LOD1/LOD2_rgb.glb - Echte RGB-Farben (Game Engine)")

print("\nFERTIG - Output in data/output/")
print("-" * 60)