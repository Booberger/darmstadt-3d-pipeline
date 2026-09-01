import laspy
import numpy as np

with laspy.open("data/processed/ausschnitt.laz") as f:
    # Nur ersten Chunk lesen
    chunk = next(f.chunk_iterator(1_000_000))
    
    print("Eindeutige Classifications:", np.unique(chunk.classification))
    print("Anzahl unique point_source_ids:", len(np.unique(chunk.point_source_id)))
    print("Unique point_source_ids:", np.unique(chunk.point_source_id))
    print("\nreturn_number Verteilung:")
    unique, counts = np.unique(chunk.return_number, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Return {u}: {c:,} Punkte ({c/len(chunk.x)*100:.1f}%)")