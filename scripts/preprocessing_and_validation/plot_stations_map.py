"""
Script per visualizzare le stazioni di monitoraggio su una mappa geografica.
Crea sia una mappa interattiva HTML che una visualizzazione statica PNG.
Le stazioni sono connesse ai loro vicini più prossimi per formare una rete spaziale completamente connessa.
"""

import json
import folium
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def load_station_data(data_path: str) -> pd.DataFrame:
    """Carica i dati delle stazioni dal file JSON."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_network_edges(df: pd.DataFrame, k_neighbors: int = 5, ensure_connected: bool = True) -> list:
    """
    Calcola gli edges della rete connettendo ogni stazione ai suoi k vicini più prossimi.
    Se ensure_connected=True, aggiunge edges per garantire una rete completamente connessa.
    
    Args:
        df: DataFrame con i dati delle stazioni
        k_neighbors: Numero di vicini più prossimi da connettere
        ensure_connected: Se True, assicura che la rete sia completamente connessa
    
    Returns:
        Lista di tuple (idx_from, idx_to, distance) rappresentanti gli edges
    """
    n = len(df)
    # Estrai coordinate
    coords = df[['Latitude', 'Longitude']].values
    
    # Calcola matrice delle distanze euclidee
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    # Per ogni stazione, trova i k vicini più prossimi
    edges = []
    seen_edges = set()  # Per evitare duplicati (A-B e B-A)
    
    for i in range(n):
        # Ottieni gli indici ordinati per distanza (escludendo sé stessa)
        neighbors_idx = np.argsort(dist_matrix[i])[1:k_neighbors+1]
        
        for j in neighbors_idx:
            # Crea una chiave univoca per l'edge (ordine indipendente)
            edge_key = tuple(sorted([i, j]))
            if edge_key not in seen_edges:
                edges.append((i, j, dist_matrix[i, j]))
                seen_edges.add(edge_key)
    
    # Se richiesto, assicura che la rete sia connessa
    if ensure_connected:
        # Crea una matrice di adiacenza per verificare la connettività
        adj_matrix = np.zeros((n, n))
        for i, j, _ in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        # Trova le componenti connesse
        sparse_adj = csr_matrix(adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)
        
        if n_components > 1:
            print(f"  ⚠ Trovate {n_components} componenti disconnesse, connessione in corso...")
            
            # Per ogni coppia di componenti, trova il collegamento più breve
            components = [np.where(labels == i)[0] for i in range(n_components)]
            
            # Connetti le componenti in modo iterativo
            while n_components > 1:
                # Trova il collegamento più breve tra componenti diverse
                min_dist = float('inf')
                best_edge = None
                
                for comp_idx in range(len(components)):
                    for other_idx in range(comp_idx + 1, len(components)):
                        comp1 = components[comp_idx]
                        comp2 = components[other_idx]
                        
                        # Trova il miglior edge tra queste due componenti
                        for i in comp1:
                            for j in comp2:
                                edge_key = tuple(sorted([i, j]))
                                if edge_key not in seen_edges and dist_matrix[i, j] < min_dist:
                                    min_dist = dist_matrix[i, j]
                                    best_edge = (i, j, dist_matrix[i, j])
                                    best_comp1 = comp_idx
                                    best_comp2 = other_idx
                
                # Aggiungi il miglior edge
                if best_edge:
                    edges.append(best_edge)
                    seen_edges.add(tuple(sorted([best_edge[0], best_edge[1]])))
                    
                    # Unisci le due componenti
                    components[best_comp1] = np.concatenate([components[best_comp1], components[best_comp2]])
                    components.pop(best_comp2)
                    n_components -= 1
                else:
                    break
            
            print(f"  ✓ Rete ora completamente connessa")
    
    return edges


def create_interactive_map(df: pd.DataFrame, edges: list, output_path: str) -> None:
    """
    Crea una mappa interattiva con Folium.
    
    Args:
        df: DataFrame con i dati delle stazioni
        edges: Lista di tuple (idx_from, idx_to, distance) per gli edges
        output_path: Percorso dove salvare la mappa HTML
    """
    # Calcola il centro della mappa (media delle coordinate)
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Crea la mappa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Aggiungi gli edges prima dei nodi (così i nodi appaiono sopra)
    for idx_from, idx_to, distance in edges:
        station_from = df.iloc[idx_from]
        station_to = df.iloc[idx_to]
        
        folium.PolyLine(
            locations=[
                [station_from['Latitude'], station_from['Longitude']],
                [station_to['Latitude'], station_to['Longitude']]
            ],
            color='gray',
            weight=1.5,
            opacity=0.4,
            popup=f"Distanza: {distance:.4f}"
        ).add_to(m)
    
    # Definisci colori per regione
    region_colors = {
        'Trentino (APPA)': 'blue',
        'South Tyrol (Bolzano)': 'red',
        'Lombardia': 'green',
        'Veneto': 'orange',
        'Friuli Venezia Giulia': 'purple'
    }
    
    # Aggiungi marker per ogni stazione
    for idx, row in df.iterrows():
        color = region_colors.get(row['Region'], 'gray')
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            popup=folium.Popup(
                f"<b>{row['Station_ID']}</b><br>"
                f"Regione: {row['Region']}<br>"
                f"Lat: {row['Latitude']:.5f}<br>"
                f"Lon: {row['Longitude']:.5f}",
                max_width=300
            ),
            tooltip=row['Station_ID'],
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Aggiungi legenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin-bottom: 5px;"><b>Regioni</b></p>
    '''
    
    for region, color in region_colors.items():
        if region in df['Region'].values:
            legend_html += f'<p style="margin: 5px;"><span style="color:{color}">●</span> {region}</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Salva la mappa
    m.save(output_path)
    print(f"✓ Mappa interattiva salvata in: {output_path}")


def create_static_map(df: pd.DataFrame, edges: list, output_path: str) -> None:
    """
    Crea una visualizzazione statica con Matplotlib.
    
    Args:
        df: DataFrame con i dati delle stazioni
        edges: Lista di tuple (idx_from, idx_to, distance) per gli edges
        output_path: Percorso dove salvare l'immagine PNG
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Disegna gli edges prima dei nodi
    for idx_from, idx_to, distance in edges:
        station_from = df.iloc[idx_from]
        station_to = df.iloc[idx_to]
        ax.plot(
            [station_from['Longitude'], station_to['Longitude']],
            [station_from['Latitude'], station_to['Latitude']],
            color='gray',
            linewidth=0.8,
            alpha=0.4,
            zorder=1
        )
    
    # Definisci colori per regione
    region_colors = {
        'Trentino (APPA)': '#1f77b4',
        'South Tyrol (Bolzano)': '#ff7f0e',
        'Lombardia': '#2ca02c',
        'Veneto': '#d62728',
        'Friuli Venezia Giulia': '#9467bd'
    }
    
    # Plotta le stazioni per regione (sopra gli edges)
    regions = df['Region'].unique()
    for region in regions:
        region_df = df[df['Region'] == region]
        color = region_colors.get(region, 'gray')
        ax.scatter(
            region_df['Longitude'],
            region_df['Latitude'],
            c=color,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            label=region,
            zorder=2
        )
    
    # Aggiungi etichette per alcune stazioni (per non sovraccaricare)
    for idx, row in df.iterrows():
        if idx % 3 == 0:  # Mostra solo 1 su 3 per leggibilità
            ax.annotate(
                row['Station_ID'],
                xy=(row['Longitude'], row['Latitude']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                alpha=0.7
            )
    
    # Configurazione assi e griglia
    ax.set_xlabel('Longitudine', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitudine', fontsize=12, fontweight='bold')
    ax.set_title('Rete Spaziale delle Stazioni di Monitoraggio (Completamente Connessa)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Aggiungi informazioni statistiche
    stats_text = (
        f'Totale stazioni: {len(df)}\n'
        f'Totale connessioni: {len(edges)}\n'
        f'Latitudine: [{df["Latitude"].min():.3f}, {df["Latitude"].max():.3f}]\n'
        f'Longitudine: [{df["Longitude"].min():.3f}, {df["Longitude"].max():.3f}]'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Mappa statica salvata in: {output_path}")


def print_station_summary(df: pd.DataFrame) -> None:
    """Stampa un riepilogo delle stazioni."""
    print("\n" + "="*60)
    print("RIEPILOGO STAZIONI DI MONITORAGGIO")
    print("="*60)
    print(f"\nTotale stazioni: {len(df)}")
    print("\nStazioni per regione:")
    region_counts = df['Region'].value_counts()
    for region, count in region_counts.items():
        print(f"  • {region}: {count} stazioni")
    
    print(f"\nArea coperta:")
    print(f"  Latitudine: {df['Latitude'].min():.5f} → {df['Latitude'].max():.5f}")
    print(f"  Longitudine: {df['Longitude'].min():.5f} → {df['Longitude'].max():.5f}")
    print("="*60 + "\n")


def main():
    """Funzione principale."""
    # Definisci i percorsi
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "data_stations_metadata.json"
    output_dir = base_path / "assets" / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carica i dati
    print("Caricamento dati delle stazioni...")
    df = load_station_data(data_path)
    
    # Stampa riepilogo
    print_station_summary(df)
    
    # Calcola la rete di connessioni
    print("Calcolo rete di connessioni...")
    k_neighbors = 5  # Numero di vicini più prossimi da connettere
    edges = compute_network_edges(df, k_neighbors, ensure_connected=True)
    print(f"✓ Create {len(edges)} connessioni totali")
    
    # Crea le mappe
    print("\nCreazione mappe con rete di connessioni...")
    
    # Mappa interattiva
    interactive_map_path = output_dir / "stations_interactive_map.html"
    create_interactive_map(df, edges, str(interactive_map_path))
    
    # Mappa statica
    static_map_path = output_dir / "stations_static_map.png"
    create_static_map(df, edges, str(static_map_path))
    
    print(f"\n✓ Processo completato con successo!")
    print(f"  → Apri {interactive_map_path} nel browser per la mappa interattiva")
    print(f"  → Visualizza {static_map_path} per la mappa statica\n")


if __name__ == "__main__":
    main()
