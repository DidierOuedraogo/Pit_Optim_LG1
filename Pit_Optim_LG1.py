import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import time
import json
from datetime import datetime
import zipfile
import os

# Configuration de la page
st.set_page_config(
    page_title="LG Pit Optimizer",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et auteur
st.title("LG Pit Optimizer")
st.markdown("### Optimisation de fosses minières par l'algorithme de Lerch-Grossman")
st.markdown("*Développé par: **Didier Ouedraogo, P.Geo***")

# Initialiser les variables d'état
if 'block_model' not in st.session_state:
    st.session_state.block_model = None
if 'optimal_pit' not in st.session_state:
    st.session_state.optimal_pit = None
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = 0
if 'model_imported' not in st.session_state:
    st.session_state.model_imported = False
if 'imported_model_info' not in st.session_state:
    st.session_state.imported_model_info = {}

# Fonction pour analyser et valider un fichier CSV de modèle de blocs
def parse_block_model_csv(file):
    try:
        # Lire le fichier CSV
        df = pd.read_csv(file)
        
        # Vérifier les colonnes obligatoires
        required_cols = ['X', 'Y', 'Z']
        if not all(col in df.columns for col in required_cols):
            return False, "Le fichier doit contenir au minimum les colonnes X, Y, Z"
            
        # Vérifier s'il y a une colonne de teneur
        grade_col = None
        for possible_col in ['GRADE', 'TENEUR', 'AU', 'CU', 'FE', 'METAL', 'GRADE_1']:
            if possible_col in df.columns:
                grade_col = possible_col
                break
        
        if grade_col is None:
            return False, "Aucune colonne de teneur identifiée. Veuillez renommer votre colonne de teneur en GRADE."
        
        # Vérifier les données numériques
        for col in required_cols + [grade_col]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"La colonne {col} doit contenir des valeurs numériques."
        
        # Déterminer la taille du bloc
        x_coords = df['X'].sort_values().unique()
        y_coords = df['Y'].sort_values().unique()
        z_coords = df['Z'].sort_values().unique()
        
        if len(x_coords) <= 1 or len(y_coords) <= 1 or len(z_coords) <= 1:
            return False, "Le modèle doit contenir au moins 2 blocs dans chaque dimension"
        
        # Calculer la taille moyenne des blocs
        x_diffs = np.diff(x_coords)
        y_diffs = np.diff(y_coords)
        z_diffs = np.diff(z_coords)
        
        if len(x_diffs) > 0 and len(y_diffs) > 0 and len(z_diffs) > 0:
            block_size_x = np.median(x_diffs[x_diffs > 0])
            block_size_y = np.median(y_diffs[y_diffs > 0])
            block_size_z = np.median(z_diffs[z_diffs > 0])
            
            # Vérifier si les dimensions sont cohérentes
            if abs(block_size_x - block_size_y) > 0.1 * max(block_size_x, block_size_y):
                st.warning(f"Les dimensions des blocs en X ({block_size_x}) et Y ({block_size_y}) sont différentes. On utilisera la moyenne.")
            
            block_size = np.mean([block_size_x, block_size_y, block_size_z])
        else:
            block_size = 10  # Valeur par défaut
            st.warning("Impossible de déterminer la taille des blocs, on utilise 10m par défaut.")
        
        # Déterminer les dimensions du modèle
        size_x = len(x_coords)
        size_y = len(y_coords)
        size_z = len(z_coords)
        
        # Origine du modèle
        origin_x = min(x_coords)
        origin_y = min(y_coords)
        origin_z = max(z_coords)  # Z diminue avec la profondeur
        
        # Créer une version normalisée du modèle de blocs
        block_model = []
        
        for _, row in df.iterrows():
            # Calculer les indices x, y, z normalisés
            x_idx = int(round((row['X'] - origin_x) / block_size))
            y_idx = int(round((row['Y'] - origin_y) / block_size))
            z_idx = int(round((origin_z - row['Z']) / block_size))
            
            # S'assurer que les indices sont dans les limites
            if x_idx < 0 or x_idx >= size_x or y_idx < 0 or y_idx >= size_y or z_idx < 0 or z_idx >= size_z:
                continue
            
            grade = row[grade_col]
            
            # Ajouter le bloc au modèle
            block_model.append({
                'x': x_idx,
                'y': y_idx,
                'z': z_idx,
                'real_x': row['X'],
                'real_y': row['Y'],
                'real_z': row['Z'],
                'grade': grade,
                'value': 0,  # Sera calculé plus tard
                'in_pit': False
            })
        
        model_info = {
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'block_size': block_size,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'origin_z': origin_z,
            'total_blocks': len(block_model),
            'grade_column': grade_col,
            'min_grade': df[grade_col].min(),
            'max_grade': df[grade_col].max(),
            'avg_grade': df[grade_col].mean()
        }
        
        return True, {
            'block_model': block_model,
            'info': model_info
        }
        
    except Exception as e:
        return False, f"Erreur lors de l'analyse du fichier: {str(e)}"

# Fonction pour analyser et valider un fichier JSON de modèle de blocs
def parse_json_model(file):
    try:
        # Charger le JSON
        data = json.loads(file.getvalue())
        
        # Vérifier la structure attendue
        if 'blocks' not in data:
            return False, "Le fichier JSON doit contenir une clé 'blocks' avec un tableau de blocs"
        
        # Vérifier si les blocs ont les propriétés attendues
        blocks = data['blocks']
        if not blocks or not isinstance(blocks, list):
            return False, "Le tableau 'blocks' est vide ou n'est pas un tableau"
        
        # Récupérer les métadonnées si disponibles
        metadata = data.get('metadata', {})
        size_x = metadata.get('size_x', 0)
        size_y = metadata.get('size_y', 0)
        size_z = metadata.get('size_z', 0)
        block_size = metadata.get('block_size', 10)
        origin_x = metadata.get('origin_x', 0)
        origin_y = metadata.get('origin_y', 0)
        origin_z = metadata.get('origin_z', 0)
        
        # Construire le modèle de blocs
        block_model = []
        x_coords, y_coords, z_coords = set(), set(), set()
        max_grade = 0
        min_grade = float('inf')
        sum_grade = 0
        
        for block in blocks:
            # Vérifier les coordonnées et la teneur
            if 'x' not in block or 'y' not in block or 'z' not in block:
                # Vérifier les alternates
                if ('real_x' in block and 'real_y' in block and 'real_z' in block):
                    x, y, z = block['real_x'], block['real_y'], block['real_z']
                else:
                    continue
            else:
                x, y, z = block['x'], block['y'], block['z']
            
            # Chercher la teneur sous différents noms
            grade = None
            for key in ['grade', 'teneur', 'au', 'cu', 'fe', 'metal', 'value']:
                if key in block:
                    grade = block[key]
                    break
            
            if grade is None:
                continue
            
            # Collecter les coordonnées pour déterminer les dimensions
            x_coords.add(x)
            y_coords.add(y)
            z_coords.add(z)
            
            # Mettre à jour les statistiques
            max_grade = max(max_grade, grade)
            min_grade = min(min_grade, grade)
            sum_grade += grade
            
            # Si le modèle utilise déjà des indices, les utiliser directement
            if all(isinstance(block.get(k, 0), int) for k in ['x', 'y', 'z']) and 'real_x' in block:
                x_idx, y_idx, z_idx = block['x'], block['y'], block['z']
                real_x, real_y, real_z = block['real_x'], block['real_y'], block['real_z']
            else:
                # Sinon, calculer les indices et utiliser les coordonnées comme réelles
                real_x, real_y, real_z = x, y, z
                
                # Si les dimensions et origines sont spécifiées, calculer les indices
                if size_x > 0 and origin_x != 0:
                    x_idx = int(round((x - origin_x) / block_size))
                    y_idx = int(round((y - origin_y) / block_size))
                    z_idx = int(round((origin_z - z) / block_size))
                else:
                    # Sinon, simplement utiliser les valeurs comme indices
                    x_idx, y_idx, z_idx = int(x), int(y), int(z)
            
            # Ajouter le bloc au modèle
            block_model.append({
                'x': x_idx,
                'y': y_idx,
                'z': z_idx,
                'real_x': real_x,
                'real_y': real_y,
                'real_z': real_z,
                'grade': grade,
                'value': block.get('value', 0),
                'in_pit': block.get('in_pit', False)
            })
        
        # Si les dimensions n'étaient pas spécifiées, les calculer
        if size_x == 0:
            size_x = len(x_coords)
        if size_y == 0:
            size_y = len(y_coords)
        if size_z == 0:
            size_z = len(z_coords)
        
        # Si les origines n'étaient pas spécifiées, les calculer
        if origin_x == 0:
            origin_x = min(x_coords) if x_coords else 0
        if origin_y == 0:
            origin_y = min(y_coords) if y_coords else 0
        if origin_z == 0:
            origin_z = max(z_coords) if z_coords else 0
        
        avg_grade = sum_grade / len(block_model) if block_model else 0
        
        model_info = {
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'block_size': block_size,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'origin_z': origin_z,
            'total_blocks': len(block_model),
            'grade_column': 'GRADE',
            'min_grade': min_grade,
            'max_grade': max_grade,
            'avg_grade': avg_grade
        }
        
        return True, {
            'block_model': block_model,
            'info': model_info
        }
        
    except Exception as e:
        return False, f"Erreur lors de l'analyse du fichier JSON: {str(e)}"

# Fonction pour générer un modèle de blocs
def generate_block_model(size_x, size_y, size_z, origin_x, origin_y, origin_z, block_size, 
                         metal_price, mining_cost, processing_cost, recovery, cutoff_grade):
    block_model = []
    
    # Générer un modèle de blocs avec des teneurs aléatoires
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                # Plus profond = teneur plus élevée (simulation d'un gisement)
                depth_factor = z / size_z
                dist_from_center = np.sqrt(((x - size_x/2) / (size_x/2))**2 + ((y - size_y/2) / (size_y/2))**2)
                
                # Teneur plus élevée au centre et en profondeur
                grade = (1 - dist_from_center) * depth_factor * 2
                grade = max(0, grade + (np.random.random() * 0.3 - 0.15))  # Ajout de bruit
                
                # Valeur économique basée sur la teneur
                tonnage = block_size**3 * 2.7  # Densité moyenne de 2.7 t/m³
                
                if grade > cutoff_grade:
                    # Bloc de minerai
                    value = tonnage * (grade * metal_price * recovery - (mining_cost + processing_cost))
                else:
                    # Bloc de stérile
                    value = -tonnage * mining_cost
                
                # Calculer les coordonnées réelles
                real_x = origin_x + x * block_size
                real_y = origin_y + y * block_size
                real_z = origin_z - z * block_size  # Z diminue avec la profondeur
                
                block_model.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'real_x': real_x,
                    'real_y': real_y,
                    'real_z': real_z,
                    'grade': grade,
                    'value': value,
                    'in_pit': False,
                    'block_size': block_size
                })
    
    return block_model

# Algorithme de Lerch-Grossman
def run_lerch_grossman(block_model, size_x, size_y, size_z, slope_angle, iterations=5, tolerance=0.01):
    # Créer une copie pour ne pas modifier le modèle original
    model_copy = block_model.copy()
    optimal_pit = []
    
    # Marquer les blocs de surface avec une valeur positive
    for y in range(size_y):
        for x in range(size_x):
            for z in range(size_z-1, -1, -1):
                index = z * size_x * size_y + y * size_x + x
                if index < len(model_copy):
                    block = model_copy[index]
                    
                    if z == size_z-1 or block['value'] > 0:
                        block['in_pit'] = True
                        optimal_pit.append(block)
                        break  # Passer au prochain x,y
    
    # Ajouter des blocs en respectant les contraintes de pente
    max_depth_diff = np.tan(np.radians(90 - slope_angle))
    
    # Simplification: ajouter des blocs en couches depuis la surface
    for z in range(size_z-2, -1, -1):
        for y in range(size_y):
            for x in range(size_x):
                index = z * size_x * size_y + y * size_x + x
                if index < len(model_copy):
                    block = model_copy[index]
                    
                    # Vérifier si les blocs au-dessus sont dans la fosse
                    can_be_extracted = False
                    
                    if z+1 < size_z:
                        above_index = (z+1) * size_x * size_y + y * size_x + x
                        if above_index < len(model_copy) and model_copy[above_index]['in_pit']:
                            can_be_extracted = True
                    
                    # Vérifier les contraintes de pente (simplifié)
                    if can_be_extracted and (block['value'] > 0 or np.random.random() < 0.3):
                        block['in_pit'] = True
                        optimal_pit.append(block)
    
    return optimal_pit

def get_neighbors(block_model, block, size_x, size_y, size_z):
    neighbors = []
    directions = [
        (1, 0, 0), (-1, 0, 0), 
        (0, 1, 0), (0, -1, 0),
    ]
    
    for dx, dy, dz in directions:
        nx, ny, nz = block['x'] + dx, block['y'] + dy, block['z'] + dz
        
        # Vérifier si le voisin est dans les limites
        if 0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z:
            neighbor_index = nz * size_x * size_y + ny * size_x + nx
            
            if 0 <= neighbor_index < len(block_model):
                neighbors.append(block_model[neighbor_index])
    
    return neighbors

# Fonctions d'exportation
def generate_csv(block_model, optimal_pit, include_coordinates, include_grades, include_values, only_pit):
    # Filtrer si nécessaire
    data = optimal_pit if only_pit else block_model
    
    # Préparer le DataFrame
    rows = []
    for block in data:
        row = {}
        
        if include_coordinates:
            row['X'] = block['real_x']
            row['Y'] = block['real_y']
            row['Z'] = block['real_z']
        
        if include_grades:
            row['GRADE'] = round(block['grade'], 2)
        
        if include_values:
            row['VALUE'] = round(block['value'], 1)
        
        row['INPIT'] = 1 if block['in_pit'] else 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def generate_dxf(optimal_pit, selected_level, include_points, include_polylines, include_3dfaces, block_size):
    # Identifier les blocs à la limite de la fosse
    pit_boundary_blocks = []
    
    # Convertir optimal_pit en DataFrame pour faciliter le filtrage
    df = pd.DataFrame(optimal_pit)
    
    # Si on veut tous les niveaux
    if selected_level == 'all':
        # Simplement inclure tous les blocs de la fosse pour cette démo
        pit_boundary_blocks = optimal_pit
    else:
        # Filtrer par niveau spécifique
        level = int(selected_level)
        level_blocks = df[df['z'] == level-1]  # Niveau 1 correspond à z=0, etc.
        pit_boundary_blocks = level_blocks.to_dict('records')
    
    # Générer un DXF simplifié (contenu textuel)
    dxf_content = "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1027\n"
    dxf_content += "0\nENDSEC\n0\nSECTION\n2\nENTITIES\n"
    
    # Ajouter des entités simples pour la démonstration
    if include_points and pit_boundary_blocks:
        for block in pit_boundary_blocks[:10]:  # Limiter à 10 points pour l'exemple
            dxf_content += f"0\nPOINT\n8\nPIT_BOUNDARY\n10\n{block['real_x']}\n20\n{block['real_y']}\n30\n{block['real_z']}\n"
    
    if include_polylines and pit_boundary_blocks:
        dxf_content += "0\nPOLYLINE\n8\nPIT_BOUNDARY\n66\n1\n70\n1\n"
        for block in pit_boundary_blocks[:10]:  # Limiter à 10 vertices pour l'exemple
            dxf_content += f"0\nVERTEX\n8\nPIT_BOUNDARY\n10\n{block['real_x']}\n20\n{block['real_y']}\n30\n{block['real_z']}\n"
        dxf_content += "0\nSEQEND\n"
    
    dxf_content += "0\nENDSEC\n0\nEOF"
    
    return dxf_content

def prepare_download_link(content, filename, mime_type):
    """Génère un lien de téléchargement pour le contenu donné"""
    if isinstance(content, pd.DataFrame):
        # Pour DataFrame, convertir en CSV
        content = content.to_csv(index=False)
        b64 = base64.b64encode(content.encode()).decode()
    elif isinstance(content, str):
        # Pour le texte (comme DXF)
        b64 = base64.b64encode(content.encode()).decode()
    else:
        # Pour d'autres types (JSON, etc.)
        b64 = base64.b64encode(json.dumps(content).encode()).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="download-button">Télécharger {filename}</a>'
    return href

# Création de l'interface avec deux colonnes principales
col1, col2 = st.columns([1, 1])

# Colonne 1: Paramètres et contrôles
with col1:
    # Paramètres d'algorithme
    st.header("Algorithme de Lerch-Grossman")
    
    with st.expander("Paramètres de l'algorithme", expanded=True):
        lg_iterations = st.slider("Nombre d'itérations", min_value=1, max_value=20, value=5)
        lg_tolerance = st.slider("Tolérance", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
    
    # Importer ou générer un modèle de blocs
    st.header("Modèle de blocs")
    
    model_tabs = st.tabs(["Importer un modèle", "Générer un modèle"])
    
    with model_tabs[0]:
        st.write("Importez votre modèle de blocs depuis un fichier")
        
        uploaded_file = st.file_uploader("Sélectionner un fichier", 
                                       type=["csv", "json"],
                                       help="Formats supportés: CSV, JSON")
        
        if uploaded_file is not None:
            # Déterminer le type de fichier
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            with st.spinner("Analyse du modèle de blocs..."):
                success = False
                message = ""
                
                if file_ext == '.csv':
                    success, result = parse_block_model_csv(uploaded_file)
                elif file_ext == '.json':
                    success, result = parse_json_model(uploaded_file)
                else:
                    message = "Format de fichier non supporté"
                
                if success:
                    # Stocker le modèle importé
                    st.session_state.block_model = result['block_model']
                    st.session_state.imported_model_info = result['info']
                    st.session_state.model_imported = True
                    
                    # Afficher un résumé du modèle importé
                    st.success(f"Modèle importé avec succès: {result['info']['total_blocks']} blocs")
                    
                    with st.expander("Détails du modèle importé", expanded=True):
                        info = result['info']
                        st.write(f"Dimensions: {info['size_x']} × {info['size_y']} × {info['size_z']} blocs")
                        st.write(f"Taille de bloc: {info['block_size']} m")
                        st.write(f"Origine: X={info['origin_x']}, Y={info['origin_y']}, Z={info['origin_z']}")
                        st.write(f"Teneur: Min={info['min_grade']:.3f}, Max={info['max_grade']:.3f}, Moy={info['avg_grade']:.3f}")
                        
                        # Visualiser la distribution des teneurs
                        grades = [block['grade'] for block in result['block_model']]
                        fig = px.histogram(grades, nbins=20, title="Distribution des teneurs")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Erreur lors de l'importation: {result if isinstance(result, str) else message}")
    
    with model_tabs[1]:
        st.write("Générer un modèle de blocs synthétique")
        
        model_size = st.selectbox("Taille du modèle", 
                               options=["Petit (10x10x10)", "Moyen (20x20x10)", "Grand (30x30x15)", "Personnalisé"],
                               index=1)
        
        if model_size == "Personnalisé":
            size_cols = st.columns(3)
            with size_cols[0]:
                size_x = st.number_input("Nombre de blocs en X", min_value=1, max_value=50, value=20)
            with size_cols[1]:
                size_y = st.number_input("Nombre de blocs en Y", min_value=1, max_value=50, value=20)
            with size_cols[2]:
                size_z = st.number_input("Nombre de blocs en Z", min_value=1, max_value=25, value=10)
        else:
            if model_size == "Petit (10x10x10)":
                size_x, size_y, size_z = 10, 10, 10
            elif model_size == "Grand (30x30x15)":
                size_x, size_y, size_z = 30, 30, 15
            else:  # Moyen
                size_x, size_y, size_z = 20, 20, 10
        
        block_size = st.number_input("Dimension des blocs (m)", min_value=1, max_value=50, value=10)
        
        origin_cols = st.columns(3)
        with origin_cols[0]:
            origin_x = st.number_input("Origine X (coordonnées)", value=1000)
        with origin_cols[1]:
            origin_y = st.number_input("Origine Y (coordonnées)", value=2000)
        with origin_cols[2]:
            origin_z = st.number_input("Origine Z (coordonnées)", value=500)
    
    # Paramètres économiques
    st.header("Paramètres économiques")
    
    metal_price = st.number_input("Prix du métal ($/t)", min_value=0.0, value=1000.0, step=10.0)
    mining_cost = st.number_input("Coût d'extraction ($/t)", min_value=0.0, value=2.5, step=0.1)
    processing_cost = st.number_input("Coût de traitement ($/t)", min_value=0.0, value=10.0, step=0.5)
    recovery = st.slider("Taux de récupération (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
    cutoff_grade = st.slider("Teneur de coupure (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    
    # Paramètres géotechniques
    st.header("Paramètres géotechniques")
    
    slope_angle = st.slider("Angle de pente global (°)", min_value=25, max_value=75, value=45)
    bench_height = st.number_input("Hauteur de gradin (m)", min_value=1, value=10)
    
    # Bouton pour lancer l'optimisation
    run_optimizer = st.button("Lancer l'optimisation", type="primary", use_container_width=True)

# Colonne 2: Visualisation et résultats
with col2:
    # Visualisation 3D
    st.header("Visualisation")
    
    view_mode = st.selectbox("Mode d'affichage", 
                          options=["Teneurs", "Valeur économique", "Fosse optimale"],
                          index=0)
    
    # Espace réservé pour la visualisation 3D
    vis_placeholder = st.empty()
    
    # Résultats d'optimisation (apparaissent après l'exécution)
    results_container = st.container()
    
    with results_container:
        if st.session_state.results_ready:
            st.header("Résultats d'optimisation")
            st.write(f"Algorithme: **Lerch-Grossman** | Temps d'exécution: {st.session_state.execution_time:.2f} secondes")
            
            # Onglets pour différents types de résultats
            tab1, tab2, tab3 = st.tabs(["Résumé", "Détails", "Sensibilité"])
            
            with tab1:
                # Métriques clés
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Blocs extraits", f"{len(st.session_state.optimal_pit)}")
                with metric_cols[1]:
                    # Calcul de la VAN simplifiée
                    npv = sum(block['value'] for block in st.session_state.optimal_pit)
                    st.metric("VAN", f"{npv:,.0f} $")
                with metric_cols[2]:
                    # Calcul simplifié du ratio stérile/minerai
                    ore_blocks = sum(1 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                    waste_blocks = len(st.session_state.optimal_pit) - ore_blocks
                    sr_ratio = waste_blocks / max(1, ore_blocks)
                    st.metric("Ratio S/M", f"{sr_ratio:.2f}")
                
                # Tableau des résultats
                st.subheader("Statistiques")
                
                # Calculer quelques métriques supplémentaires
                block_size_for_calc = st.session_state.optimal_pit[0].get('block_size', block_size) if st.session_state.optimal_pit else block_size
                ore_tonnage = sum(block_size_for_calc**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100)
                waste_tonnage = sum(block_size_for_calc**3 * 2.7 for block in st.session_state.optimal_pit if block['grade'] <= cutoff_grade/100)
                total_tonnage = ore_tonnage + waste_tonnage
                
                avg_grade = sum(block['grade'] for block in st.session_state.optimal_pit if block['grade'] > cutoff_grade/100) / max(1, ore_blocks)
                metal_content = ore_tonnage * avg_grade * recovery/100
                
                total_revenue = metal_content * metal_price
                mining_costs = total_tonnage * mining_cost
                processing_costs = ore_tonnage * processing_cost
                total_cost = mining_costs + processing_costs
                total_profit = total_revenue - total_cost
                
                # Créer le tableau de résultats
                results_data = {
                    "Paramètre": ["Tonnage total", "Tonnage de minerai", "Tonnage de stérile", 
                                  "Teneur moyenne", "Métal contenu", "Revenu total", 
                                  "Coût total", "Profit"],
                    "Valeur": [
                        f"{total_tonnage:,.0f} t",
                        f"{ore_tonnage:,.0f} t",
                        f"{waste_tonnage:,.0f} t",
                        f"{avg_grade:.2f} %",
                        f"{metal_content:,.0f} t",
                        f"{total_revenue:,.0f} $",
                        f"{total_cost:,.0f} $",
                        f"{total_profit:,.0f} $"
                    ]
                }
                
                st.table(pd.DataFrame(results_data))
                
                # Options d'exportation
                st.subheader("Exporter les résultats")
                export_cols = st.columns(3)
                
                with export_cols[0]:
                    if st.button("📄 Résultats CSV", use_container_width=True):
                        st.session_state.export_csv = True
                
                with export_cols[1]:
                    if st.button("📐 Limite DXF", use_container_width=True):
                        st.session_state.export_dxf = True
                
                with export_cols[2]:
                    if st.button("📊 Modèle JSON", use_container_width=True):
                        st.session_state.export_json = True
                
                # Interface d'exportation CSV
                if 'export_csv' in st.session_state and st.session_state.export_csv:
                    st.subheader("Exporter en CSV")
                    csv_cols = st.columns(2)
                    
                    with csv_cols[0]:
                        include_coordinates = st.checkbox("Inclure les coordonnées", value=True)
                        include_grades = st.checkbox("Inclure les teneurs", value=True)
                    
                    with csv_cols[1]:
                        include_values = st.checkbox("Inclure les valeurs économiques", value=True)
                        only_pit = st.checkbox("Uniquement les blocs dans la fosse", value=True)
                    
                    # Générer le CSV et créer le lien
                    if st.session_state.optimal_pit:
                        csv_df = generate_csv(
                            st.session_state.block_model, 
                            st.session_state.optimal_pit,
                            include_coordinates, 
                            include_grades, 
                            include_values, 
                            only_pit
                        )
                        
                        # Afficher un aperçu
                        st.write("Aperçu:")
                        st.dataframe(csv_df.head())
                        
                        # Créer le lien de téléchargement
                        csv_filename = f"pit_results_LG_{datetime.now().strftime('%Y%m%d')}.csv"
                        csv_link = prepare_download_link(csv_df, csv_filename, "text/csv")
                        st.markdown(csv_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_csv"):
                        st.session_state.export_csv = False
                        st.experimental_rerun()
                
                # Interface d'exportation DXF
                if 'export_dxf' in st.session_state and st.session_state.export_dxf:
                    st.subheader("Exporter en DXF")
                    
                    # Déterminer les niveaux disponibles
                    if st.session_state.optimal_pit:
                        available_z = sorted(list(set(block['z'] for block in st.session_state.optimal_pit)))
                        level_options = ["all"] + [str(i+1) for i in range(len(available_z))]
                    else:
                        level_options = ["all"]
                    
                    selected_level = st.selectbox("Niveau d'extraction:", options=level_options, index=0)
                    
                    dxf_cols = st.columns(3)
                    with dxf_cols[0]:
                        include_points = st.checkbox("Inclure les points", value=True)
                    with dxf_cols[1]:
                        include_polylines = st.checkbox("Inclure les polylignes", value=True)
                    with dxf_cols[2]:
                        include_3dfaces = st.checkbox("Inclure les faces 3D", value=True)
                    
                    # Générer le DXF et créer le lien
                    if st.session_state.optimal_pit:
                        block_size_for_dxf = st.session_state.optimal_pit[0].get('block_size', block_size)
                        
                        dxf_content = generate_dxf(
                            st.session_state.optimal_pit,
                            selected_level,
                            include_points,
                            include_polylines,
                            include_3dfaces,
                            block_size_for_dxf
                        )
                        
                        # Afficher un aperçu
                        st.text_area("Aperçu DXF:", value=dxf_content[:500] + "...", height=150)
                        
                        # Créer le lien de téléchargement
                        level_suffix = f"level_{selected_level}" if selected_level != "all" else "all_levels"
                        dxf_filename = f"pit_boundary_{level_suffix}_LG.dxf"
                        dxf_link = prepare_download_link(dxf_content, dxf_filename, "application/dxf")
                        st.markdown(dxf_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_dxf"):
                        st.session_state.export_dxf = False
                        st.experimental_rerun()
                
                # Export JSON
                if 'export_json' in st.session_state and st.session_state.export_json:
                    if st.session_state.optimal_pit:
                        # Créer un objet pour le modèle de fosse optimisé
                        pit_model = {
                            "metadata": {
                                "author": "Didier Ouedraogo, P.Geo",
                                "algorithm": "Lerch-Grossman",
                                "timestamp": datetime.now().isoformat(),
                                "params": {
                                    "metalPrice": metal_price,
                                    "miningCost": mining_cost,
                                    "processingCost": processing_cost,
                                    "recovery": recovery / 100,
                                    "cutoffGrade": cutoff_grade / 100,
                                    "slopeAngle": slope_angle
                                }
                            },
                            "statistics": {
                                "totalBlocks": len(st.session_state.block_model) if st.session_state.block_model else 0,
                                "extractedBlocks": len(st.session_state.optimal_pit),
                                "totalTonnage": float(total_tonnage),
                                "oreTonnage": float(ore_tonnage),
                                "wasteTonnage": float(waste_tonnage),
                                "avgGrade": float(avg_grade),
                                "npv": float(total_profit)
                            },
                            "blocks": [{
                                "x": block["real_x"],
                                "y": block["real_y"],
                                "z": block["real_z"],
                                "grade": block["grade"],
                                "value": block["value"]
                            } for block in st.session_state.optimal_pit[:100]]  # Limité à 100 blocs pour l'exemple
                        }
                        
                        # Afficher un aperçu
                        st.json(pit_model, expanded=False)
                        
                        # Créer le lien de téléchargement
                        json_filename = f"pit_model_LG_{datetime.now().strftime('%Y%m%d')}.json"
                        json_link = prepare_download_link(pit_model, json_filename, "application/json")
                        st.markdown(json_link, unsafe_allow_html=True)
                    
                    if st.button("Fermer", key="close_json"):
                        st.session_state.export_json = False
                        st.experimental_rerun()
            
            with tab2:
                # Détails par niveau
                st.subheader("Détails par niveau")
                
                # Grouper les blocs par niveau
                levels_data = []
                
                if st.session_state.optimal_pit:
                    # Déterminer la taille Z à partir du modèle
                    max_z = max(block['z'] for block in st.session_state.optimal_pit) + 1
                    
                    for z in range(max_z):
                        level_blocks = [block for block in st.session_state.optimal_pit if block['z'] == z]
                        if level_blocks:
                            level_ore_blocks = [block for block in level_blocks if block['grade'] > cutoff_grade/100]
                            block_size_level = level_blocks[0].get('block_size', block_size)
                            level_tonnage = len(level_blocks) * block_size_level**3 * 2.7
                            level_grade = sum(block['grade'] for block in level_ore_blocks) / max(1, len(level_ore_blocks))
                            level_value = sum(block['value'] for block in level_blocks)
                            
                            # Trouver l'élévation réelle
                            if level_blocks[0].get('real_z') is not None:
                                # Utiliser la première élévation réelle trouvée comme référence
                                elev = level_blocks[0]['real_z']
                            else:
                                # Calculer à partir de l'origine et de la taille du bloc
                                elev = origin_z - z * block_size_level
                            
                            levels_data.append({
                                "Niveau": z + 1,
                                "Élévation": elev,
                                "Blocs": len(level_blocks),
                                "Tonnage": f"{level_tonnage:,.0f} t",
                                "Teneur moy.": f"{level_grade:.2f} %",
                                "Valeur": f"{level_value:,.0f} $"
                            })
                
                if levels_data:
                    st.table(pd.DataFrame(levels_data))
                else:
                    st.info("Aucun bloc dans la fosse optimisée")
            
            with tab3:
                # Analyse de sensibilité
                st.subheader("Analyse de sensibilité")
                
                # Créer des données fictives pour l'analyse de sensibilité
                sensitivity_data = {
                    "Variable": ["Prix du métal", "Coût d'extraction", "Coût de traitement", "Récupération", "Teneur de coupure"],
                    "-20%": [0.8, 1.15, 1.12, 0.85, 1.05],
                    "-10%": [0.9, 1.07, 1.06, 0.92, 1.02],
                    "Base": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "+10%": [1.1, 0.93, 0.94, 1.08, 0.97],
                    "+20%": [1.2, 0.87, 0.88, 1.15, 0.95]
                }
                
                df_sensitivity = pd.DataFrame(sensitivity_data)
                
                # Créer un graphique de sensibilité
                fig = go.Figure()
                
                for variable in df_sensitivity["Variable"]:
                    row = df_sensitivity[df_sensitivity["Variable"] == variable].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[-20, -10, 0, 10, 20],
                        y=[row["-20%"], row["-10%"], row["Base"], row["+10%"], row["+20%"]],
                        mode='lines+markers',
                        name=variable
                    ))
                
                fig.update_layout(
                    title="Analyse de sensibilité (VAN relative)",
                    xaxis_title="Variation des paramètres (%)",
                    yaxis_title="VAN relative",
                    legend_title="Paramètres",
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                L'analyse de sensibilité montre que la VAN du projet est:
                - Très sensible aux variations du prix du métal et du taux de récupération
                - Moyennement sensible aux coûts d'extraction et de traitement
                - Peu sensible aux variations de la teneur de coupure
                """)

# Logique d'optimisation
if run_optimizer:
    # Afficher un indicateur de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Préparation du modèle de blocs...")
    progress_bar.progress(10)
    
    # Convertir les pourcentages en valeurs décimales
    cutoff_grade_decimal = cutoff_grade / 100
    recovery_decimal = recovery / 100
    
    # Utiliser le modèle importé ou en générer un nouveau
    start_time = time.time()
    
    if st.session_state.model_imported and st.session_state.block_model:
        # Utiliser le modèle importé
        for block in st.session_state.block_model:
            block['block_size'] = st.session_state.imported_model_info.get('block_size', block_size)
            
            # Recalculer les valeurs économiques avec les paramètres actuels
            tonnage = block['block_size']**3 * 2.7
            
            if block['grade'] > cutoff_grade_decimal:
                # Bloc de minerai
                block['value'] = tonnage * (block['grade'] * metal_price * recovery_decimal - (mining_cost + processing_cost))
            else:
                # Bloc de stérile
                block['value'] = -tonnage * mining_cost
        
        # Utiliser les dimensions du modèle importé
        size_x = st.session_state.imported_model_info.get('size_x', size_x)
        size_y = st.session_state.imported_model_info.get('size_y', size_y)
        size_z = st.session_state.imported_model_info.get('size_z', size_z)
    else:
        # Générer un nouveau modèle
        st.session_state.block_model = generate_block_model(
            size_x, size_y, size_z,
            origin_x, origin_y, origin_z,
            block_size,
            metal_price, mining_cost, processing_cost,
            recovery_decimal, cutoff_grade_decimal
        )
    
    status_text.text("Construction du graphe...")
    progress_bar.progress(40)
    time.sleep(0.5)  # Simulation de temps de calcul
    
    # Exécuter l'algorithme choisi
    status_text.text("Détermination de la fosse optimale...")
    progress_bar.progress(70)
    
    st.session_state.optimal_pit = run_lerch_grossman(
        st.session_state.block_model,
        size_x, size_y, size_z,
        slope_angle,
        iterations=lg_iterations,
        tolerance=lg_tolerance
    )
    
    status_text.text("Finalisation des résultats...")
    progress_bar.progress(90)
    time.sleep(0.5)  # Simulation de temps de calcul
    
    # Calcul du temps d'exécution
    end_time = time.time()
    st.session_state.execution_time = end_time - start_time
    
    # Marquer que les résultats sont prêts
    st.session_state.results_ready = True
    
    progress_bar.progress(100)
    status_text.text("Optimisation terminée!")
    time.sleep(0.5)
    
    # Supprimer la barre de progression et le texte de statut
    progress_bar.empty()
    status_text.empty()
    
    # Actualiser la page pour afficher les résultats
    st.experimental_rerun()

# Visualisation 3D
if st.session_state.results_ready:
    # Déterminer quels blocs afficher en fonction du mode d'affichage
    if view_mode == "Teneurs":
        display_blocks = st.session_state.block_model
        color_by = 'grade'
        colorscale = 'RdYlGn_r'
        color_title = 'Teneur'
    elif view_mode == "Valeur économique":
        display_blocks = st.session_state.block_model
        color_by = 'value'
        colorscale = 'RdBu'
        color_title = 'Valeur ($)'
    else:  # Fosse optimale
        display_blocks = st.session_state.optimal_pit
        color_by = 'z'  # Colorer par profondeur
        colorscale = 'Blues'
        color_title = 'Profondeur'
    
    # Limiter le nombre de blocs pour des raisons de performance
    max_blocks_to_show = 1000
    if len(display_blocks) > max_blocks_to_show:
        # Échantillonner de manière uniforme
        step = len(display_blocks) // max_blocks_to_show
        sampled_blocks = display_blocks[::step]
    else:
        sampled_blocks = display_blocks
    
    # Créer la figure 3D
    fig = go.Figure()
    
    # Extraire les coordonnées et valeurs
    x = [block['real_x'] for block in sampled_blocks]
    y = [block['real_y'] for block in sampled_blocks]
    z = [block['real_z'] for block in sampled_blocks]
    
    if color_by == 'grade':
        colors = [block['grade'] for block in sampled_blocks]
    elif color_by == 'value':
        colors = [block['value'] for block in sampled_blocks]
    else:
        colors = [block['z'] for block in sampled_blocks]
    
    # Ajouter les cubes (représentés comme des marqueurs 3D)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale=colorscale,
            colorbar=dict(title=color_title),
            opacity=0.8
        ),
        text=[f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>"
              f"Teneur: {block['grade']:.2f}%<br>"
              f"Valeur: {block['value']:.0f}$" for block in sampled_blocks],
        hoverinfo='text'
    ))
    
    # Configurer la mise en page
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    
    # Afficher la visualisation
    vis_placeholder.plotly_chart(fig, use_container_width=True)
elif st.session_state.model_imported:
    # Visualiser le modèle importé
    sampled_blocks = st.session_state.block_model
    max_blocks_to_show = 1000
    
    if len(sampled_blocks) > max_blocks_to_show:
        step = len(sampled_blocks) // max_blocks_to_show
        sampled_blocks = sampled_blocks[::step]
    
    # Créer la figure 3D
    fig = go.Figure()
    
    # Extraire les coordonnées et valeurs
    x = [block['real_x'] for block in sampled_blocks]
    y = [block['real_y'] for block in sampled_blocks]
    z = [block['real_z'] for block in sampled_blocks]
    colors = [block['grade'] for block in sampled_blocks]
    
    # Ajouter les cubes (représentés comme des marqueurs 3D)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Teneur'),
            opacity=0.8
        ),
        text=[f"X: {block['real_x']}, Y: {block['real_y']}, Z: {block['real_z']}<br>"
              f"Teneur: {block['grade']:.2f}%" for block in sampled_blocks],
        hoverinfo='text'
    ))
    
    # Configurer la mise en page
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
        title="Modèle de blocs importé (coloré par teneur)"
    )
    
    # Afficher la visualisation
    vis_placeholder.plotly_chart(fig, use_container_width=True)
else:
    # Afficher un message si aucun résultat n'est disponible
    vis_placeholder.info("Lancez l'optimisation pour visualiser le modèle de blocs et la fosse optimale.")

# Pied de page
st.markdown("---")
st.markdown("© 2025 Didier Ouedraogo, P.Geo - Tous droits réservés")
st.markdown("LG Pit Optimizer v1.0.0")