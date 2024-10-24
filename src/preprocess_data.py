import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def remove_rare_categories(df, column, min_count=10):
    """
    Supprime les lignes contenant des catégories rares dans une variable catégorielle.

    :param df: DataFrame contenant les données
    :param column: Nom de la colonne catégorielle
    :param min_count: Seuil d'occurrence en dessous duquel supprimer les catégories (par défaut 10)
    :return: DataFrame filtré
    """
    # Compter les occurrences des catégories
    category_counts = df[column].value_counts()
    
    # Garder uniquement les catégories avec suffisamment d'occurrences
    common_categories = category_counts[category_counts >= min_count].index
    
    # Filtrer les lignes avec des catégories rares
    df_filtered = df[df[column].isin(common_categories)]
    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered

def preprocess_car_model(data, column, threshold):
    """
    Prétraite la variable car_model en regroupant les modèles ayant moins de seuil d'occurrences
    dans une catégorie 'marque + Autre'.
    :param data: DataFrame contenant les données
    :param column: Nom de la colonne car_model
    :param threshold: Nombre minimal d'occurrences pour qu'un modèle soit conservé
    """
    # Compter les occurrences de chaque modèle
    counts = data[column].value_counts()
    
    # Identifier les modèles rares
    rare_models = counts[counts < threshold].index
    
    # Remplacer les modèles rares par 'marque + Autre'
    data[column] = data[column].apply(lambda x: ' '.join(x.split()[0:1]) + ' Autre' if x in rare_models else x)
    
    return data

def merge_categories(data, column, threshold, other_label='Autre'):
    """
    Rassemble les catégories d'une variable catégorielle inférieures à un certain seuil d'occurrences 
    en une seule catégorie 'Autre'.
    :param data: DataFrame contenant les données
    :param colonne: Nom de la colonne catégorielle à modifier
    :param threshold: Nombre minimal d'occurrences pour qu'une catégorie soit conservée
    """
    # Count the occurrences of each category
    counts = data[column].value_counts()
    
    # Define rare categories
    rare_categories = counts[counts < threshold].index
    
    # Replace rare categories with 'Other'
    data[column] = data[column].apply(lambda x: other_label if x in rare_categories else x)
    
    return data

def impute_by_car_model(df, target_column, reference_column='car_model'):
    """
    Impute les valeurs manquantes de la colonne cible (target_column) en utilisant la valeur
    la plus fréquente (mode) dans le modèle de voiture correspondant.

    :param df: DataFrame contenant les données
    :param target_column: Colonne à laquelle appliquer l'imputation (ex: 'doors', 'seats')
    :param reference_column: Colonne de référence pour grouper les données (ex: 'car_model')
    :return: DataFrame avec les valeurs manquantes imputées dans la colonne cible
    """
    
    # Trouver le mode (valeur la plus fréquente) du target_column pour chaque car_model
    mode_by_model = df.groupby(reference_column)[target_column].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    
    # Fonction pour remplir les valeurs manquantes
    def fill_missing_value(row):
        if pd.isnull(row[target_column]):
            return mode_by_model.get(row[reference_column], row[target_column])
        return row[target_column]
    
    # Appliquer l'imputation sur les lignes où target_column est manquant
    df[target_column] = df.apply(fill_missing_value, axis=1)
    
    return df


def impute_with_random_forest(data, target_column, numerical_columns, categorical_columns=None, n_estimators=100):
    """
    Impute les valeurs manquantes d'une variable numérique en utilisant un modèle Random Forest.
    
    :param df: DataFrame contenant les données
    :param target_column: La colonne cible où les valeurs manquantes doivent être remplies (par ex. 'horsepower')
    :param numerical_columns: Liste des colonnes numériques à utiliser comme prédicteurs (par ex. 'price', 'engine_size')
    :param categorical_columns: Liste des colonnes catégorielles à utiliser comme prédicteurs (par ex. 'make', 'fuel_type')
    :param n_estimators: Nombre d'arbres dans la forêt aléatoire (par défaut 100)
    :return: DataFrame avec les valeurs manquantes de la colonne cible imputées
    """
    
    # Copie du DataFrame pour éviter de modifier l'original
    df = data.copy()
    # Vérifier s'il y a des valeurs manquantes à imputer
    if df[target_column].isna().sum() == 0:
        print(f"Aucune valeur manquante trouvée dans {target_column}.")
        return df

    # Gérer les variables catégorielles en utilisant get_dummies (One-Hot Encoding)
    df=df[categorical_columns+numerical_columns+[target_column]]
    if categorical_columns is not None:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Séparer les lignes avec et sans valeurs manquantes dans la variable cible
    df_missing = df[df[target_column].isna()]
    df_non_missing = df[df[target_column].notna()]

    # Les variables explicatives pour le modèle (exclure la colonne cible)
    predictors = numerical_columns + [col for col in df.columns if col not in [target_column] + numerical_columns]

    # Entraînement du modèle sur les données complètes
    X_train = df_non_missing[predictors]
    y_train = df_non_missing[target_column]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Prédire les valeurs manquantes
    X_missing = df_missing[predictors]
    data.loc[data[target_column].isna(), target_column] = model.predict(X_missing)

    return data

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    percent = round(percent,1)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def process_categorical(df, cat_columns, encoding_method='onehot', target=None):
    """
    Traite les variables catégorielles en utilisant différentes méthodes d'encodage.
    
    :param df: DataFrame contenant les données.
    :param cat_columns: Liste des colonnes catégorielles à encoder.
    :param encoding_method: Méthode d'encodage à utiliser. Peut être 'onehot', 'label', 'frequency', 'target', 'binary'.
    :param target: Nom de la colonne cible à utiliser pour Target Encoding (requis pour 'target').
    :return: DataFrame avec les colonnes catégorielles traitées.
    """
    
    df_processed = df.copy()  # Éviter de modifier le DataFrame original

    if encoding_method == 'onehot':
        # One-Hot Encoding
        df_processed = pd.get_dummies(df_processed, columns=cat_columns, drop_first=True)

    elif encoding_method == 'label':
        # Label Encoding (ordonné ou non ordonné)
        for col in cat_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

    elif encoding_method == 'frequency':
        # Frequency Encoding
        for col in cat_columns:
            freq = df_processed[col].value_counts() / len(df_processed)
            df_processed[col] = df_processed[col].map(freq)

    elif encoding_method == 'target':
        if target is None:
            raise ValueError("Target column is required for target encoding.")
        # Target Encoding
        for col in cat_columns:
            target_mean = df_processed.groupby(col)[target].mean()
            df_processed[col] = df_processed[col].map(target_mean)

    elif encoding_method == 'binary':
        # Binary Encoding
        encoder = BinaryEncoder(cols=cat_columns)
        df_processed = encoder.fit_transform(df_processed)

    else:
        raise ValueError("Unsupported encoding method. Choose from 'onehot', 'label', 'frequency', 'target', or 'binary'.")

    return df_processed

def normalize_data(data, colonnes_numeriques, methode='minmax'):
    """
    Normalise les colonnes numériques d'un DataFrame avec la méthode choisie
    :param data: DataFrame contenant les données
    :param colonnes_numeriques: Liste des colonnes numériques à normaliser
    :param methode: Méthode de normalisation ('minmax', 'standard', 'robust')
    :return: DataFrame avec les colonnes normalisées
    """
    if methode == 'minmax':
        scaler = MinMaxScaler()
    elif methode == 'standard':
        scaler = StandardScaler()
    elif methode == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("La méthode choisie n'est pas supportée. Choisissez parmi 'minmax', 'standard', ou 'robust'.")
    
    data[colonnes_numeriques] = scaler.fit_transform(data[colonnes_numeriques])
    
    return data

def extract_additional_information(text, patterns=None):
    """
    Extrait des informations additionnelles pertinentes des descriptions d'annonces de voitures.
    
    Parameters:
    - text (str): La description de l'annonce.
    - min_occurrences (int): Nombre minimum d'occurrences pour considérer un mot.
    - patterns (dict): Dictionnaire des motifs (patterns) à rechercher avec leurs clés correspondantes.
    
    Returns:
    - dict: Informations extraites.
    """
    
    # Prétraiter le texte
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Extraire les informations selon les motifs spécifiés
    extracted_info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        extracted_info[key] = match.group(0) if match else None
    
    return extracted_info