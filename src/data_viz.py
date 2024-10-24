import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def countplots(df, columns, top_n=10, figsize=(10, 6)):
    """
    Trie les catégories par fréquence, filtre les plus représentées, et crée des countplots pour plusieurs colonnes.

    :param df: DataFrame contenant les données
    :param columns: Liste des colonnes pour lesquelles générer des countplots
    :param top_n: Nombre de catégories les plus représentées à afficher (par défaut 10)
    :param figsize: Taille de la figure globale (par défaut (14, 10))
    """
    # Définir le nombre de sous-graphes
    n = len(columns)

    # Déterminer la disposition des subplots en fonction du nombre de colonnes
    rows = (n + 1) // 2
    cols = 2 if n > 1 else 1

    # Initialiser la figure
    plt.figure(figsize=figsize)

    # Générer un countplot pour chaque colonne
    for i, column in enumerate(columns):
        plt.subplot(rows, cols, i + 1)

        # Filtrer et trier les catégories par fréquence
        counts = df[column].value_counts().nlargest(top_n)
        order = counts.sort_values(ascending=False).index


        ax = sns.countplot(y=column, data=df[df[column].isin(order)], order=order)
        # Ajouter les pourcentages à côté des barres
        for p in ax.patches:
             percentage = f'{100 * p.get_width() / len(df[df[column].isin(order)]):.1f}%'
             ax.annotate(percentage, (p.get_width() + 0.2, p.get_y() + p.get_height() / 2.), 
                            ha='center', va='baseline', fontsize=12, color='black')
        plt.xlabel('Count')
        plt.ylabel(column)
        # Ajouter un titre
        plt.title(f'Countplot {column} (Top {top_n})')

    # Ajuster l'espacement entre les sous-graphes
    plt.tight_layout()
    plt.show()

def plot_avg_price(data, category_col, value_col='price', min_count=100, method='mean', figsize=(10, 6), top_n=10):
    """
    Visualise le prix par catégorie dans un DataFrame, avec option de méthode de calcul.
    
    Parameters:
    - data (pd.DataFrame): Le DataFrame contenant les données.
    - category_col (str): Le nom de la colonne catégorielle à analyser.
    - value_col (str): Le nom de la colonne des valeurs à calculer (par défaut 'price').
    - min_count (int): Le nombre minimum d'observations pour inclure une catégorie (par défaut 100).
    - method (str): Méthode de calcul ('mean', 'median', 'max', 'min').
    - figsize (tuple): La taille de la figure (par défaut (12, 8)).
    - top_n (int): Le nombre maximum de catégories à afficher (par défaut 10).
    """
    
    # Filtrer les catégories avec plus de 'min_count' observations
    category_counts = data[category_col].value_counts()
    categories_to_keep = category_counts[category_counts >= min_count].index
    filtered_df = data[data[category_col].isin(categories_to_keep)]
    
    # Calculer le prix par catégorie selon la méthode choisie
    if method == 'mean':
        method = 'moyen'
        avg_price_per_category = filtered_df.groupby(category_col)[value_col].mean().sort_values(ascending=False)
    elif method == 'median':
        avg_price_per_category = filtered_df.groupby(category_col)[value_col].median().sort_values(ascending=False)
    elif method == 'max':
        avg_price_per_category = filtered_df.groupby(category_col)[value_col].max().sort_values(ascending=False)
    elif method == 'min':
        avg_price_per_category = filtered_df.groupby(category_col)[value_col].min().sort_values(ascending=False)
    else:
        raise ValueError("La méthode doit être 'mean', 'median', 'max' ou 'min'.")
    
    # Préparer les données pour la visualisation
    categories = avg_price_per_category.index[:top_n]
    avg_prices = avg_price_per_category.values[:top_n]
    
    # Créer le graphique
    plt.figure(figsize=figsize)
    sns.barplot(x=avg_prices, y=categories)
    title = f"Prix {method} par {category_col}"
    plt.title(title)
    plt.xlabel(f"Prix {method} (€)")
    plt.ylabel(category_col.capitalize())
    plt.tight_layout()
    plt.show()

def plot_comparative_price(df, model_col, price_col, seller_type_col, min_occurrences=10):
    """
    Visualise un graphique de lignes comparatives des prix moyens des voitures
    entre les types d'annonce particulier et professionnelle selon le modèle de voiture,
    avec possibilité de filtrer sur le nombre minimum d'occurrences par modèle.

    :param df: DataFrame contenant les données des annonces de voitures
    :param model_col: Nom de la colonne des modèles de voitures
    :param price_col: Nom de la colonne des prix des voitures
    :param seller_type_col: Nom de la colonne indiquant le type d'annonce (particulier ou professionnel)
    :param min_occurrences: Nombre minimum d'occurrences pour inclure un modèle (par défaut 10)
    """
    # Filtrer les modèles qui ont au moins 'min_occurrences' annonces
    model_counts = df[model_col].value_counts()
    valid_models = model_counts[model_counts >= min_occurrences].index

    # Filtrer le DataFrame en fonction des modèles valides
    filtered_df = df[df[model_col].isin(valid_models)]

    # Calcul du prix moyen des voitures selon le modèle et le type de vendeur
    avg_prices = filtered_df.groupby([model_col, seller_type_col])[price_col].median().reset_index()

    # Pivot pour avoir une structure facilitant la comparaison entre particulier et professionnel
    avg_prices_pivot = avg_prices.pivot(index=model_col, columns=seller_type_col, values=price_col)

    # Créer le graphique de lignes comparatives
    plt.figure(figsize=(12, 8))
    
    # Tracer les lignes pour les annonces des particuliers et des professionnels
    for seller_type in avg_prices_pivot.columns:
        plt.plot(avg_prices_pivot.index, avg_prices_pivot[seller_type], label=seller_type)

    # Ajout de légendes, titres et labels
    plt.title(f'Comparaison des prix moyens entre particulier et professionnel')
    plt.xlabel('Modèle de voiture')
    plt.ylabel('Prix moyen (€)')
    plt.xticks(rotation=90)
    plt.legend(title='Type d\'annonce')
    plt.grid(True)

    # Affichage du graphique
    plt.tight_layout()
    plt.show()

def plot_price_evolution(data, date_colonne, prix_colonne, min_annonces=100, method='moyen', figsize=(10, 6)):
    """
    Visualise l'évolution des prix par mois avec un nombre minimum d'annonces
    :param data: DataFrame contenant les données
    :param date_colonne: Nom de la colonne contenant les dates de publication
    :param prix_colonne: Nom de la colonne contenant les prix
    :param min_annonces: Nombre minimum d'annonces par mois pour inclure les données
    """
    dff = data.copy()
    # Convertir la colonne 'date_colonne' en format date
    dff[date_colonne] = pd.to_datetime(dff[date_colonne], errors='coerce')

    # Extraire l'année et le mois de la date de publication
    dff['year_month'] = dff[date_colonne].dt.to_period('M')

    # Calculer le nombre d'annonces par mois
    monthly_counts = dff.groupby('year_month').size().reset_index(name='counts')

    # Filtrer les mois avec au moins 'min_annonces' annonces
    valid_months = monthly_counts[monthly_counts['counts'] >= min_annonces]['year_month']
    data_filtered = dff[dff['year_month'].isin(valid_months)]

    # Calculer le prix moyen par mois
    if method == 'moyen':
        monthly_prices = data_filtered.groupby('year_month')[prix_colonne].mean().reset_index()
    elif method == 'median':
        monthly_prices = data_filtered.groupby('year_month')[prix_colonne].median().reset_index()
    elif method == 'max':
        monthly_prices = data_filtered.groupby('year_month')[prix_colonne].max().reset_index()
    elif method == 'min':
        monthly_prices = data_filtered.groupby('year_month')[prix_colonne].min().reset_index()
    else:
        raise ValueError("La méthode doit être 'moyen', 'median', 'max' ou 'min'.")
    

    # Tracer l'évolution des prix
    plt.figure(figsize=figsize)
    plt.plot(monthly_prices['year_month'].astype(str), monthly_prices[prix_colonne], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Prix moyen')
    plt.title(f"Évolution des prix {method} par mois")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
  