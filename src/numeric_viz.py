import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_numeric_boxplots(df, figsize=(14, 10), n_cols=3):
    """
    Visualise des boxplots pour toutes les variables numériques d'un DataFrame avec des subplots organisés.

    :param df: DataFrame contenant les données
    :param figsize: Taille de la figure globale (par défaut (14, 10))
    :param n_cols: Nombre de colonnes dans la grille de subplots (par défaut 3)
    """
    # Sélectionner uniquement les colonnes numériques
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Nombre total de colonnes numériques
    n_numeric = len(numerical_cols)

    # Définir le nombre de lignes nécessaires pour les subplots
    n_rows = (n_numeric + n_cols - 1) // n_cols

    # Initialiser la figure avec la taille définie
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Applatir la matrice des axes pour une manipulation plus simple

    # Parcourir chaque colonne numérique et créer un boxplot
    for i, col in enumerate(numerical_cols):
        sns.boxplot(data=df, y=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Boxplot of {col}', fontsize=12)
        axes[i].set_ylabel(col)

    # Supprimer les axes restants si le nombre de colonnes est inférieur à la grille
    for i in range(n_numeric, len(axes)):
        fig.delaxes(axes[i])

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()
    plt.show()

def plot_numeric_distributions(df, figsize=(14, 10), n_cols=3, bins=20):
    """
    Visualise la distribution (histogrammes) pour toutes les variables numériques d'un DataFrame avec des subplots organisés.

    :param df: DataFrame contenant les données
    :param figsize: Taille de la figure globale (par défaut (14, 10))
    :param n_cols: Nombre de colonnes dans la grille de subplots (par défaut 3)
    :param bins: Nombre de classes pour les histogrammes (par défaut 20)
    """
    # Sélectionner uniquement les colonnes numériques
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Nombre total de colonnes numériques
    n_numeric = len(numerical_cols)

    # Définir le nombre de lignes nécessaires pour les subplots
    n_rows = (n_numeric + n_cols - 1) // n_cols

    # Initialiser la figure avec la taille définie
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Applatir la matrice des axes pour une manipulation plus simple

    # Parcourir chaque colonne numérique et créer un histogramme
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, bins=bins, kde=True, ax=axes[i], color="blue")
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Supprimer les axes restants si le nombre de colonnes est inférieur à la grille
    for i in range(n_numeric, len(axes)):
        fig.delaxes(axes[i])

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()
    plt.show()
    

def plot_correlation_matrix(df, threshold=0.8,figsize=(22, 14)):
    """
    Affiche une matrice de corrélation pour identifier les variables fortement corrélées.
    
    :param df: DataFrame avec les variables numériques
    :param threshold: Seuil de corrélation pour identifier la colinéarité (par défaut 0.8)
    :return: Liste des paires de variables corrélées au-dessus du seuil
    """
    # Calculer la matrice de corrélation
    corr_matrix = df.corr()
    
    # Masquer la moitié supérieure de la matrice (car elle est symétrique)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Afficher la matrice de corrélation avec seaborn
    plt.figure(figsize= figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1,linewidths=1)
    plt.show()
    
    # Identifier les paires fortement corrélées
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    return correlated_pairs  