import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from tqdm import tqdm


def compare_regression_models(models, X_train, X_test, y_train, y_test, cv=5):
    """
    Compare les performances de plusieurs modèles de régression en termes de MAE et R² score sur les ensembles
    d'entraînement et de test.
    
    Paramètres:
        models (dict): Dictionnaire des modèles de régression à comparer.
        X_train (array-like): Caractéristiques des données d'entraînement.
        X_test (array-like): Caractéristiques des données de test.
        y_train (array-like): Cibles des données d'entraînement.
        y_test (array-like): Cibles des données de test.
        cv (int): Nombre de plis pour la validation croisée (par défaut 5).
    
    Retourne:
        DataFrame: Tableau comparatif des performances des modèles.
    """
    # Liste pour stocker les résultats
    results = []
    
    # Boucle à travers chaque modèle
    for name, model in tqdm(models.items()):
                
        # Validation croisée pour évaluer le modèle sur l'ensemble d'entraînement
        cv_scores_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        mean_cv_mae = cv_scores_mae.mean()
        mean_cv_r2 = cv_scores_r2.mean()
        
        # Entraînement du modèle sur l'ensemble de formation complet
        model.fit(X_train, y_train)
        
        # Prédiction sur l'ensemble de test
        y_pred = model.predict(X_test)
        
        # Calcul du MAE et du score R² sur l'ensemble de test
        mae_test = mean_absolute_error(y_test, y_pred)
        r2_score_test = r2_score(y_test, y_pred)

        
        # Ajout des résultats à la liste
        results.append({
            'Model': name,
            'Mean Absolute Error (Train)': round(mean_cv_mae),
            'R²/Accuracy (Train)': round(mean_cv_r2,2),
            'Mean Absolute Error (Test)': round(mae_test),
            'R²/Accuracy (Test)': round(r2_score_test,2)
        })
    
    # Conversion des résultats en DataFrame pandas pour un affichage clair
    results_df = pd.DataFrame(results)
    
    return results_df

def optimize_model(model, params, X_train, y_train, cv=5, scoring='r2'):
    """
    Optimise les hyperparamètres pour un modèle de ML donné en utilisant GridSearchCV avec une barre de progression tqdm.

    Paramètres:
        modele (objet): Modèle d'apprentissage automatique à optimiser (ex: Ridge, Lasso, etc.).
        grille_params (dict): Dictionnaire avec les noms des paramètres (str)
        comme clés et les listes de valeurs de paramètres à tester.
        X_train (array-like): Caractéristiques des données d'entraînement.
        y_train (array-like): Cibles des données d'entraînement.
        cv (int): Nombre de plis pour la validation croisée (par défaut 5).
        scoring (str): Métrique d'évaluation (par défaut 'r2').

    Retourne:
        dict: Meilleurs paramètres trouvés.
        float: Meilleure score atteint.
    """
    
    tqdm.monitor_interval = 0  # Solution de contournement pour un problème de tqdm dans Jupyter
    
    # Initialiser GridSearchCV avec le modèle et les paramètres fournis
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring=scoring, cv=cv, n_jobs=-1)
    
    with tqdm(total=len(ParameterGrid(params)), desc="Optimisation des hyperparamètres") as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(1)
        
    meilleur_model = grid_search.best_estimator_
    meilleurs_params = grid_search.best_params_
    meilleur_score = round(grid_search.best_score_,2)
    
    return meilleur_model, meilleur_score


def visualize_model_performance(model, X_train, y_train, X_test, y_test,figsize=(18, 15)):
    """
    Visualise les performances d'un modèle de régression avec plusieurs métriques.

    Paramètres:
        model (object): Modèle de régression entraîné.
        X_train (array-like): Données d'entraînement.
        y_train (array-like): Cibles d'entraînement.
        X_test (array-like): Données de test.
        y_test (array-like): Cibles de test.
    """
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calcul des métriques
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Initialiser la figure
    plt.figure(figsize=figsize)
    
    # Valeurs prédites vs Valeurs réelles (Train)
    plt.subplot(3, 2, 1)
    sns.scatterplot(x=y_train, y=y_pred_train)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
    plt.xlabel('Valeurs Réelles (Train)')
    plt.ylabel('Valeurs Prédites (Train)')
    plt.title(f"Train Set\nMAE: {mae_train:.2f}, R²: {r2_train:.2f}")
    
    # Valeurs prédites vs Valeurs réelles (Test)
    plt.subplot(3, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_test)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Valeurs Réelles (Test)')
    plt.ylabel('Valeurs Prédites (Test)')
    plt.title(f"Test Set\nMAE: {mae_test:.2f}, R²: {r2_test:.2f}")
    
    # Résidus (Train)
    plt.subplot(3, 2, 3)
    sns.histplot(y_train - y_pred_train, kde=True)#stat='percent'
    plt.xlabel('Résidus (Train)')
    plt.title('Distribution des Résidus (Train)')
    
    # Résidus (Test)
    plt.subplot(3, 2, 4)
    sns.histplot(y_test - y_pred_test, kde=True)
    plt.xlabel('Résidus (Test)')
    plt.title('Distribution des Résidus (Test)')
    
    # Q-Q Plot (Train)
    plt.subplot(3, 2, 5)
    stats.probplot(y_train - y_pred_train, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Train)')
    
    # Q-Q Plot (Test)
    plt.subplot(3, 2, 6)
    stats.probplot(y_test - y_pred_test, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Test)')
    
    plt.tight_layout()
    plt.show()
    
def plot_feature_importance(model, X_test, original_columns, top_n=10, model_type='linear'):
    """
    Visualise l'importance des caractéristiques sous forme de pourcentage, basée sur:
    - Les valeurs de Shapley pour les modèles linéaires.
    - `model.feature_importances_` pour les modèles d'arbres.
    
    Les variables indicatrices issues de get_dummies sont regroupées, sans impacter les autres variables.

    Paramètres:
    -----------
    model : object
        Le modèle ajusté (linéaire ou arbre).
    X_test : DataFrame
        Les données de test transformées par get_dummies.
    original_columns : list
        La liste des colonnes d'origine avant encodage avec get_dummies.
    top_n : int, optionnel
        Le nombre de caractéristiques à afficher (par défaut 10).
    model_type : str, optionnel
        Le type de modèle ('linear' pour les modèles linéaires, 'tree' pour les modèles d'arbres).
    """
    if model_type == 'linear':
        # Utiliser Shapley pour les modèles linéaires
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
        feature_importances = np.abs(shap_values).mean(axis=0)
    elif model_type == 'tree':
        # Utiliser model.feature_importances_ pour les modèles d'arbres
        feature_importances = model.feature_importances_
    else:
        raise ValueError("Le type de modèle doit être 'linear' ou 'tree'.")

    # Extraire les noms des colonnes après transformation get_dummies
    feature_names = X_test.columns if hasattr(X_test, 'columns') else np.arange(X_test.shape[1])

    # Regrouper uniquement les variables indicatrices créées par get_dummies
    grouped_importance = {}
    
    for feature, importance in zip(feature_names, feature_importances):
        # On vérifie si la variable est une indicatrice générée par get_dummies
        # en vérifiant si la base de cette colonne se trouve dans original_columns
        base_feature = feature.rsplit('_', 1)[0] if feature.rsplit('_', 1)[0] in original_columns else feature
        
        if base_feature not in grouped_importance:
            grouped_importance[base_feature] = 0
        grouped_importance[base_feature] += importance

    # Convertir les résultats en pourcentage
    total_importance = sum(grouped_importance.values())
    grouped_importance_percentage = {k: (v / total_importance) * 100 for k, v in grouped_importance.items()}

    # Trier par importance
    sorted_importance = sorted(grouped_importance_percentage.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Extraire les noms et importances triées
    features_sorted, importance_sorted = zip(*sorted_importance)

    # Visualiser les importances regroupées en pourcentage
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance_sorted, color='skyblue')
    plt.yticks(range(top_n), features_sorted)
    plt.xlabel('Percentage Importance (%)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()  # Inverser l'axe y pour afficher les plus importantes en haut
    plt.show()

    # Retourner un DataFrame avec l'importance en pourcentage des caractéristiques
    importance_df = pd.DataFrame({
        'Feature': features_sorted,
        'Importance (%)': importance_sorted
    })

    return importance_df

        
def visualize_coefficients(model, X, columns=None, top_n=20):
    """
    Visualise les coefficients d'un modèle linéaire avec les caractéristiques en axe Y et l'impact sur le prix en axe X.
    Affiche les top_n coefficients les plus importants pour les variables sélectionnées.
    :param model: Modèle linéaire entraîné
    :param X: DataFrame des variables d'entrée
    :param variables_selectionnees: Liste des noms de variables pour lesquelles afficher les coefficients
    (peut être None pour toutes les afficher)
    :param top_n: Nombre de coefficients les plus importants à visualiser
    """
    # Récupérer les coefficients du modèle
    coefficients = model.coef_

    # Créer un DataFrame pour les coefficients
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients
    })

    # Filtrer les coefficients par les variables sélectionnées si spécifié
    if columns:
        mask = coef_df['Feature'].apply(lambda x: any(var in x for var in columns))
        coef_df = coef_df[mask]

    # Trier les coefficients par ordre décroissant d'importance absolue
    top_coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index).head(top_n)

    # Visualiser les coefficients en barres horizontales
    top_coef_df.set_index('Feature')['Coefficient'].sort_values().plot(kind='barh', figsize=(12, 8))
    plt.title(f'Coefficients du modèle de régression')
    plt.xlabel('Impact sur le Prix')
    plt.ylabel('Feature')
    plt.grid(True)
    plt.show()

