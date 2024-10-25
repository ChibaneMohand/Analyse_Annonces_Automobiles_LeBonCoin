# Analyse Marché Automobile LeBonCoin

## Contexte

Dans le cadre de l'achat d'une voiture d'occasion avec un budget limité, ce projet vise à analyser les données des annonces de véhicules d'occasion publiées sur Le Bon Coin. Le but est d'étudier les tendances des prix, d'identifier les principaux facteurs influençant ces prix, et de développer un modèle de régression capable d'estimer le prix d'une voiture en fonction de ses caractéristiques spécifiques. Ce modèle, conçu pour être simple et interprétable, a pour objectif de fournir aux acheteurs des informations utiles pour les aider dans leurs décisions.

## Structure du Projet

Le projet se compose des éléments suivants :

- **Dossier `notebook`** : Contient un notebook Jupyter qui présente les résultats détaillés de l'analyse. Il s'agit de l'élément central du projet, où les diverses étapes de l'analyse et les visualisations des données sont documentées.
- **Dossier `data`** : Contient les données collectées via un script de scraping, en se concentrant sur les voitures dont le prix est inférieur ou égal à 20 000 euros et dont l'année de construction est postérieure à 2010. L'objectif est de disposer d'un échantillon pertinent pour l'analyse et la modélisation.
- **Dossier `src`** : Contient les fonctions utilisées dans le notebook. Ces fonctions facilitent le traitement des données, les analyses statistiques et la création des visualisations.

## Exécution du Projet

1. **Cloner le Repository** :
   ```bash
   git clone https://github.com/ChibaneMohand/Analyse_Annonces_Automobiles_LeBonCoin.git
   cd Analyse_Annonces_Automobiles_LeBonCoin

2. **Installer les Packages Nécessaires** :

Créez un environnement virtuel et activez-le.

Installez les packages requis:

```bash
 pip install numpy pandas scipy openpyxl matplotlib seaborn shap scikit-learn 
```
ou 
```bash
pip install -r requirements.txt
```

3. **Exécuter le Notebook** :

Ouvrez le notebook dans Jupyter et exécutez les cellules pour reproduire l'analyse.

## Références
Ce projet s'appuie sur de nombreuses analyses de prédiction de prix de voitures disponibles sur GitHub et tire parti de l’assistance de ChatGPT et Copilot pour la génération et l'optimisation du code.

