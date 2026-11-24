# Mon Budget IA 💸

## Description du projet

Prototype développé dans le cadre du cours TECH60711 à HEC Montréal.

L'application permet d'importer des relevés bancaires fictifs au format PDF, d'en extraire les transactions, de catégoriser automatiquement les dépenses à l'aide d'un modèle d'apprentissage automatique et de consolider les dépenses d'un ménage.

## Technologies utilisées

- **Python** : développement et traitement des données
- **pdfplumber et Regex** : extraction des transactions
- **Pandas** : nettoyage et manipulation des données
- **Scikit-learn** : TF-IDF et régression logistique
- **Streamlit** : interface web
- **SQLite** : stockage des transactions validées

## Fonctionnement du projet

1. **Extraction (`01_extraction_pdf.py`)** : extraire les transactions des PDF.
2. **Entraînement (`02_entrainement_modele.py`)** : entraîner et évaluer différentes configurations du modèle.
3. **Tests (`03_test_releves.py`)** : mesurer les performances sur les transactions extraites de relevés fictifs.
4. **Application (`app.py`)** : utiliser le modèle entraîné pour catégoriser et afficher les dépenses.

## Résultats du modèle

La configuration sélectionnée utilise TF-IDF sur des séquences de caractères et une régression logistique.

- Accuracy sur le test synthétique : **98,9 %**
- F1-score macro sur le test synthétique : **97,8 %**
- Accuracy sur les dépenses des trois PDF fictifs : **77,8 %**

Ces résultats montrent les limites de généralisation du modèle et ne garantissent pas sa performance sur des relevés bancaires réels.

## Exécuter l'application localement

Installer les dépendances :

`pip install -r requirements.txt`

Lancer l'application :

`streamlit run app.py`

## Limites et confidentialité

Le prototype a été développé et évalué avec des données fictives. Il n'est pas destiné à traiter de véritables renseignements bancaires sans vérifications supplémentaires de sécurité et de fiabilité.
