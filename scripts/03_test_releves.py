
# ============================================================
# TECH60711 - TEST DU MODELE SUR LES RELEVES BANCAIRES
# ============================================================

# Importer les bibliothèques
import pandas as pd
import joblib
import re

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. CHARGER LE MODELE ET LES DONNEES DE REFERENCE
# ============================================================

# Trouver le dossier du projet
dossier = Path(__file__).resolve().parent

# Charger notre NOUVEAU modèle entraîné
modele = joblib.load(
    dossier / "modele_categories_nouveau.pkl"
)

# Charger les 90 transactions catégorisées manuellement
validation = pd.read_excel(
    dossier / "validation_categories_rempli.xlsx"
)

# Conserver les descriptions et les catégories confirmées
donnees_test = validation[
    ["Description", "Categorie_confirmee"]
].dropna().copy()


# ============================================================
# 2. NETTOYER LES DESCRIPTIONS COMME A L'ENTRAINEMENT
# ============================================================

def nettoyer_description(description):

    # Mettre le texte en majuscules
    texte = str(description).upper()

    # Supprimer les nombres isolés
    texte = re.sub(r"\b\d+\b", " ", texte)

    # Retirer les expressions bancaires génériques
    expressions = (
        r"ACHAT INTERAC|"
        r"INTERAC PURCHASE|"
        r"POS PURCHASE|"
        r"VISA DEBIT|"
        r"PRELEVEMENT AUTOMATIQUE|"
        r"PAIEMENT PREAUTORISE|"
        r"PRE-AUTH PMT|"
        r"ONLINE BILL PMT"
    )

    texte = re.sub(expressions, " ", texte)

    # Supprimer les espaces multiples
    return re.sub(r"\s+", " ", texte).strip()


# Préparer les descriptions à classifier
X_test = donnees_test["Description"].apply(
    nettoyer_description
)

# Récupérer les vraies catégories
y_test = donnees_test["Categorie_confirmee"]


# ============================================================
# 3. TESTER LE NOUVEAU MODELE
# ============================================================

# Faire prédire les catégories par l'IA
predictions = modele.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, predictions)

# Calculer le F1-score macro
f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

# Afficher les résultats
print("Nombre de transactions :", len(donnees_test))

print(
    "Accuracy :",
    round(accuracy * 100, 1),
    "%"
)

print(
    "F1-score macro :",
    round(f1 * 100, 1),
    "%"
)

print("\nPerformance par catégorie :")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 4. IDENTIFIER LES MAUVAISES PREDICTIONS
# ============================================================

# Ajouter les prédictions au tableau
donnees_test["Categorie_predite"] = predictions

# Vérifier si chaque prédiction est correcte
donnees_test["Correct"] = (
    donnees_test["Categorie_confirmee"]
    == donnees_test["Categorie_predite"]
)

# Conserver uniquement les erreurs
erreurs = donnees_test[
    ~donnees_test["Correct"]
]

print("\nNombre d'erreurs :", len(erreurs))

print("\nTransactions mal catégorisées :")
print(erreurs.to_string(index=False))

# Enregistrer toutes les prédictions pour les analyser
donnees_test.to_csv(
    dossier / "bilan_test_releves.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("\n✅ Résultats enregistrés dans bilan_test_releves.csv")
