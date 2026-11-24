
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

# Trouver le dossier principal du projet
RACINE = Path(__file__).resolve().parent.parent

# Définir les dossiers utilisés
DOSSIER_VALIDATION = RACINE / "donnees" / "validation"
DOSSIER_RESULTATS = RACINE / "resultats"

# Charger notre modèle entraîné (aucun réentraînement)
modele = joblib.load(
    RACINE / "modele_categories_nouveau.pkl"
)

# Charger les transactions catégorisées manuellement
validation = pd.read_excel(
    DOSSIER_VALIDATION / "validation_categories_rempli.xlsx"
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
# 3. TESTER LE MODELE
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


# ============================================================
# 5. ANALYSER LA CONFIANCE SELON LES COULEURS DU PROTOTYPE
# ============================================================

# Même seuils que dans app.py : vert >= 75 %, orange >= 50 %, rouge < 50 %.
SEUIL_CONFIANCE_ELEVEE = 0.75
SEUIL_CONFIANCE_MOYENNE = 0.50

# Pour chaque transaction, conserver la probabilité maximale parmi
# les 11 catégories. Il s'agit du score du modèle, pas de l'accuracy.
probabilites = modele.predict_proba(X_test)
donnees_test["Score_confiance"] = probabilites.max(axis=1)

# Attribuer une couleur à chaque transaction selon son score.
# right=False : 0,50 est orange et 0,75 est vert.
donnees_test["Couleur"] = pd.cut(
    donnees_test["Score_confiance"],
    bins=[float("-inf"), SEUIL_CONFIANCE_MOYENNE,
          SEUIL_CONFIANCE_ELEVEE, float("inf")],
    labels=["Rouge", "Orange", "Vert"],
    right=False
)

# Résumer la performance observée pour chaque couleur.
analyse = (
    donnees_test.groupby("Couleur", observed=False)
    .agg(
        Transactions=("Correct", "size"),
        Correctes=("Correct", "sum"),
        Score_moyen=("Score_confiance", "mean")
    )
    .reindex(["Vert", "Orange", "Rouge"])
    .reset_index()
)

analyse["Erreurs"] = analyse["Transactions"] - analyse["Correctes"]

# Calculer le taux d'erreur uniquement lorsque le groupe contient des lignes.
analyse["Taux_erreur_%"] = (
    100 * analyse["Erreurs"] / analyse["Transactions"].replace(0, float("nan"))
).round(1)

analyse["Score_moyen_%"] = (100 * analyse["Score_moyen"]).round(1)
analyse = analyse.drop(columns="Score_moyen")

print("\n=== CONFIANCE ET ERREURS PAR COULEUR ===")
print(analyse.to_string(index=False, na_rep="N/A"))

# Parmi toutes les erreurs, quelle proportion est signalée en rouge ?
nombre_erreurs = int((~donnees_test["Correct"]).sum())
erreurs_rouges = int((
    (~donnees_test["Correct"]) & (donnees_test["Couleur"] == "Rouge")
).sum())

print("\n=== ERREURS REPEREES PAR LE ROUGE ===")
if nombre_erreurs > 0:
    print(
        f"{erreurs_rouges} erreur(s) sur {nombre_erreurs} sont rouges, "
        f"soit {erreurs_rouges / nombre_erreurs * 100:.1f} % des erreurs."
    )
else:
    print("Aucune erreur observée sur cet ensemble de test.")

print(
    "Attention : les probabilités du modèle ne sont pas nécessairement "
    "calibrées. Cette analyse est exploratoire (90 relevés fictifs attendus)."
)


# ============================================================
# 6. ENREGISTRER LES RESULTATS
# ============================================================

# S'assurer que le dossier des résultats existe
DOSSIER_RESULTATS.mkdir(
    parents=True,
    exist_ok=True
)

# Enregistrer les prédictions, scores et couleurs transaction par transaction.
donnees_test.to_csv(
    DOSSIER_RESULTATS / "bilan_test_releves.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

# Enregistrer le résumé des performances par couleur.
analyse.to_csv(
    DOSSIER_RESULTATS / "analyse_confiance_releves.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("\n✅ Résultats enregistrés dans resultats/bilan_test_releves.csv")
print("✅ Résumé enregistré dans resultats/analyse_confiance_releves.csv")
