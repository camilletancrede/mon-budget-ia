
# ============================================================
# TECH60711 - ENTRAINEMENT DU MODELE DE CLASSIFICATION
# ============================================================

from pathlib import Path
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RACINE = Path(__file__).resolve().parent.parent

DOSSIER_DONNEES = RACINE / "donnees" / "entrainement"
DOSSIER_RESULTATS = RACINE / "resultats"

FICHIER_DEVELOPPEMENT = DOSSIER_DONNEES / "transactions_synthetiques_enrichies.csv"
FICHIER_TEST = DOSSIER_DONNEES / "transactions_synthetiques_test_reserve.csv"


# ============================================================
# 2. NETTOYAGE V2 DES DESCRIPTIONS
# ============================================================

def nettoyer_description(description):

    # Mettre les descriptions en majuscules
    texte = str(description).upper()

    # Supprimer les nombres isolés
    texte = re.sub(
        r"\b\d+\b",
        " ",
        texte
    )

    # Supprimer les expressions bancaires génériques
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

    texte = re.sub(
        expressions,
        " ",
        texte
    )

    # Uniformiser les espaces
    return re.sub(
        r"\s+",
        " ",
        texte
    ).strip()


# ============================================================
# 3. CHARGER ET PREPARER LES DONNEES
# ============================================================

def charger_csv(chemin):

    # Vérifier que le fichier existe
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier manquant : {chemin.name}"
        )

    # Importer le CSV
    tableau = pd.read_csv(
        chemin,
        sep=";",
        encoding="utf-8-sig"
    )

    # Vérifier les colonnes indispensables
    obligatoires = {
        "Description",
        "Categorie_confirmee"
    }

    if not obligatoires.issubset(tableau.columns):
        raise ValueError(
            f"Colonnes manquantes : "
            f"{obligatoires - set(tableau.columns)}"
        )

    # Retirer les descriptions ou catégories manquantes
    tableau = tableau.dropna(
        subset=["Description", "Categorie_confirmee"]
    ).copy()

    # Supprimer les espaces inutiles
    tableau["Description"] = (
        tableau["Description"].astype(str).str.strip()
    )

    tableau["Categorie_confirmee"] = (
        tableau["Categorie_confirmee"]
        .astype(str)
        .str.strip()
    )

    # Créer la description utilisée par le modèle
    tableau["Texte_modele"] = (
        tableau["Description"].apply(
            nettoyer_description
        )
    )

    # Retirer les descriptions devenues vides
    tableau = tableau[
        (tableau["Texte_modele"] != "")
        & (tableau["Categorie_confirmee"] != "")
    ]

    # Vérifier si une même description possède
    # plusieurs catégories contradictoires
    conflits = tableau.groupby(
        "Texte_modele"
    )["Categorie_confirmee"].nunique()

    contradictoires = set(
        conflits[conflits > 1].index
    )

    if contradictoires:

        print(
            f"⚠️ {len(contradictoires)} descriptions "
            "avec catégories contradictoires exclues."
        )

        tableau = tableau[
            ~tableau["Texte_modele"].isin(
                contradictoires
            )
        ]

    # Retirer les descriptions identiques
    # après le nettoyage V2
    tableau = tableau.drop_duplicates(
        subset="Texte_modele"
    )

    return tableau.reset_index(drop=True)


# ============================================================
# 4. CREER LES MODELES A COMPARER
# ============================================================

def creer_modele(variante, equilibre):

    # TF-IDF basé sur les mots
    if variante == "mots":

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode"
        )

    # TF-IDF basé sur les caractères
    else:

        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            strip_accents="unicode"
        )

    # Régression logistique
    regression = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if equilibre else None
    )

    # Assembler TF-IDF et régression logistique
    return make_pipeline(
        tfidf,
        regression
    )


# ============================================================
# 5. ENTRAINEMENT ET EVALUATION
# ============================================================

def main():

    # Importer les deux ensembles de données
    donnees = charger_csv(
        FICHIER_DEVELOPPEMENT
    )

    test = charger_csv(
        FICHIER_TEST
    )

    # Détecter les descriptions présentes
    # dans les deux ensembles après nettoyage
    chevauchements = (
        set(donnees["Texte_modele"])
        & set(test["Texte_modele"])
    )

    # Exclure les descriptions déjà présentes
    # dans les données de développement
    if chevauchements:

        test = test[
            ~test["Texte_modele"].isin(
                chevauchements
            )
        ].copy()

        print(
            f"⚠️ {len(chevauchements)} descriptions "
            "chevauchantes retirées du test."
        )

    if test.empty:
        raise ValueError(
            "Le test ne contient plus "
            "aucune description indépendante."
        )

    # Diviser le développement en :
    # 80 % entraînement et 20 % validation
    train, validation = train_test_split(
        donnees,
        test_size=0.20,
        random_state=42,
        stratify=donnees["Categorie_confirmee"]
    )

    print("Train :", len(train))
    print("Validation :", len(validation))
    print("Test réservé :", len(test))

    # Stocker les résultats des modèles
    resultats = []

    meilleur_score = (-1.0, -1.0)
    meilleur_choix = None

    # Comparer quatre configurations
    for variante in ("mots", "caracteres"):

        for equilibre in (False, True):

            # Créer le modèle
            modele = creer_modele(
                variante,
                equilibre
            )

            # Entraîner uniquement avec TRAIN
            modele.fit(
                train["Texte_modele"],
                train["Categorie_confirmee"]
            )

            # Prédire les catégories de VALIDATION
            predictions = modele.predict(
                validation["Texte_modele"]
            )

            # Calculer l'accuracy
            accuracy = accuracy_score(
                validation["Categorie_confirmee"],
                predictions
            )

            # Calculer le F1 macro
            f1_macro = f1_score(
                validation["Categorie_confirmee"],
                predictions,
                average="macro",
                zero_division=0
            )

            # Enregistrer les résultats
            resultats.append({
                "Variante": variante,
                "Classes_equilibrees": equilibre,
                "Accuracy_validation": round(accuracy, 4),
                "F1_macro_validation": round(f1_macro, 4)
            })

            # Choisir principalement selon le F1 macro
            # et ensuite selon l'accuracy
            if (f1_macro, accuracy) > meilleur_score:

                meilleur_score = (
                    f1_macro,
                    accuracy
                )

                meilleur_choix = (
                    variante,
                    equilibre
                )

    # ========================================================
    # 6. COMPARER LES RESULTATS
    # ========================================================

    classement = pd.DataFrame(
        resultats
    ).sort_values(
        [
            "F1_macro_validation",
            "Accuracy_validation"
        ],
        ascending=False
    )

    print("\nComparaison des modèles :")
    print(classement.to_string(index=False))

    # Enregistrer les résultats dans un CSV
    classement.to_csv(
        DOSSIER_RESULTATS / "comparaison_modeles.csv",
        sep=";", index=False, encoding="utf-8-sig"
    )

    # ========================================================
    # 7. REENTRAINER LE MEILLEUR MODELE
    # ========================================================

    # Récupérer la meilleure configuration
    variante, equilibre = meilleur_choix

    # Créer un nouveau modèle avec cette configuration
    modele_final = creer_modele(
        variante,
        equilibre
    )

    # Réentraîner sur TRAIN + VALIDATION
    # Le test réservé n'est jamais utilisé ici
    modele_final.fit(
        donnees["Texte_modele"],
        donnees["Categorie_confirmee"]
    )

    # ========================================================
    # 8. EVALUER SUR LE TEST RESERVE
    # ========================================================

    # Prédire les catégories des données de test
    predictions_test = modele_final.predict(
        test["Texte_modele"]
    )

    # Calculer les métriques finales
    accuracy_test = accuracy_score(
        test["Categorie_confirmee"],
        predictions_test
    )

    f1_test = f1_score(
        test["Categorie_confirmee"],
        predictions_test,
        average="macro",
        zero_division=0
    )

    print("\nMeilleur modèle :", variante)

    print(
        "Classes équilibrées :",
        equilibre
    )

    print(
        "Accuracy test :",
        round(accuracy_test * 100, 1),
        "%"
    )

    print(
        "F1 macro test :",
        round(f1_test * 100, 1),
        "%"
    )

    # Afficher les performances par catégorie
    print("\nRapport de classification :")

    print(
        classification_report(
            test["Categorie_confirmee"],
            predictions_test,
            zero_division=0
        )
    )

    # ========================================================
    # 9. ENREGISTRER LES PREDICTIONS
    # ========================================================

    # Créer un tableau pour analyser les erreurs
    bilan = test[
        ["Description", "Categorie_confirmee"]
    ].copy()

    bilan["Categorie_predite"] = predictions_test

    bilan["Correct"] = (
        bilan["Categorie_confirmee"]
        == bilan["Categorie_predite"]
    )

    # Exporter le tableau des prédictions
    bilan.to_csv(
        DOSSIER_RESULTATS / "predictions_test.csv",
        sep=";", index=False, encoding="utf-8-sig"
    )

    # ========================================================
    # 10. SAUVEGARDER LE NOUVEAU MODELE
    # ========================================================

    # Sauvegarder le modèle final à la racine du projet
    # Attention : remplace le fichier existant du même nom
    joblib.dump(
        modele_final,
        RACINE / "modele_categories_nouveau.pkl"
    )

    print("\n✅ Modèle sauvegardé !")
    print("Fichier : modele_categories_nouveau.pkl")

    print(
        "⚠️ Le test ne couvre pas toutes les catégories "
        "et certains commerçants peuvent être partagés "
        "entre entraînement et test."
    )


# Exécuter le programme
if __name__ == "__main__":
    main()
