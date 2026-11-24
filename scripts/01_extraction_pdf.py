
# ============================================================
# TECH60711 - EXTRACTION DES RELEVES BANCAIRES PDF
# ============================================================

from pathlib import Path
from collections import Counter
import re
import unicodedata
import pandas as pd
import pdfplumber

# Dossier contenant ce script et les fichiers PDF
DOSSIER = Path(__file__).resolve().parent

# Détecter les montants avec exactement deux décimales
# et éviter de confondre les numéros de magasins avec les montants
MONTANT_RE = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0]\d{3})+,\d{2}|"
    r"\d{1,3}(?:,\d{3})+\.\d{2}|\d+[,\.]\d{2})(?!\d)"
)

# Mois français et anglais
MOIS = {
    "jan": 1, "janv": 1, "january": 1,
    "feb": 2, "fev": 2, "fevr": 2, "february": 2,
    "mar": 3, "mars": 3, "march": 3,
    "avr": 4, "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "jun": 6, "juin": 6, "june": 6,
    "jul": 7, "juil": 7, "july": 7,
    "aou": 8, "aout": 8, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "decembre": 12, "december": 12,
}


def sans_accents(texte):
    # Retirer les accents pour reconnaître les mois
    texte = unicodedata.normalize("NFD", str(texte))

    return "".join(
        c for c in texte
        if unicodedata.category(c) != "Mn"
    ).lower()


def en_nombre(texte):
    # Convertir les formats 1 234,56 et 1,234.56
    # en nombres décimaux utilisables par Python
    valeur = (
        texte.replace(" ", "")
        .replace("\u00a0", "")
        .replace("$", "")
    )

    if valeur.rfind(",") > valeur.rfind("."):
        valeur = valeur.replace(".", "").replace(",", ".")
    else:
        valeur = valeur.replace(",", "")

    return round(float(valeur), 2)


def date_et_description(ligne, annee):
    # Format ISO : 2026-10-01 DESCRIPTION
    iso = re.match(
        r"^(\d{4}-\d{2}-\d{2})\s+(.+)$",
        ligne
    )

    if iso:
        try:
            return pd.Timestamp(iso.group(1)), iso.group(2)
        except ValueError:
            return None, None

    # Format français : 02 sept. DESCRIPTION
    francais = re.match(
        r"^(\d{1,2})\s+([A-Za-zÀ-ÿ.]+)\s+(.+)$",
        ligne
    )

    if francais:
        jour, mois_texte, reste = francais.groups()

        mois = MOIS.get(
            sans_accents(mois_texte).rstrip(".")
        )

        if mois:
            try:
                return pd.Timestamp(
                    annee, mois, int(jour)
                ), reste
            except ValueError:
                pass

    # Format anglais : Oct 01 DESCRIPTION
    anglais = re.match(
        r"^([A-Za-zÀ-ÿ.]+)\s+(\d{1,2})\s+(.+)$",
        ligne
    )

    if anglais:
        mois_texte, jour, reste = anglais.groups()

        mois = MOIS.get(
            sans_accents(mois_texte).rstrip(".")
        )

        if mois:
            try:
                return pd.Timestamp(
                    annee, mois, int(jour)
                ), reste
            except ValueError:
                pass

    return None, None


def solde_initial(lignes):
    # Chercher le solde d'ouverture du compte
    marqueurs = (
        "solde au",
        "solde d'ouverture",
        "solde initial",
        "opening balance"
    )

    for i, ligne in enumerate(lignes):
        texte = sans_accents(ligne).replace("’", "'")

        if not any(m in texte for m in marqueurs):
            continue

        montants = MONTANT_RE.findall(ligne)

        if montants:
            return en_nombre(montants[-1])

        # Certains relevés affichent le solde sur
        # la ligne suivant le titre OPENING BALANCE
        for suivante in lignes[i + 1:i + 3]:
            montants = MONTANT_RE.findall(suivante)

            if montants:
                return en_nombre(montants[-1])

    return None


def extraire_pdf(chemin):
    # Ouvrir le PDF et lire toutes ses pages
    with pdfplumber.open(chemin) as pdf:
        lignes = []

        for page in pdf.pages:
            texte = page.extract_text() or ""
            lignes.extend(texte.splitlines())

    # Retrouver l'année du relevé
    annees = re.findall(
        r"\b20\d{2}\b",
        "\n".join(lignes)
    )

    if not annees:
        print(
            f"⚠️ {chemin.name} : "
            "année introuvable, fichier ignoré."
        )
        return []

    annee = int(
        Counter(annees).most_common(1)[0][0]
    )

    # Initialiser le solde précédent
    precedent = solde_initial(lignes)

    operations = []

    # Parcourir les lignes du relevé
    for ligne in lignes:
        date, reste = date_et_description(
            ligne.strip(),
            annee
        )

        # Ignorer les lignes sans date reconnue
        if date is None:
            continue

        # Nos relevés fictifs utilisent :
        # DESCRIPTION + MONTANT + SOLDE
        montants = list(
            MONTANT_RE.finditer(reste)
        )

        if len(montants) < 2:
            continue

        # Extraire les deux derniers montants
        montant = en_nombre(
            montants[-2].group()
        )

        solde = en_nombre(
            montants[-1].group()
        )

        # Récupérer la description bancaire
        description = reste[
            :montants[-2].start()
        ].strip()

        if not description or montant <= 0:
            continue

        # Identifier le type selon la variation du solde
        if precedent is None:
            type_operation = "À vérifier"

        else:
            variation = round(
                solde - precedent, 2
            )

            # Vérifier la cohérence du montant
            if abs(abs(variation) - montant) > 0.02:
                type_operation = "À vérifier"

            elif variation > 0:
                type_operation = "Crédit"

            else:
                type_operation = "Débit"

        # Actualiser le solde précédent
        precedent = solde

        # Identifier les opérations ambiguës
        desc = sans_accents(description).upper()

        ambigu = (
            desc.startswith("RETRAIT GUICHET")
            or "ATM WITHDRAWAL" in desc
            or (
                "VISA CARD" in desc
                and "BILL PMT" in desc
            )
            or "PAIEMENT CARTE CREDIT" in desc
        )

        if type_operation == "Débit" and ambigu:
            type_operation = "À examiner"

        # Enregistrer la transaction
        operations.append({
            "Fichier": chemin.name,
            "Date": date.strftime("%Y-%m-%d"),
            "Description": description,
            "Montant": montant,
            "Solde": solde,
            "Type": type_operation
        })

    return operations


def main():
    # Regrouper les opérations de tous les PDF du dossier
    toutes = []

    for fichier in sorted(DOSSIER.glob("*.pdf")):
        try:
            operations = extraire_pdf(fichier)

            print(
                f"📄 {fichier.name} : "
                f"{len(operations)} opérations extraites"
            )

            toutes.extend(operations)

        except Exception as erreur:
            print(f"⚠️ {fichier.name} : {erreur}")

    if not toutes:
        print(
            "Aucune transaction détectée. "
            "Place les relevés PDF dans ce dossier."
        )
        return

    # Créer le tableau général
    tableau = pd.DataFrame(toutes)

    # Conserver uniquement les débits reconnus
    depenses = tableau[
        tableau["Type"] == "Débit"
    ].copy()

    # Enregistrer toutes les opérations
    tableau.to_csv(
        DOSSIER / "transactions_extraites.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    # Enregistrer les dépenses à catégoriser
    depenses.to_csv(
        DOSSIER / "depenses_a_categoriser.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    # Afficher un résumé
    print("\nOpérations par type :")
    print(tableau["Type"].value_counts())

    print(
        "\nTotal des dépenses reconnues :",
        round(depenses["Montant"].sum(), 2),
        "$"
    )

    print("\n✅ Fichiers CSV créés avec succès !")
    print("⚠️ Vérifier les montants et les descriptions extraites.")


# Exécuter le programme
if __name__ == "__main__":
    main()
