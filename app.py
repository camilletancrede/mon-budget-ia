# ============================================================
# TECH60711 - MON BUDGET IA
# Application Streamlit - version 3
# ============================================================

from pathlib import Path
from io import BytesIO
from collections import Counter
import hashlib
import re
import sqlite3
import unicodedata

import altair as alt
import joblib
import pandas as pd
import pdfplumber
import streamlit as st


# ============================================================
# 1. CONFIGURATION ET COULEURS
# ============================================================

DOSSIER = Path(__file__).resolve().parent
DB = DOSSIER / "budget_local.db"
MODELE = DOSSIER / "modele_categories_nouveau.pkl"

# Palette douce, mais avec des couleurs suffisamment distinctes
COULEURS_CATEGORIES = {
    "Abonnements": "#cdb4db",       # Violet
    "Animaux": "#da627d",           # Framboise
    "Frais bancaires": "#31572c",  # Vert fonçé
    "Logement": "#fcbf49",         # Jaune
    "Loisirs": "#59656f",          # Gris
    "Magasinage": "#ffafcc",       # Rose
    "Restaurants": "#ff9b54",      # Orange
    "Santé et soins": "#a7c957",   # Vert
    "Transport": "#e3d5ca",        # Beige
    "Voyage": "#786452",           # Brun
    "Épicerie": "#a8dadc",         # Turquoise
}
CATEGORIES = list(COULEURS_CATEGORIES)

st.set_page_config(
    page_title="Mon budget IA",
    page_icon="🌷",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    --mauve-pale: #E4DFE9;
    background-color: #FCFBF9;
    color: #332D3A;
}
[data-testid="stSidebar"] {
    background-color: #F0EDF3;
    border-right: 1px solid var(--mauve-pale);
}
.block-container { max-width: 1180px; padding-top: 2.4rem; }
h1, h2, h3 {
    font-family: Georgia, 'Times New Roman', serif;
    color: #342D40;
}
h1 { font-size: 2.9rem !important; letter-spacing: -1px; }
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E9E4ED;
    padding: 15px 17px;
    border-radius: 15px;
}
[data-testid="stMetricValue"] { color: #66528C; }
.st-key-releves_graphiques [data-baseweb="tag"],
.stButton button:enabled,
.stFormSubmitButton button:enabled {
    background-color: var(--mauve-pale);
    color: #342D40;
}
.stButton button:enabled,
.stFormSubmitButton button:enabled {
    border-color: #B5A8C3;
}
.stButton button:enabled:hover,
.stFormSubmitButton button:enabled:hover {
    background-color: #D8CFE2;
    border-color: #705A94;
}
.brand {
    font-family: Georgia, serif;
    color: #342D40;
    font-size: 2.3rem;
    font-weight: bold;
    margin-bottom: 0;
}
.subbrand { color: #797181; margin-top: 0; margin-bottom: 1.6rem; }
</style>
""", unsafe_allow_html=True)


def titre(nom, description):
    st.title(nom)
    st.caption(description)
    st.write("")


def dollars(valeur):
    return f"{float(valeur):,.2f} $".replace(",", " ").replace(".", ",")


def palette():
    return alt.Scale(
        domain=CATEGORIES,
        range=list(COULEURS_CATEGORIES.values()),
    )


# ============================================================
# 2. BASE DE DONNÉES (compatible avec l'ancien budget_local.db)
# ============================================================

def connexion():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def initialiser_base():
    con = connexion()
    try:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS membres (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nom TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                membre_id INTEGER NOT NULL REFERENCES membres(id),
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                montant REAL NOT NULL,
                categorie TEXT NOT NULL,
                fichier TEXT NOT NULL,
                fichier_hash TEXT NOT NULL,
                numero_ligne INTEGER NOT NULL,
                UNIQUE(membre_id, fichier_hash, numero_ligne)
            );
        """)
        con.commit()
    finally:
        con.close()


def lire_membres():
    con = connexion()
    try:
        return pd.read_sql_query(
            "SELECT id, nom FROM membres ORDER BY nom", con
        )
    finally:
        con.close()


def lire_depenses():
    con = connexion()
    try:
        df = pd.read_sql_query("""
            SELECT t.id, t.date AS Date,
                   t.description AS Description,
                   t.montant AS Montant,
                   t.categorie AS Catégorie,
                   m.nom AS Membre,
                   t.membre_id AS Membre_id,
                   t.fichier AS Fichier,
                   t.fichier_hash AS Empreinte
            FROM transactions t
            JOIN membres m ON m.id = t.membre_id
            ORDER BY t.date DESC, t.id DESC
        """, con)
    finally:
        con.close()

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def lire_releves_enregistres():
    """Un relevé enregistré = un membre + une empreinte PDF."""
    con = connexion()
    try:
        return pd.read_sql_query("""
            SELECT t.membre_id, m.nom AS Membre,
                   t.fichier_hash AS Empreinte,
                   MIN(t.fichier) AS Fichier,
                   COUNT(*) AS Transactions,
                   SUM(t.montant) AS Total
            FROM transactions t
            JOIN membres m ON m.id = t.membre_id
            GROUP BY t.membre_id, t.fichier_hash, m.nom
            ORDER BY m.nom, MIN(t.fichier)
        """, con)
    finally:
        con.close()


# ============================================================
# 3. MODÈLE IA : même nettoyage que lors de l'entraînement
# ============================================================

@st.cache_resource
def charger_modele():
    if not MODELE.exists():
        raise FileNotFoundError(
            f"Le fichier {MODELE.name} est introuvable dans TECH60711."
        )
    # Uniquement le modèle entraîné dans notre projet
    return joblib.load(MODELE)


def nettoyer_description(description):
    texte = str(description).upper()
    texte = re.sub(r"\b\d+\b", " ", texte)
    expressions = (
        r"ACHAT INTERAC|INTERAC PURCHASE|POS PURCHASE|VISA DEBIT|"
        r"PRELEVEMENT AUTOMATIQUE|PAIEMENT PREAUTORISE|"
        r"PRE-AUTH PMT|ONLINE BILL PMT"
    )
    texte = re.sub(expressions, " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


# ============================================================
# 4. EXTRACTION DES PDF FICTIFS AVEC TEXTE SÉLECTIONNABLE
# ============================================================

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

MONTANT_RE = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0]\d{3})+,\d{2}|"
    r"\d{1,3}(?:,\d{3})+\.\d{2}|\d+[,.]\d{2})(?!\d)"
)


def sans_accents(texte):
    normal = unicodedata.normalize("NFD", str(texte))
    return "".join(
        caractere for caractere in normal
        if unicodedata.category(caractere) != "Mn"
    ).lower()


def nombre(texte):
    texte = texte.replace(" ", "").replace("\u00a0", "").replace("$", "")
    if texte.rfind(",") > texte.rfind("."):
        texte = texte.replace(".", "").replace(",", ".")
    else:
        texte = texte.replace(",", "")
    return round(float(texte), 2)


def lire_date(ligne, annee):
    # Ex. 2026-10-01 RESTAURANT...
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(.+)$", ligne)
    if m:
        try:
            return pd.Timestamp(m.group(1)).date(), m.group(2)
        except ValueError:
            pass

    # Ex. 02 sept. RESTAURANT...
    m = re.match(r"^(\d{1,2})\s+([A-Za-zÀ-ÿ.]+)\s+(.+)$", ligne)
    if m:
        jour, mois, reste = m.groups()
    else:
        # Ex. Oct 02 RESTAURANT...
        m = re.match(r"^([A-Za-zÀ-ÿ.]+)\s+(\d{1,2})\s+(.+)$", ligne)
        if not m:
            return None, None
        mois, jour, reste = m.groups()

    numero_mois = MOIS.get(sans_accents(mois).rstrip("."))
    if numero_mois:
        try:
            return pd.Timestamp(
                annee, numero_mois, int(jour)
            ).date(), reste
        except ValueError:
            pass
    return None, None


def solde_initial(lignes):
    for i, ligne in enumerate(lignes):
        texte = sans_accents(ligne).replace("’", "'")
        marqueurs = (
            "solde au", "solde d'ouverture", "solde initial",
            "opening balance",
        )
        if not any(m in texte for m in marqueurs):
            continue
        montants = MONTANT_RE.findall(ligne)
        if montants:
            return nombre(montants[-1])
        for suivante in lignes[i + 1:i + 3]:
            montants = MONTANT_RE.findall(suivante)
            if montants:
                return nombre(montants[-1])
    return None


def extraire_pdf(fichier):
    """Lire les opérations d'un PDF fictif; le PDF brut n'est pas stocké."""
    contenu = fichier.getvalue()
    empreinte = hashlib.sha256(contenu).hexdigest()
    with pdfplumber.open(BytesIO(contenu)) as pdf:
        lignes = []
        for page in pdf.pages:
            lignes.extend((page.extract_text() or "").splitlines())

    annees = re.findall(r"\b20\d{2}\b", "\n".join(lignes))
    if not annees:
        return [], "Année introuvable dans le relevé."

    annee = int(Counter(annees).most_common(1)[0][0])
    solde_precedent = solde_initial(lignes)
    transactions = []
    numero = 0

    for ligne in lignes:
        date, reste = lire_date(ligne.strip(), annee)
        if date is None:
            continue
        montants = list(MONTANT_RE.finditer(reste))
        if len(montants) < 2:
            continue

        montant = nombre(montants[-2].group())
        solde = nombre(montants[-1].group())
        description = reste[:montants[-2].start()].strip()
        if montant <= 0 or not description:
            continue

        if solde_precedent is None:
            typ = "À vérifier"
        else:
            variation = round(solde - solde_precedent, 2)
            if abs(abs(variation) - montant) > 0.02:
                typ = "À vérifier"
            elif variation > 0:
                typ = "Crédit"
            else:
                typ = "Débit"
        solde_precedent = solde

        desc = sans_accents(description).upper()
        ambigu = (
            desc.startswith("RETRAIT GUICHET")
            or "ATM WITHDRAWAL" in desc
            or ("VISA CARD" in desc and "BILL PMT" in desc)
            or "PAIEMENT CARTE CREDIT" in desc
        )
        if typ == "Débit" and ambigu:
            typ = "À examiner"

        numero += 1
        transactions.append({
            "Date": date,
            "Description": description,
            "Montant": montant,
            "Type": typ,
            "Fichier": fichier.name,
            "Empreinte": empreinte,
            "Numero": numero,
        })

    return transactions, None


# ============================================================
# 5. DONNÉES DE DÉMONSTRATION
# ============================================================

def creer_demo():
    exemples = [
        ("2026-09-03", "IGA EXTRA", 95.40, "Épicerie"),
        ("2026-09-06", "ESSO", 64.30, "Transport"),
        ("2026-09-09", "BELL INTERNET", 82.00, "Logement"),
        ("2026-09-12", "ST-HUBERT", 42.50, "Restaurants"),
        ("2026-09-16", "NETFLIX", 20.68, "Abonnements"),
        ("2026-09-22", "WINNERS", 77.30, "Magasinage"),
        ("2026-10-02", "METRO", 121.40, "Épicerie"),
        ("2026-10-05", "HYDRO-QUEBEC", 112.50, "Logement"),
        ("2026-10-09", "SHELL", 59.20, "Transport"),
        ("2026-10-12", "RESTAURANT LOCAL", 36.90, "Restaurants"),
        ("2026-10-17", "SEPHORA", 49.00, "Santé et soins"),
        ("2026-10-19", "CINEPLEX", 27.50, "Loisirs"),
        ("2026-10-22", "MONDOU", 43.50, "Animaux"),
        ("2026-10-27", "AIR CANADA", 229.00, "Voyage"),
    ]
    df = pd.DataFrame(
        exemples, columns=["Date", "Description", "Montant", "Catégorie"]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["Membre"] = "Démonstration"
    df["Fichier"] = "Exemple fictif"
    return df


# ============================================================
# 6. NAVIGATION
# ============================================================

initialiser_base()
membres = lire_membres()

with st.sidebar:
    st.markdown('<div class="brand">mon budget.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subbrand">Vos dépenses, plus simplement.</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Vue d’ensemble", "Importer des relevés", "Membres du ménage"],
    )
    st.divider()
    st.caption("Prototype universitaire · TECH60711")

if st.session_state.get("message"):
    st.success(st.session_state.pop("message"))


# ============================================================
# 7. VUE D'ENSEMBLE
# ============================================================

if page == "Vue d’ensemble":
    titre("Vue d’ensemble", "Explorez les dépenses de votre ménage.")
    df = lire_depenses()
    demonstration = False

    if df.empty:
        st.info("Aucune dépense enregistrée pour l'instant.")
        demonstration = st.toggle(
            "Explorer avec des données fictives", value=True
        )
        if demonstration:
            df = creer_demo()
            st.caption("🌷 Ces données sont fictives et ne sont pas enregistrées.")

    if not df.empty:
        # Filtre : membre
        personnes = sorted(df["Membre"].unique().tolist())
        personne = st.selectbox(
            "Dépenses de", ["Tout le ménage"] + personnes
        )
        if personne != "Tout le ménage":
            df = df.loc[df["Membre"] == personne].copy()

        # Filtre : mois
        df["Mois"] = df["Date"].dt.strftime("%Y-%m")
        mois = st.selectbox(
            "Période",
            ["Tous les mois"] + sorted(df["Mois"].unique(), reverse=True),
        )
        if mois != "Tous les mois":
            df = df.loc[df["Mois"] == mois].copy()

        # Filtre : relevés actuellement visualisés
        df["Source"] = df["Membre"] + " · " + df["Fichier"]
        sources = sorted(df["Source"].unique().tolist())
        sources_visibles = st.multiselect(
            "Relevés affichés dans les graphiques",
            options=sources,
            default=sources,
            key="releves_graphiques",
            help="Décochez un relevé pour le masquer temporairement des graphiques.",
        )
        df = df.loc[df["Source"].isin(sources_visibles)].copy()
        st.caption(
            f"📄 {len(sources_visibles)} relevé(s) actuellement pris en compte. "
            "Le filtre ne supprime aucune donnée."
        )

        if df.empty:
            st.info("Sélectionnez au moins un relevé pour afficher les résultats.")
        else:
            # Indicateurs principaux
            resume = (
                df.groupby("Catégorie", as_index=False)["Montant"]
                .sum().sort_values("Montant", ascending=False)
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Dépenses totales", dollars(df["Montant"].sum()))
            c2.metric("Transactions", len(df))
            c3.metric("Catégorie principale", resume.iloc[0]["Catégorie"])

            # Graphiques : une palette fixe par catégorie
            gauche, droite = st.columns([1, 1.15], gap="large")
            with gauche:
                st.subheader("Répartition des dépenses")
                anneau = (
                    alt.Chart(resume)
                    .mark_arc(innerRadius=75, stroke="#FFFFFF", strokeWidth=2)
                    .encode(
                        theta=alt.Theta("Montant:Q"),
                        color=alt.Color(
                            "Catégorie:N", scale=palette(),
                            legend=alt.Legend(title=None),
                        ),
                        tooltip=[
                            "Catégorie:N",
                            alt.Tooltip("Montant:Q", format=",.2f", title="Montant ($)"),
                        ],
                    )
                    .properties(height=345)
                )
                st.altair_chart(anneau, width="stretch")

            with droite:
                st.subheader("Dépenses par catégorie")
                barres = (
                    alt.Chart(resume)
                    .mark_bar(cornerRadiusEnd=5)
                    .encode(
                        x=alt.X("Montant:Q", title="Montant ($)"),
                        y=alt.Y("Catégorie:N", title=None, sort="-x"),
                        color=alt.Color(
                            "Catégorie:N", scale=palette(), legend=None
                        ),
                        tooltip=[
                            "Catégorie:N",
                            alt.Tooltip("Montant:Q", format=",.2f", title="Montant ($)"),
                        ],
                    )
                    .properties(height=345)
                )
                st.altair_chart(barres, width="stretch")

            mensuel = df.groupby("Mois", as_index=False)["Montant"].sum()
            if len(mensuel) > 1:
                st.subheader("Évolution mensuelle")
                courbe = (
                    alt.Chart(mensuel)
                    .mark_line(point=True, color="#806BA4", strokeWidth=3)
                    .encode(
                        x=alt.X("Mois:N", title="Mois", sort="ascending"),
                        y=alt.Y("Montant:Q", title="Dépenses ($)"),
                        tooltip=[
                            "Mois:N",
                            alt.Tooltip("Montant:Q", format=",.2f"),
                        ],
                    )
                    .properties(height=235)
                )
                st.altair_chart(courbe, width="stretch")

            with st.expander("📄 Quels relevés sont utilisés dans cette vue ?"):
                sources_df = (
                    df.groupby(["Membre", "Fichier"], as_index=False)
                    .agg(Transactions=("Montant", "size"), Total=("Montant", "sum"))
                )
                st.dataframe(sources_df, hide_index=True, width="stretch")

            st.subheader("Transactions récentes")
            st.dataframe(
                df.sort_values("Date", ascending=False)[
                    ["Date", "Description", "Catégorie", "Montant", "Membre", "Fichier"]
                ].head(20),
                hide_index=True, width="stretch",
            )

            # Corriger une catégorie même après l'importation
            if not demonstration:
                with st.expander("✏️ Corriger des catégories déjà enregistrées"):
                    st.write(
                        "Cliquez dans la colonne **Catégorie ✏️**, choisissez "
                        "la bonne catégorie, puis enregistrez vos corrections."
                    )
                    avant = df[
                        ["id", "Date", "Description", "Catégorie", "Montant", "Membre"]
                    ].reset_index(drop=True)
                    with st.form("corriger_categories_existantes"):
                        apres = st.data_editor(
                            avant,
                            hide_index=True, width="stretch", num_rows="fixed",
                            disabled=["id", "Date", "Description", "Montant", "Membre"],
                            column_config={
                                "id": None,
                                "Catégorie": st.column_config.SelectboxColumn(
                                    "Catégorie ✏️", options=CATEGORIES, required=True
                                ),
                                "Montant": st.column_config.NumberColumn(
                                    "Montant ($)", format="%.2f"
                                ),
                            },
                        )
                        enregistrer_corrections = st.form_submit_button(
                            "Enregistrer mes corrections", type="primary"
                        )
                    if enregistrer_corrections:
                        nb = 0
                        con = connexion()
                        try:
                            for i, ligne in apres.iterrows():
                                if (ligne["Catégorie"] != avant.iloc[i]["Catégorie"]
                                        and ligne["Catégorie"] in CATEGORIES):
                                    con.execute(
                                        "UPDATE transactions SET categorie=? WHERE id=?",
                                        (str(ligne["Catégorie"]), int(ligne["id"])),
                                    )
                                    nb += 1
                            con.commit()
                        finally:
                            con.close()
                        st.session_state["message"] = f"{nb} catégorie(s) corrigée(s)."
                        st.rerun()


# ============================================================
# 8. IMPORTATION ET GESTION DES RELEVÉS
# ============================================================

elif page == "Importer des relevés":
    titre(
        "Importer des relevés",
        "Importez plusieurs PDF, attribuez-les à leurs membres et validez l'IA.",
    )

    if membres.empty:
        st.warning("Ajoutez d'abord un membre dans « Membres du ménage ».")
        st.stop()

    choix_membres = {
        ligne.nom: int(ligne.id) for ligne in membres.itertuples()
    }
    fichiers = st.file_uploader(
        "1. Ajouter les relevés PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="fichiers_a_analyser",
        help="Vous pouvez choisir le relevé de plusieurs personnes en même temps.",
    )
    st.caption(
        "Les fichiers restent modifiables après l'analyse. "
        "Pour remplacer un PDF, retirez-le avec la croix (×) ci-dessus, "
        "puis choisissez le nouveau fichier."
    )

    if "fichiers_exclus" not in st.session_state:
        st.session_state["fichiers_exclus"] = set()

    attribution = []
    enregistres = lire_releves_enregistres()
    cles_enregistrees = {
        (int(r.membre_id), str(r.Empreinte))
        for r in enregistres.itertuples()
    }

    # La liste reste visible même après avoir cliqué sur « Analyser »
    if fichiers:
        st.subheader("2. Relevés sélectionnés")
        st.write(
            "Chaque fichier peut être associé à un membre différent. "
            "Utilisez **Retirer** pour l'exclure de la prochaine analyse, "
            "ou la croix (×) dans la zone d'import pour l'enlever définitivement de la sélection."
        )

        for i, fichier in enumerate(fichiers):
            empreinte = hashlib.sha256(fichier.getvalue()).hexdigest()
            cle = f"{i}_{empreinte[:14]}"
            exclu = empreinte in st.session_state["fichiers_exclus"]

            with st.container(border=True):
                col_fichier, col_membre, col_action = st.columns(
                    [2.5, 1.4, 1.1], vertical_alignment="center"
                )
                with col_fichier:
                    st.write(f"**📄 {fichier.name}**")
                    st.caption(f"{fichier.size / 1024:.1f} Ko")

                with col_membre:
                    membre = st.selectbox(
                        "Membre associé",
                        list(choix_membres),
                        key=f"association_{cle}",
                        disabled=exclu,
                    )

                with col_action:
                    if exclu:
                        if st.button("Réintégrer", key=f"activer_{cle}"):
                            st.session_state["fichiers_exclus"].discard(empreinte)
                            st.rerun()
                    else:
                        if st.button("Retirer", key=f"retirer_{cle}"):
                            st.session_state["fichiers_exclus"].add(empreinte)
                            st.rerun()

                if exclu:
                    st.caption("⏸️ Exclu de la prochaine analyse")
                else:
                    id_membre = choix_membres[membre]
                    deja = (id_membre, empreinte) in cles_enregistrees
                    st.caption(
                        "💾 Déjà enregistré pour cette personne"
                        if deja else "🌷 Prêt à analyser"
                    )
                    attribution.append(
                        (fichier, id_membre, membre, empreinte)
                    )

        st.caption(
            f"**{len(attribution)} relevé(s) actif(s)** sur "
            f"{len(fichiers)} sélectionné(s)."
        )

    # Une signature sert à éviter d'enregistrer un résultat devenu périmé
    signature = tuple(sorted(
        (empreinte, id_membre, fichier.name)
        for fichier, id_membre, _, empreinte in attribution
    ))

    if st.button(
        "Analyser les relevés avec l'IA",
        type="primary",
        disabled=not attribution,
    ):
        lignes_total = []
        avertissements = []
        bilan = []

        with st.spinner("Extraction et catégorisation en cours..."):
            for fichier, id_membre, membre, empreinte in attribution:
                try:
                    lignes, erreur = extraire_pdf(fichier)
                    if erreur or not lignes:
                        avertissements.append(
                            f"{fichier.name} : {erreur or 'aucune opération détectée'}"
                        )
                        continue
                    for ligne in lignes:
                        ligne["Membre_id"] = id_membre
                        ligne["Membre"] = membre
                    lignes_total.extend(lignes)
                    bilan.append({
                        "Membre": membre,
                        "Relevé": fichier.name,
                        "Opérations détectées": len(lignes),
                        "Dépenses": sum(x["Type"] == "Débit" for x in lignes),
                    })
                except Exception as erreur:
                    avertissements.append(
                        f"{fichier.name} : lecture impossible ({type(erreur).__name__})."
                    )

        depenses = pd.DataFrame(
            [r for r in lignes_total if r["Type"] == "Débit"]
        )
        if not depenses.empty:
            try:
                modele = charger_modele()
                textes = depenses["Description"].apply(nettoyer_description)
                depenses["Suggestion IA"] = modele.predict(textes.tolist())
                depenses["Ma catégorie"] = depenses["Suggestion IA"]
                depenses["Inclure"] = True
            except Exception as erreur:
                st.error(f"Impossible de charger le modèle IA : {erreur}")
                depenses = pd.DataFrame()

        st.session_state["analyse_signature"] = signature
        st.session_state["analyse_avertissements"] = avertissements
        st.session_state["analyse_bilan"] = pd.DataFrame(bilan)
        st.session_state["analyse_autres"] = pd.DataFrame(lignes_total)
        st.session_state["analyse_apercu"] = depenses.reset_index(drop=True)
        st.session_state["analyse_validee"] = False
        st.session_state["analyse_revision"] = (
            st.session_state.get("analyse_revision", 0) + 1
        )

    # Afficher l'analyse seulement si les fichiers et membres correspondent
    meme_selection = (
        st.session_state.get("analyse_signature") == signature
        and bool(attribution)
    )

    if attribution and not meme_selection and "analyse_signature" in st.session_state:
        st.info(
            "🔄 Vous avez modifié les relevés ou leur attribution. "
            "Cliquez sur **Analyser les relevés avec l'IA** pour actualiser "
            "le tableau. Les dépenses déjà sauvegardées ne sont pas supprimées."
        )

    if meme_selection:
        st.divider()
        st.subheader("3. Relevés de cette analyse")

        bilan = st.session_state.get("analyse_bilan", pd.DataFrame())
        if not bilan.empty:
            st.dataframe(bilan, hide_index=True, width="stretch")

        for avertissement in st.session_state.get("analyse_avertissements", []):
            st.warning(avertissement)

        apercu = st.session_state.get("analyse_apercu", pd.DataFrame())
        validee = st.session_state.get("analyse_validee", False)

        if apercu is not None and not apercu.empty:
            if validee:
                st.success(
                    "✅ Les dépenses de cette analyse ont été enregistrées. "
                    "Vos fichiers restent sélectionnés : vous pouvez en retirer, "
                    "en remplacer ou en ajouter, puis relancer l'analyse."
                )
                st.dataframe(
                    apercu[
                        ["Inclure", "Membre", "Date", "Description", "Montant", "Ma catégorie", "Fichier"]
                    ],
                    hide_index=True, width="stretch",
                )
                st.caption(
                    "Les lignes dont « Inclure » est décoché n'ont pas été enregistrées. "
                    "Pour corriger une catégorie déjà sauvegardée, ouvrez "
                    "« Vue d'ensemble » puis « Corriger des catégories »."
                )
            else:
                st.subheader("4. Corriger les catégories ✏️")
                st.write(
                    "La colonne **Suggestion IA** montre la prédiction originale. "
                    "Cliquez dans **Ma catégorie ✏️** pour la changer. "
                    "Décochez **Inclure** pour ne pas enregistrer une dépense."
                )
                st.caption(
                    "Les modifications ne seront enregistrées qu'après "
                    "avoir cliqué sur le bouton de validation."
                )

                colonnes = [
                    "Inclure", "Membre", "Date", "Description", "Montant",
                    "Suggestion IA", "Ma catégorie", "Fichier",
                ]

                with st.form(
                    f"validation_analyse_{st.session_state['analyse_revision']}"
                ):
                    corrections = st.data_editor(
                        apercu[colonnes].copy(),
                        hide_index=True,
                        width="stretch",
                        num_rows="fixed",
                        disabled=[
                            "Membre", "Date", "Description", "Montant",
                            "Suggestion IA", "Fichier",
                        ],
                        column_config={
                            "Inclure": st.column_config.CheckboxColumn(
                                "Inclure ✓",
                                help="Décochez pour exclure cette dépense.",
                            ),
                            "Date": st.column_config.DateColumn(
                                "Date", format="DD/MM/YYYY"
                            ),
                            "Montant": st.column_config.NumberColumn(
                                "Montant ($)", format="%.2f"
                            ),
                            "Ma catégorie": st.column_config.SelectboxColumn(
                                "Ma catégorie ✏️",
                                options=CATEGORIES,
                                required=True,
                            ),
                        },
                    )
                    confirmer = st.form_submit_button(
                        "✓ Valider et enregistrer les dépenses",
                        type="primary",
                    )

                if confirmer:
                    ajouts = doublons = exclusions = modifs = 0
                    con = connexion()
                    try:
                        for i, ligne in corrections.reset_index(drop=True).iterrows():
                            originale = apercu.iloc[i]
                            if not bool(ligne["Inclure"]):
                                exclusions += 1
                                continue
                            categorie = str(ligne["Ma catégorie"])
                            if categorie not in CATEGORIES:
                                exclusions += 1
                                continue
                            if categorie != originale["Suggestion IA"]:
                                modifs += 1
                            curseur = con.execute("""
                                INSERT OR IGNORE INTO transactions
                                (membre_id, date, description, montant,
                                 categorie, fichier, fichier_hash, numero_ligne)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                int(originale["Membre_id"]),
                                str(originale["Date"]),
                                str(originale["Description"]),
                                float(originale["Montant"]),
                                categorie,
                                str(originale["Fichier"]),
                                str(originale["Empreinte"]),
                                int(originale["Numero"]),
                            ))
                            ajouts += curseur.rowcount
                            if curseur.rowcount == 0:
                                doublons += 1
                        con.commit()
                    finally:
                        con.close()
                    st.session_state["analyse_apercu"]["Ma catégorie"] = (
                        corrections["Ma catégorie"].tolist()
                    )
                    st.session_state["analyse_apercu"]["Inclure"] = (
                        corrections["Inclure"].tolist()
                    )
                    st.session_state["analyse_validee"] = True
                    st.session_state["message"] = (
                        f"{ajouts} dépense(s) enregistrée(s), "
                        f"{modifs} catégorie(s) corrigée(s), "
                        f"{exclusions} exclue(s), {doublons} déjà enregistrée(s)."
                    )
                    st.rerun()

        else:
            st.warning("Aucune dépense reconnue dans les relevés sélectionnés.")

        autres = st.session_state.get("analyse_autres", pd.DataFrame())
        if not autres.empty:
            autres = autres.loc[autres["Type"] != "Débit"]
            if not autres.empty:
                with st.expander(f"Crédits et opérations à examiner ({len(autres)})"):
                    st.dataframe(
                        autres[
                            ["Membre", "Date", "Description", "Montant", "Type", "Fichier"]
                        ],
                        hide_index=True, width="stretch",
                    )
                    st.caption(
                        "Ces opérations sont exclues automatiquement des dépenses."
                    )

    # Liste persistante : ce qui est réellement dans le tableau de bord
    st.divider()
    st.subheader("📚 Relevés déjà enregistrés")
    st.caption(
        "Ces relevés alimentent actuellement la Vue d'ensemble. "
        "Ils restent enregistrés même si vous retirez un PDF de la sélection ci-dessus."
    )

    enregistres = lire_releves_enregistres()
    if enregistres.empty:
        st.info("Aucun relevé enregistré pour l'instant.")
    else:
        afficher = enregistres[
            ["Membre", "Fichier", "Transactions", "Total"]
        ].copy()
        afficher = afficher.rename(columns={"Total": "Dépenses ($)"})
        st.dataframe(afficher, hide_index=True, width="stretch")

        with st.expander("🗑️ Retirer ou remplacer un relevé enregistré"):
            st.write(
                "Sélectionnez un relevé à supprimer du tableau de bord. "
                "Pour le remplacer, supprimez d'abord ses dépenses enregistrées, "
                "puis importez et validez la nouvelle version du PDF."
            )
            options = {}
            for ligne in enregistres.itertuples():
                libelle = (
                    f"{ligne.Membre} · {ligne.Fichier} "
                    f"({ligne.Transactions} dépenses, {ligne.Empreinte[:6]})"
                )
                options[libelle] = (int(ligne.membre_id), str(ligne.Empreinte))

            choix = st.selectbox("Relevé à retirer", list(options))
            confirmer_suppression = st.checkbox(
                "Je confirme vouloir supprimer les dépenses "
                "de ce relevé du tableau de bord."
            )
            if st.button(
                "Supprimer ce relevé enregistré",
                disabled=not confirmer_suppression,
            ):
                id_membre, empreinte = options[choix]
                con = connexion()
                try:
                    curseur = con.execute(
                        "DELETE FROM transactions WHERE membre_id=? AND fichier_hash=?",
                        (id_membre, empreinte),
                    )
                    suppression = curseur.rowcount
                    con.commit()
                finally:
                    con.close()

                # Forcer une nouvelle analyse si ces fichiers sont resélectionnés
                st.session_state.pop("analyse_signature", None)
                st.session_state.pop("analyse_apercu", None)
                st.session_state["message"] = (
                    f"{suppression} dépense(s) supprimée(s) du tableau de bord. "
                    "Vous pouvez maintenant importer le relevé de remplacement."
                )
                st.rerun()


# ============================================================
# 9. MEMBRES DU MÉNAGE
# ============================================================

else:
    titre(
        "Membres du ménage",
        "Gérez les personnes dont vous souhaitez consolider les dépenses.",
    )

    st.subheader("Ajouter un membre")
    with st.form("ajouter_membre", clear_on_submit=True):
        nom = st.text_input(
            "Prénom ou nom du membre", placeholder="Ex. : Personne A"
        )
        ajouter = st.form_submit_button("Ajouter le membre", type="primary")

    if ajouter:
        nom = nom.strip()
        if not nom:
            st.warning("Entrez un nom.")
        elif nom.casefold() in {m.casefold() for m in membres["nom"]}:
            st.warning("Ce membre existe déjà.")
        else:
            con = connexion()
            try:
                con.execute("INSERT INTO membres (nom) VALUES (?)", (nom,))
                con.commit()
            finally:
                con.close()
            st.session_state["message"] = "Membre ajouté !"
            st.rerun()

    st.subheader("Membres enregistrés")
    if membres.empty:
        st.info("Aucun membre enregistré.")
    else:
        st.dataframe(
            membres[["nom"]].rename(columns={"nom": "Membre"}),
            hide_index=True, width="stretch",
        )
        with st.expander("Supprimer un membre"):
            nom_a_supprimer = st.selectbox(
                "Membre à supprimer", membres["nom"].tolist()
            )
            confirmation = st.checkbox(
                "Je confirme vouloir supprimer ce membre."
            )
            if st.button("Supprimer le membre", disabled=not confirmation):
                id_membre = int(membres.loc[
                    membres["nom"] == nom_a_supprimer, "id"
                ].iloc[0])
                con = connexion()
                try:
                    n = con.execute(
                        "SELECT COUNT(*) FROM transactions WHERE membre_id=?",
                        (id_membre,),
                    ).fetchone()[0]
                    if n:
                        st.error(
                            "Ce membre possède des transactions. "
                            "Supprimez d'abord ses relevés dans « Importer des relevés »."
                        )
                    else:
                        # Supprimer d'éventuels anciens budgets de la V1
                        try:
                            con.execute(
                                "DELETE FROM budgets WHERE proprietaire=?",
                                (f"membre:{id_membre}",),
                            )
                        except sqlite3.OperationalError:
                            pass
                        con.execute("DELETE FROM membres WHERE id=?", (id_membre,))
                        con.commit()
                        st.session_state["message"] = "Membre supprimé."
                        st.rerun()
                finally:
                    con.close()


# ============================================================
# 10. PIED DE PAGE
# ============================================================

st.divider()
st.caption(
    "Prototype universitaire TECH60711 · Extraction testée sur des PDF fictifs "
    "contenant du texte · Prédictions IA à valider · SQLite local non chiffré : "
    "ne pas utiliser pour stocker des données financières réelles."
)
