from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Analiza Superstore", layout="wide")

fisierDate = Path(__file__).parent / "Superstore.xlsx"


@st.cache_data
def citesteDate():
    tabel = pd.read_excel(fisierDate)

    tabel["Order Date"] = pd.to_datetime(tabel["Order Date"])
    tabel["Ship Date"] = pd.to_datetime(tabel["Ship Date"])
    tabel["Year"] = tabel["Order Date"].dt.year
    tabel["Month"] = tabel["Order Date"].dt.month
    tabel["Delivery_Days"] = (tabel["Ship Date"] - tabel["Order Date"]).dt.days

    return tabel


def completeazaLipsa(tabel):
    tabel = tabel.copy()

    for coloana in tabel.columns:
        if tabel[coloana].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(tabel[coloana]):
            tabel[coloana] = tabel[coloana].fillna(tabel[coloana].median())
        else:
            tabel[coloana] = tabel[coloana].fillna(tabel[coloana].mode()[0])

    return tabel


def trateazaExtreme(tabel, coloane):
    tabel = tabel.copy()

    for coloana in coloane:
        prag25 = tabel[coloana].quantile(0.25)
        prag75 = tabel[coloana].quantile(0.75)
        interval_quart = prag75 - prag25

        limitaJos = prag25 - 1.5 * interval_quart
        limitaSus = prag75 + 1.5 * interval_quart

        tabel[coloana] = tabel[coloana].clip(limitaJos, limitaSus)

    return tabel


def afiseazaLipsa(tabelInitial, tabelCurat):
    lipsa = pd.DataFrame(
        {
            "coloana": tabelInitial.columns,
            "lipsa_initial": tabelInitial.isna().sum().values,
            "lipsa_dupa_tratare": tabelCurat.isna().sum().values,
        }
    )
    lipsa = lipsa[lipsa["lipsa_initial"] > 0]

    if lipsa.empty:
        st.info("Nu exista valori lipsa in setul de date.")
    else:
        st.dataframe(lipsa, use_container_width=True, hide_index=True)


# metoda elbow
def gasesteKElbow(valoriK, listaWCSS):
    primulK, primulWCSS = valoriK[0], listaWCSS[0]
    ultimulK, ultimulWCSS = valoriK[-1], listaWCSS[-1]

    distante = []
    for kCurent, wcssCurent in zip(valoriK, listaWCSS):
        distanta = abs(
            (ultimulWCSS - primulWCSS) * kCurent
            - (ultimulK - primulK) * wcssCurent
            + ultimulK * primulWCSS
            - ultimulWCSS * primulK
        )
        distanta = distanta / np.sqrt(
            (ultimulWCSS - primulWCSS) ** 2 + (ultimulK - primulK) ** 2
        )
        distante.append(distanta)

    return int(valoriK[np.argmax(distante)])


def spatiu(px):
    st.markdown(f"<div style='height: {px}px;'></div>", unsafe_allow_html=True)


dateInitiale = citesteDate()
date = completeazaLipsa(dateInitiale)


with st.sidebar:
    st.title("Superstore")
    pagina = st.radio(
        "Meniu",
        [
            "Date si vanzari",
            "Segmentare clienti",
            "Profitabilitate",
            "Regresie multipla",
        ],
    )
    st.caption(f"{len(date):,} tranzactii | {date['Year'].min()}-{date['Year'].max()}")


st.title("Analiza activitatii Superstore")


# DATE SI VANZARI:
if pagina == "Date si vanzari":
    st.caption("Vedere generala: date, valori lipsa, statistici, grupari si grafice.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vanzari", f"${date['Sales'].sum():,.0f}")
    c2.metric("Profit", f"${date['Profit'].sum():,.0f}")
    c3.metric("Comenzi", f"{date['Order ID'].nunique():,}")
    c4.metric("Clienti", f"{date['Customer ID'].nunique():,}")

    st.subheader("Date")
    st.dataframe(date, use_container_width=True, height=500)

    st.subheader("Valori lipsa")
    afiseazaLipsa(dateInitiale, date)

    st.subheader("Statistici numerice")
    coloaneNumerice = ["Sales", "Quantity", "Discount", "Profit", "Delivery_Days"]
    st.dataframe(date[coloaneNumerice].describe().round(2), use_container_width=True)

    st.subheader("Vanzari lunare")
    lunar = (
        date.groupby(pd.Grouper(key="Order Date", freq="ME"))
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    graficLunar = lunar.set_index("Order Date")[["Sales", "Profit"]]
    st.line_chart(graficLunar)

    st.subheader("Grupari pandas")
    categorii = (
        date.groupby("Category")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
            Avg_Discount=("Discount", "mean"),
        )
        .sort_values("Sales", ascending=False)
        .round(2)
    )
    st.dataframe(categorii, use_container_width=True)
    st.caption("Tabelul arata vanzari totale, profit total, numar de comenzi si discount mediu pentru fiecare categorie.")

    spatiu(80)
    st.bar_chart(categorii[["Sales", "Profit"]])

    spatiu(100)
    st.subheader("Functii de grup")
    regiuneSegment = pd.pivot_table(
        date,
        values="Sales",
        index="Region",
        columns="Segment",
        aggfunc="sum",
    ).round(2)
    st.dataframe(regiuneSegment, use_container_width=True)
    st.caption("Valorile din celule reprezinta vanzarile totale pentru fiecare combinatie Region - Segment.")


# SEGMENTARE CLIENTI:
elif pagina == "Segmentare clienti":
    st.caption("Clientii sunt grupati dupa vanzarile totale si cantitatea comandata.")

    clienti = (
        date.groupby("Customer ID")
        .agg(
            Sales=("Sales", "sum"),
            Quantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    coloaneKMeans = ["Sales", "Quantity"]
    clientiModel = trateazaExtreme(clienti, coloaneKMeans)

    scalerClienti = StandardScaler()
    valoriClienti = scalerClienti.fit_transform(clientiModel[coloaneKMeans])

    st.subheader("Alegerea numarului de clustere")

    valoriK = np.arange(1, 9)
    listaWCSS = []
    for kTest in valoriK:
        modelTest = KMeans(n_clusters=kTest, random_state=42, n_init=10)
        modelTest.fit(valoriClienti)
        listaWCSS.append(modelTest.inertia_)

    kOptim = gasesteKElbow(valoriK, listaWCSS)
    kOptim = max(2, min(6, kOptim))
    st.info(f"Conform metodei Elbow, numarul optim de clustere este: {kOptim}")

    kAles = st.slider("Numar clustere ales", min_value=2, max_value=6, value=kOptim)
    st.metric("K folosit in KMeans", kAles)

    kmeans = KMeans(n_clusters=kAles, random_state=42, n_init=10)
    clienti["Cluster"] = kmeans.fit_predict(valoriClienti)

    stanga, dreapta = st.columns([2, 1])

    with stanga:
        fig, ax = plt.subplots()
        puncte = ax.scatter(
            clienti["Sales"],
            clienti["Quantity"],
            c=clienti["Cluster"],
            cmap="tab10",
            alpha=0.7,
        )
        ax.set_xlabel("Sales")
        ax.set_ylabel("Quantity")
        ax.legend(*puncte.legend_elements(), title="Cluster")
        st.pyplot(fig)

    with dreapta:
        st.subheader("Clustere")
        st.write("Culorile diferite reprezinta grupuri diferite de clienti.")
        st.write("Variabile folosite in KMeans:")
        st.write("- Sales: vanzari totale ($)")
        st.write("- Quantity: cantitate totala comandata")

    st.subheader("Rezultat pe clustere")
    st.caption(f"Rezultatul de mai jos este calculat pentru K = {kAles}.")

    rezultatClustere = (
        clienti.groupby("Cluster")
        .agg(
            Nr_Clienti=("Customer ID", "count"),
            Sales_mean=("Sales", "mean"),
            Sales_sum=("Sales", "sum"),
            Quantity_mean=("Quantity", "mean"),
            Quantity_sum=("Quantity", "sum"),
        )
        .round(2)
    )
    st.dataframe(rezultatClustere, use_container_width=True)


# PROFITABILITATE:
elif pagina == "Profitabilitate":
    st.caption("Model simplu pentru tranzactii profitabile/neprofitabile.")

    lucru = date.copy()
    lucru["Profitabil"] = (lucru["Profit"] > 0).astype(int)

    numerice = ["Sales", "Quantity", "Discount", "Delivery_Days"]
    categorice = ["Segment", "Category", "Ship Mode", "Region"]

    lucru = trateazaExtreme(lucru, numerice)

    x = lucru[numerice + categorice]
    x = pd.get_dummies(x, columns=categorice, drop_first=True, dtype=float)
    y = lucru["Profitabil"]

    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    scalerLogistic = StandardScaler()
    xTrainScalat = scalerLogistic.fit_transform(xTrain)
    xTestScalat = scalerLogistic.transform(xTest)

    modelLogistic = LogisticRegression(max_iter=1000)
    modelLogistic.fit(xTrainScalat, yTrain)

    predictii = modelLogistic.predict(xTestScalat)
    probabilitati = modelLogistic.predict_proba(xTestScalat)[:, 1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Acuratete", f"{accuracy_score(yTest, predictii):.3f}")
    m2.metric("ROC AUC", f"{roc_auc_score(yTest, probabilitati):.3f}")
    m3.metric("Tranzactii test", f"{len(yTest):,}")

    colStanga, colDreapta = st.columns(2)

    with colStanga:
        st.subheader("Matrice de confuzie")

        # matrice confuzie
        cm = confusion_matrix(yTest, predictii)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pierdere", "Profit"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Pierdere", "Profit"])
        ax.set_xlabel("Prezis")
        ax.set_ylabel("Real")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

    with colDreapta:
        st.subheader("Coeficienti importanti")

        coeficienti = pd.DataFrame(
            {
                "variabila": x.columns,
                "coeficient": modelLogistic.coef_[0],
            }
        )
        coeficienti["important"] = coeficienti["coeficient"].abs()
        coeficienti = coeficienti.sort_values("important", ascending=False)
        coeficienti = coeficienti.drop(columns="important").head(12)

        st.dataframe(coeficienti.round(4), use_container_width=True, hide_index=True)

    st.subheader("Date folosite in model")
    st.write("Codificare: get_dummies pentru variabilele categorice.")
    st.write("Scalare: StandardScaler inainte de regresia logistica.")


# REGRESIE MULTIPLA:
elif pagina == "Regresie multipla":
    st.caption("Model statsmodels pentru estimarea profitului.")

    regresie = date.copy()
    factoriNumerici = ["Sales", "Quantity", "Discount", "Delivery_Days"]
    regresie = trateazaExtreme(regresie, factoriNumerici + ["Profit"])

    standardizare = st.toggle("Standardizeaza variabilele numerice")

    st.subheader("Date folosite")
    st.write("Variabila estimata: Profit")
    st.write("Variabile folosite: Sales, Quantity, Discount, Category, Order Date si Ship Date.")
    st.caption("Din Order Date si Ship Date se calculeaza numarul de zile de livrare.")

    xReg = regresie[factoriNumerici + ["Category"]]
    xReg = pd.get_dummies(xReg, columns=["Category"], drop_first=True, dtype=float)

    if standardizare:
        xReg[factoriNumerici] = StandardScaler().fit_transform(xReg[factoriNumerici])
        st.info("Variabilele numerice au fost standardizate cu StandardScaler.")

    xReg = sm.add_constant(xReg)
    yReg = regresie["Profit"]

    modelOLS = sm.OLS(yReg, xReg).fit()
    profitEstimat = modelOLS.predict(xReg)
    reziduuri = yReg - profitEstimat

    r1, r2, r3 = st.columns(3)
    r1.metric("R patrat", f"{modelOLS.rsquared:.3f}")
    r2.metric("R patrat ajustat", f"{modelOLS.rsquared_adj:.3f}")
    r3.metric("Observatii", f"{len(regresie):,}")

    st.subheader("Coeficienti")
    st.caption("Coeficient pozitiv: asociat cu profit mai mare. Coeficient negativ: asociat cu profit mai mic.")

    tabelCoef = pd.DataFrame(
        {
            "variabila": modelOLS.params.index,
            "coeficient": modelOLS.params.values,
            "p_value": modelOLS.pvalues.values,
        }
    )
    st.dataframe(tabelCoef, use_container_width=True, hide_index=True)

    grafic1, grafic2 = st.columns(2)

    with grafic1:
        st.subheader("Profit real vs estimat")
        fig, ax = plt.subplots()
        ax.scatter(yReg, profitEstimat, alpha=0.25)
        ax.set_xlabel("Profit real")
        ax.set_ylabel("Profit estimat")
        st.pyplot(fig)

    with grafic2:
        st.subheader("Reziduuri")
        fig, ax = plt.subplots()
        ax.hist(reziduuri, bins=40)
        ax.set_xlabel("Reziduu")
        ax.set_ylabel("Numar tranzactii")
        st.pyplot(fig)

    with st.expander("Raport statsmodels"):
        st.text(modelOLS.summary().as_text())
