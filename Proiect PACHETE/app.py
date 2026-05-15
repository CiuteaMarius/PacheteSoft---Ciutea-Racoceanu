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

fisier_date = Path(__file__).parent / "Superstore.xlsx"


@st.cache_data
def citeste_date():
    tabel = pd.read_excel(fisier_date)

    tabel["Order Date"] = pd.to_datetime(tabel["Order Date"])
    tabel["Ship Date"] = pd.to_datetime(tabel["Ship Date"])
    tabel["Year"] = tabel["Order Date"].dt.year
    tabel["Month"] = tabel["Order Date"].dt.month
    tabel["Delivery_Days"] = (tabel["Ship Date"] - tabel["Order Date"]).dt.days

    return tabel


def completeaza_lipsa(tabel):
    tabel = tabel.copy()

    for coloana in tabel.columns:
        if tabel[coloana].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(tabel[coloana]):
            tabel[coloana] = tabel[coloana].fillna(tabel[coloana].median())
        else:
            tabel[coloana] = tabel[coloana].fillna(tabel[coloana].mode()[0])

    return tabel


def trateaza_extreme(tabel, coloane):
    tabel = tabel.copy()

    for coloana in coloane:
        prag25 = tabel[coloana].quantile(0.25)
        prag75 = tabel[coloana].quantile(0.75)
        interval_quart = prag75 - prag25

        limita_jos = prag25 - 1.5 * interval_quart
        limita_sus = prag75 + 1.5 * interval_quart

        tabel[coloana] = tabel[coloana].clip(limita_jos, limita_sus)

    return tabel


date_initiale = citeste_date()
date = completeaza_lipsa(date_initiale)


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
    def afiseaza_lipsa(tabel_initial, tabel_curat):
        lipsa = pd.DataFrame(
            {
                "coloana": tabel_initial.columns,
                "lipsa_initial": tabel_initial.isna().sum().values,
                "lipsa_dupa_tratare": tabel_curat.isna().sum().values,
            }
        )
        lipsa = lipsa[lipsa["lipsa_initial"] > 0]

        if lipsa.empty:
            st.info("Nu exista valori lipsa in setul de date.")
        else:
            st.dataframe(lipsa, use_container_width=True, hide_index=True)

    def spatiu(px):
        st.markdown(f"<div style='height: {px}px;'></div>", unsafe_allow_html=True)

    st.caption("Vedere generala: date, valori lipsa, statistici, grupari si grafice.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vanzari", f"${date['Sales'].sum():,.0f}")
    c2.metric("Profit", f"${date['Profit'].sum():,.0f}")
    c3.metric("Comenzi", f"{date['Order ID'].nunique():,}")
    c4.metric("Clienti", f"{date['Customer ID'].nunique():,}")

    st.subheader("Date")
    st.dataframe(date, use_container_width=True, height=500)

    st.subheader("Valori lipsa")
    afiseaza_lipsa(date_initiale, date)

    st.subheader("Statistici numerice")
    coloane_numerice = ["Sales", "Quantity", "Discount", "Profit", "Delivery_Days"]
    st.dataframe(date[coloane_numerice].describe().round(2), use_container_width=True)

    st.subheader("Vanzari lunare")
    lunar = (
        date.groupby(pd.Grouper(key="Order Date", freq="ME"))
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    grafic_lunar = lunar.set_index("Order Date")[["Sales", "Profit"]]
    st.line_chart(grafic_lunar)

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
    regiune_segment = pd.pivot_table(
        date,
        values="Sales",
        index="Region",
        columns="Segment",
        aggfunc="sum",
    ).round(2)
    st.dataframe(regiune_segment, use_container_width=True)
    st.caption("Valorile din celule reprezinta vanzarile totale pentru fiecare combinatie Region - Segment.")


# SEGMENTARE CLIENTI:
elif pagina == "Segmentare clienti":
    # metoda elbow
    def gaseste_k_elbow(valori_k, lista_wcss):
        primul_k, primul_wcss = valori_k[0], lista_wcss[0]
        ultimul_k, ultimul_wcss = valori_k[-1], lista_wcss[-1]

        distante = []
        for k_curent, wcss_curent in zip(valori_k, lista_wcss):
            distanta = abs(
                (ultimul_wcss - primul_wcss) * k_curent
                - (ultimul_k - primul_k) * wcss_curent
                + ultimul_k * primul_wcss
                - ultimul_wcss * primul_k
            )
            distanta = distanta / np.sqrt(
                (ultimul_wcss - primul_wcss) ** 2 + (ultimul_k - primul_k) ** 2
            )
            distante.append(distanta)

        return int(valori_k[np.argmax(distante)])

    st.caption("Clientii sunt grupati dupa vanzarile totale si cantitatea comandata.")

    clienti = (
        date.groupby("Customer ID")
        .agg(
            Sales=("Sales", "sum"),
            Quantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    coloane_kmeans = ["Sales", "Quantity"]
    clienti_model = trateaza_extreme(clienti, coloane_kmeans)

    scaler_clienti = StandardScaler()
    valori_clienti = scaler_clienti.fit_transform(clienti_model[coloane_kmeans])

    st.subheader("Alegerea numarului de clustere")

    valori_k = np.arange(1, 9)
    lista_wcss = []
    for k_test in valori_k:
        model_test = KMeans(n_clusters=k_test, random_state=42, n_init=10)
        model_test.fit(valori_clienti)
        lista_wcss.append(model_test.inertia_)

    k_optim = gaseste_k_elbow(valori_k, lista_wcss)
    k_optim = max(2, min(6, k_optim))
    st.info(f"Conform metodei Elbow, numarul optim de clustere este: {k_optim}")

    k_ales = st.slider("Numar clustere ales", min_value=2, max_value=6, value=k_optim)
    st.metric("K folosit in KMeans", k_ales)

    kmeans = KMeans(n_clusters=k_ales, random_state=42, n_init=10)
    clienti["Cluster"] = kmeans.fit_predict(valori_clienti)

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
    st.caption(f"Rezultatul de mai jos este calculat pentru K = {k_ales}.")

    rezultat_clustere = (
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
    st.dataframe(rezultat_clustere, use_container_width=True)


# PROFITABILITATE:
elif pagina == "Profitabilitate":
    st.caption("Model simplu pentru tranzactii profitabile/neprofitabile.")

    lucru = date.copy()
    lucru["Profitabil"] = (lucru["Profit"] > 0).astype(int)

    numerice = ["Sales", "Quantity", "Discount", "Delivery_Days"]
    categorice = ["Segment", "Category", "Ship Mode", "Region"]

    lucru = trateaza_extreme(lucru, numerice)

    x = lucru[numerice + categorice]
    x = pd.get_dummies(x, columns=categorice, drop_first=True, dtype=float)
    y = lucru["Profitabil"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler_logistic = StandardScaler()
    x_train_scalat = scaler_logistic.fit_transform(x_train)
    x_test_scalat = scaler_logistic.transform(x_test)

    model_logistic = LogisticRegression(max_iter=1000)
    model_logistic.fit(x_train_scalat, y_train)

    predictii = model_logistic.predict(x_test_scalat)
    probabilitati = model_logistic.predict_proba(x_test_scalat)[:, 1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Acuratete", f"{accuracy_score(y_test, predictii):.3f}")
    m2.metric("ROC AUC", f"{roc_auc_score(y_test, probabilitati):.3f}")
    m3.metric("Tranzactii test", f"{len(y_test):,}")

    col_stanga, col_dreapta = st.columns(2)

    with col_stanga:
        st.subheader("Matrice de confuzie")

        # matrice confuzie
        cm = confusion_matrix(y_test, predictii)
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

    with col_dreapta:
        st.subheader("Coeficienti importanti")

        coeficienti = pd.DataFrame(
            {
                "variabila": x.columns,
                "coeficient": model_logistic.coef_[0],
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
    factori_numerici = ["Sales", "Quantity", "Discount", "Delivery_Days"]
    regresie = trateaza_extreme(regresie, factori_numerici + ["Profit"])

    standardizare = st.toggle("Standardizeaza variabilele numerice")

    st.subheader("Date folosite")
    st.write("Variabila estimata: Profit")
    st.write("Variabile folosite: Sales, Quantity, Discount, Category, Order Date si Ship Date.")
    st.caption("Din Order Date si Ship Date se calculeaza numarul de zile de livrare.")

    x_reg = regresie[factori_numerici + ["Category"]]
    x_reg = pd.get_dummies(x_reg, columns=["Category"], drop_first=True, dtype=float)

    if standardizare:
        x_reg[factori_numerici] = StandardScaler().fit_transform(x_reg[factori_numerici])
        st.info("Variabilele numerice au fost standardizate cu StandardScaler.")

    x_reg = sm.add_constant(x_reg)
    y_reg = regresie["Profit"]

    model_OLS = sm.OLS(y_reg, x_reg).fit()
    profit_estimat = model_OLS.predict(x_reg)
    reziduuri = y_reg - profit_estimat

    r1, r2, r3 = st.columns(3)
    r1.metric("R patrat", f"{model_OLS.rsquared:.3f}")
    r2.metric("R patrat ajustat", f"{model_OLS.rsquared_adj:.3f}")
    r3.metric("Observatii", f"{len(regresie):,}")

    st.subheader("Coeficienti")
    st.caption("Coeficient pozitiv: asociat cu profit mai mare. Coeficient negativ: asociat cu profit mai mic.")

    tabel_coef = pd.DataFrame(
        {
            "variabila": model_OLS.params.index,
            "coeficient": model_OLS.params.values,
            "p_value": model_OLS.pvalues.values,
        }
    )
    st.dataframe(tabel_coef, use_container_width=True, hide_index=True)

    grafic1, grafic2 = st.columns(2)

    with grafic1:
        st.subheader("Profit real vs estimat")
        fig, ax = plt.subplots()
        ax.scatter(y_reg, profit_estimat, alpha=0.25)
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
        st.text(model_OLS.summary().as_text())
