# =============================================================================
# Proiect Pachete Software — Python (Streamlit)
# Dataset: Superstore Sales
# Cerinte acoperite: Streamlit, geopandas, valori lipsa, valori extreme,
#   codificare, scalare, prelucrari statistice pandas, functii de grup,
#   scikit-learn (KMeans, regresie logistica), statsmodels (regresie multipla)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="Superstore Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-right: 2px solid #e94560;
}
section[data-testid="stSidebar"] * { color: #eee !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px; padding: 4px 0; cursor: pointer; transition: all 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #e94560 !important; }

.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

.section-header {
    padding: 20px 24px 14px;
    border-radius: 12px;
    margin-bottom: 18px;
    
}
.section-header h1 {
    margin: 0 0 4px; font-size: 1.6rem; font-weight: 700;
    color: #fff; letter-spacing: -0.3px;
}
.section-header p { margin: 0; font-size: 13px; opacity: 0.75; color: #fff; }

.info-box {
    background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
}
.info-box h3 { margin: 0 0 6px; font-size: 14px; font-weight: 600; }
.info-box p { font-size: 13px; color: #495057; line-height: 1.6; margin: 0; }

.formula-box {
    background: #1a1a2e; color: #53d8fb; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; padding: 10px 14px; border-radius: 8px;
    margin: 8px 0; border-left: 3px solid #e94560; line-height: 1.7;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 16px; text-align: center; color: white;
}
.metric-card h4 { margin: 0; font-size: 12px; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
.metric-card p { margin: 4px 0 0; font-size: 24px; font-weight: 700; }

.interpret-box {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 10px;
    padding: 12px 15px; margin-top: 10px;
}
.interpret-box h4 { color: #856404; margin: 0 0 5px; font-size: 13px; font-weight: 600; }
.interpret-box p { color: #856404; font-size: 12.5px; margin: 0; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INCARCARE DATE
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_excel("Superstore.xlsx")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Delivery_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Profit_Margin'] = df['Profit'] / df['Sales']
    df['Profit_Margin'] = df['Profit_Margin'].replace([np.inf, -np.inf], np.nan)
    return df

df_original = load_data()


def section_header(title, subtitle, bg_color="#1a1a2e"):
    st.markdown(f"""<div class="section-header" style="background:{bg_color};">
    <h1>{title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)


def interpretation_box(title, text):
    st.markdown(f"""<div class="interpret-box">
    <h4>{title}</h4><p>{text}</p></div>""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR — NAVIGARE
# =============================================================================
SECTIONS = [
    "1. Prezentare Date",
    "2. Valori Lipsă",
    "3. Valori Extreme (Outlieri)",
    "4. Codificare Date",
    "5. Scalare Date",
    "6. Prelucrări Statistice & Grupări",
    "7. Vizualizare Geospațială",
    "8. Clustering K-Means",
    "9. Regresie Logistică",
    "10. Regresie Multiplă (statsmodels)",
]

with st.sidebar:
    st.markdown("## Superstore Analytics")
    st.markdown("*Analiza unui magazin de tip retail*")
    st.markdown("---")
    section = st.radio("Navigare:", SECTIONS, label_visibility="collapsed")
    idx = SECTIONS.index(section)
    st.markdown("---")


    st.markdown("---")
    st.markdown("**Dataset:** Superstore Sales")
    st.markdown(f"**Înregistrări:** {len(df_original):,}")
    st.markdown(f"**Coloane:** {df_original.shape[1]}")
    st.markdown(f"**Perioada:** 2011 – 2014")


# =============================================================================
# 1. PREZENTARE DATE
# =============================================================================
if section == SECTIONS[0]:
    section_header("Prezentarea Setului de Date",
                   "Analiza exploratorie a setului Superstore Sales — 9.994 tranzacții din retailul american", "2d6a4f")

    st.subheader("Primele înregistrări")
    st.dataframe(df_original.head(10), use_container_width=True, height=300)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tipuri de date")
        dtypes_df = pd.DataFrame({
            'Coloană': df_original.dtypes.index,
            'Tip': df_original.dtypes.values.astype(str),
            'Non-Null': df_original.notnull().sum().values,
            'Null': df_original.isnull().sum().values
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Statistici descriptive (variabile numerice)")
        st.dataframe(df_original.describe().T.round(2), use_container_width=True)

    st.subheader("Distribuția variabilelor categorice")
    cat_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    cols = st.columns(len(cat_cols))
    for i, col_name in enumerate(cat_cols):
        with cols[i]:
            st.markdown(f"**{col_name}**")
            vc = df_original[col_name].value_counts()
            st.dataframe(vc, use_container_width=True)

    st.subheader("Distribuția vânzărilor în timp")
    monthly = df_original.groupby(df_original['Order Date'].dt.to_period('M')).agg(
        Sales=('Sales', 'sum'), Profit=('Profit', 'sum')
    ).reset_index()
    monthly['Order Date'] = monthly['Order Date'].astype(str)
    fig = px.line(monthly, x='Order Date', y=['Sales', 'Profit'],
                  labels={'value': 'Valoare ($)', 'variable': 'Metric'},
                  title='Evoluția lunară a vânzărilor și profitului')
    fig.update_layout(height=350, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    interpretation_box("Interpretare economică",
        "Setul de date conține 9.994 de tranzacții din 4 ani (2011-2014) ale unui retailer american. "
        "Vânzările prezintă un trend ascendent cu sezonalitate puternică spre finalul fiecărui an "
        "(Q4 — sezonul cadourilor). Profitul urmează același pattern, dar cu variabilitate mai mare, "
        "indicând că anumite perioade generează marje negative (discounturi agresive).")


# =============================================================================
# 2. VALORI LIPSĂ
# =============================================================================
elif section == SECTIONS[1]:
    section_header("Tratarea Valorilor Lipsă",
                   "Identificarea și gestionarea valorilor lipsă din setul de date", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    Valorile lipsă (missing values) pot afecta calitatea analizelor statistice și performanța
    modelelor de machine learning. Este esențial să identificăm dacă există valori lipsă,
    în ce proporție, și să decidem strategia de tratare: **imputare** (înlocuire) sau **eliminare**.
    """)

    st.subheader("b) Verificarea valorilor lipsă în setul original")
    missing = df_original.isnull().sum()
    missing_pct = (df_original.isnull().sum() / len(df_original) * 100).round(2)
    missing_df = pd.DataFrame({'Total Lipsă': missing, 'Procent (%)': missing_pct}).sort_values(
        'Total Lipsă', ascending=False)
    st.dataframe(missing_df, use_container_width=True)

    st.success("Setul de date Superstore **nu conține valori lipsă** în coloanele originale.")

    # --- Simulare valori lipsă ---
    st.subheader("c) Simularea și tratarea valorilor lipsă (demonstrație)")
    st.markdown("""
    Pentru a demonstra tehnicile de tratare, vom introduce **artificial** valori lipsă
    în coloanele `Sales`, `Profit` și `Postal Code`, apoi le vom trata.
    """)

    df_missing = df_original.copy()
    np.random.seed(42)
    # Introducem 5% valori lipsă
    n_missing = int(len(df_missing) * 0.05)
    idx_sales = np.random.choice(df_missing.index, n_missing, replace=False)
    idx_profit = np.random.choice(df_missing.index, n_missing, replace=False)
    idx_postal = np.random.choice(df_missing.index, n_missing // 2, replace=False)

    df_missing.loc[idx_sales, 'Sales'] = np.nan
    df_missing.loc[idx_profit, 'Profit'] = np.nan
    df_missing.loc[idx_postal, 'Postal Code'] = np.nan

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Valori lipsă după simulare:**")
        missing_sim = df_missing.isnull().sum()
        missing_sim = missing_sim[missing_sim > 0]
        missing_sim_pct = (missing_sim / len(df_missing) * 100).round(2)
        st.dataframe(pd.DataFrame({
            'Total Lipsă': missing_sim,
            'Procent (%)': missing_sim_pct
        }), use_container_width=True)

    with col2:
        st.markdown("**Vizualizare valori lipsă:**")
        fig, ax = plt.subplots(figsize=(6, 3))
        cols_with_missing = df_missing.columns[df_missing.isnull().any()]
        sns.heatmap(df_missing[cols_with_missing].isnull().T, cbar=True, cmap='YlOrRd',
                    yticklabels=True, ax=ax)
        ax.set_title('Heatmap valori lipsă (galben = lipsă)', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("d) Metode de tratare aplicate")

    st.markdown("**Metoda 1: Înlocuire cu media (pentru variabile numerice continue)**")
    mean_sales = df_missing['Sales'].mean()
    mean_profit = df_missing['Profit'].mean()
    df_missing['Sales'] = df_missing['Sales'].fillna(mean_sales)
    df_missing['Profit'] = df_missing['Profit'].fillna(mean_profit)
    st.code(f"""
# Înlocuire Sales și Profit cu media
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())   # media Sales = {mean_sales:.2f}
df['Profit'] = df['Profit'].fillna(df['Profit'].mean()) # media Profit = {mean_profit:.2f}
    """, language='python')

    st.markdown("**Metoda 2: Înlocuire cu modul (pentru Postal Code — variabilă discretă)**")
    mode_postal = df_missing['Postal Code'].mode()[0]
    df_missing['Postal Code'] = df_missing['Postal Code'].fillna(mode_postal)
    st.code(f"""
# Înlocuire Postal Code cu modul (valoarea cea mai frecventă)
df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].mode()[0])  # modul = {mode_postal}
    """, language='python')

    st.markdown("**Metoda 3: Eliminarea rândurilor cu valori lipsă (dropna)**")
    st.code("""
# Alternativ, putem elimina rândurile cu valori lipsă
df_clean = df.dropna()  # elimină toate rândurile care au cel puțin o valoare lipsă
    """, language='python')

    st.subheader("e) Verificare după tratare")
    remaining = df_missing.isnull().sum().sum()
    st.success(f"După tratare: **{remaining} valori lipsă rămase** (din {n_missing * 2 + n_missing // 2} introduse)")

    interpretation_box("Interpretare economică",
        "Alegerea metodei de tratare depinde de context: înlocuirea cu media păstrează distribuția generală, "
        "dar poate masca variabilitatea naturală. Pentru analize financiare (Sales, Profit), media este o "
        "alegere conservatoare. Pentru coduri poștale (variabilă categorică), modul este mai potrivit. "
        "Eliminarea rândurilor este recomandată doar când procentul de valori lipsă este mic (<5%).")


# =============================================================================
# 3. VALORI EXTREME (OUTLIERI)
# =============================================================================
elif section == SECTIONS[2]:
    section_header("Detectarea și Tratarea Valorilor Extreme",
                   "Identificarea outlierilor cu metoda IQR și boxplot", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    Valorile extreme (outlieri) sunt observații care se abat semnificativ de la restul datelor.
    Pot fi erori de înregistrare sau valori reale dar atipice. Afectează negativ modelele
    statistice, în special regresia și K-Means.
    """)

    st.subheader("b) Detectarea outlierilor cu metoda IQR")
    st.markdown("""
    **Metoda IQR (Interquartile Range):**
    """)
    st.markdown("""
    <div class="formula-box">
    IQR = Q3 - Q1<br>
    Limita inferioară = Q1 - 1.5 × IQR<br>
    Limita superioară = Q3 + 1.5 × IQR<br>
    Orice valoare în afara acestor limite = outlier
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Boxplot — variabile numerice:**")
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for i, col_name in enumerate(numeric_cols):
            ax = axes[i // 2][i % 2]
            sns.boxplot(data=df_original, y=col_name, ax=ax, color='#e94560',
                        flierprops=dict(marker='o', markersize=3))
            ax.set_title(col_name, fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Statistici outlieri per coloană:**")
        outlier_stats = []
        for col_name in numeric_cols:
            Q1 = df_original[col_name].quantile(0.25)
            Q3 = df_original[col_name].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((df_original[col_name] < lower) | (df_original[col_name] > upper)).sum()
            outlier_stats.append({
                'Coloană': col_name,
                'Q1': round(Q1, 2),
                'Q3': round(Q3, 2),
                'IQR': round(IQR, 2),
                'Limita Inf.': round(lower, 2),
                'Limita Sup.': round(upper, 2),
                'Nr. Outlieri': n_outliers,
                'Procent (%)': round(n_outliers / len(df_original) * 100, 2)
            })
        st.dataframe(pd.DataFrame(outlier_stats), use_container_width=True, hide_index=True)

    st.subheader("c) Tratarea outlierilor — Metoda Capping (Winsorizing)")
    st.markdown("""
    Înlocuim valorile extreme cu limitele IQR (capping/winsorizing).
    Această metodă păstrează toate observațiile, dar limitează valorile extreme.
    """)

    df_capped = df_original.copy()
    capping_log = []
    for col_name in ['Sales', 'Profit']:
        Q1 = df_capped[col_name].quantile(0.25)
        Q3 = df_capped[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df_capped[(df_capped[col_name] < lower) | (df_capped[col_name] > upper)])
        df_capped[col_name] = df_capped[col_name].clip(lower=lower, upper=upper)
        after = len(df_capped[(df_capped[col_name] < lower) | (df_capped[col_name] > upper)])
        capping_log.append(f"{col_name}: {before} outlieri tratați (cap la [{lower:.2f}, {upper:.2f}])")

    st.code(f"""
# Capping cu metoda IQR
for col in ['Sales', 'Profit']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    """, language='python')

    for log in capping_log:
        st.info(f" {log}")

    st.subheader("d) Boxplot după tratare")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, col_name in enumerate(['Sales', 'Profit']):
        sns.boxplot(data=df_capped, y=col_name, ax=axes[i], color='#2d6a4f',
                    flierprops=dict(marker='o', markersize=3))
        axes[i].set_title(f'{col_name} (după capping)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    interpretation_box("Interpretare economică",
        "Coloana Profit are cei mai mulți outlieri (atât pozitivi cât și negativi), "
        "ceea ce indică tranzacții cu marje extreme — fie vânzări foarte profitabile, fie pierderi "
        "semnificative (discounturi agresive). Capping-ul cu IQR reduce impactul acestor valori "
        "extreme fără a le elimina complet, păstrând dimensiunea setului de date.")


# =============================================================================
# 4. CODIFICARE DATE
# =============================================================================
elif section == SECTIONS[3]:
    section_header("Codificarea Datelor Categorice",
                   "Transformarea variabilelor categorice în format numeric", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    Modelele de Machine Learning (KMeans, Regresie, etc.) lucrează cu date numerice.
    Variabilele categorice trebuie convertite în numere prin tehnici de **codificare**:
    **Label Encoding** (pentru variabile ordinale) și **One-Hot Encoding** (pentru variabile nominale).
    """)

    cat_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']

    st.subheader("b) Variabile categorice identificate")
    cat_info = []
    for col_name in cat_cols:
        cat_info.append({
            'Coloană': col_name,
            'Valori unice': df_original[col_name].nunique(),
            'Exemple': ', '.join(df_original[col_name].unique()[:5])
        })
    st.dataframe(pd.DataFrame(cat_info), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("c) Label Encoding")
        st.markdown("""
        Atribuie un număr unic fiecărei categorii.
        Potrivit pentru variabile **ordinale** (cu ordine naturală).
        """)
        from sklearn.preprocessing import LabelEncoder
        df_label = df_original[cat_cols].copy()
        le = LabelEncoder()
        label_results = {}
        for col_name in cat_cols:
            df_label[col_name + '_encoded'] = le.fit_transform(df_label[col_name])
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            label_results[col_name] = mapping

        st.code("""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_columns:
    df[col + '_encoded'] = le.fit_transform(df[col])
        """, language='python')

        st.markdown("**Mapări rezultate:**")
        for col_name, mapping in label_results.items():
            st.markdown(f"**{col_name}:** {mapping}")

        st.dataframe(df_label.head(8), use_container_width=True)

    with col2:
        st.subheader("d) One-Hot Encoding (get_dummies)")
        st.markdown("""
        Creează o coloană binară (0/1) pentru fiecare valoare unică.
        Potrivit pentru variabile **nominale** (fără ordine).
        """)

        df_onehot = pd.get_dummies(df_original[['Ship Mode', 'Segment', 'Region']],
                                    prefix=['ShipMode', 'Segment', 'Region'])

        st.code("""
# One-Hot Encoding cu pandas get_dummies
df_encoded = pd.get_dummies(
    df[['Ship Mode', 'Segment', 'Region']],
    prefix=['ShipMode', 'Segment', 'Region']
)
        """, language='python')

        st.markdown(f"**Coloane rezultate:** {len(df_onehot.columns)}")
        st.dataframe(df_onehot.head(8), use_container_width=True)

    st.subheader("e) Codificare cu map() — variabilă binară")
    st.markdown("Creăm o variabilă binară `Is_Profitable` (1 dacă Profit > 0, 0 altfel):")
    df_binary = df_original[['Profit']].copy()
    df_binary['Is_Profitable'] = (df_original['Profit'] > 0).astype(int)
    st.code("""
# Codificare binară manuală cu map/apply
df['Is_Profitable'] = (df['Profit'] > 0).astype(int)
    """, language='python')

    col1, col2 = st.columns(2)
    with col1:
        vc = df_binary['Is_Profitable'].value_counts()
        st.dataframe(pd.DataFrame({
            'Categorie': ['Profitabil (1)', 'Neprofitabil (0)'],
            'Nr. tranzacții': [vc.get(1, 0), vc.get(0, 0)],
            'Procent': [f"{vc.get(1, 0)/len(df_binary)*100:.1f}%", f"{vc.get(0, 0)/len(df_binary)*100:.1f}%"]
        }), use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        vc.plot.pie(autopct='%1.1f%%', colors=['#2d6a4f', '#e94560'],
                    labels=['Profitabil', 'Neprofitabil'], ax=ax)
        ax.set_ylabel('')
        ax.set_title('Distribuția profitabilității', fontsize=10)
        st.pyplot(fig)
        plt.close()

    interpretation_box("Interpretare economică",
        "Label Encoding este potrivit pentru Ship Mode (are o ordine implicită: Same Day > First Class > "
        "Second Class > Standard Class). One-Hot Encoding este potrivit pentru Region și Segment, "
        "unde nu există ordine naturală. ~80% din tranzacții sunt profitabile, ceea ce ne oferă "
        "o variabilă target dezechilibrată pentru clasificare.")


# =============================================================================
# 5. SCALARE DATE
# =============================================================================
elif section == SECTIONS[4]:
    section_header("Scalarea Datelor",
                   "Standardizare și normalizare pentru algoritmii de ML", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    Algoritmii precum KMeans și Regresia sunt sensibili la scala variabilelor.
    O variabilă cu valori de ordinul miilor (Sales) va domina una cu valori de ordinul unităților (Discount).
    **Scalarea** aduce toate variabilele la o scală comparabilă.
    """)

    num_cols = ['Sales', 'Quantity', 'Discount', 'Profit']

    st.subheader("b) Distribuțiile înainte de scalare")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for i, col_name in enumerate(num_cols):
        sns.histplot(df_original[col_name], kde=True, ax=axes[i], color='#e94560', bins=30)
        axes[i].set_title(f'{col_name}\nMedia={df_original[col_name].mean():.1f}', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("c) StandardScaler (Standardizare Z-score)")
        st.markdown("""
        <div class="formula-box">
        z = (x - μ) / σ<br>
        Rezultat: media = 0, deviația standard = 1<br>
        Potrivit pentru: Regresie, SVM, Rețele neurale
        </div>
        """, unsafe_allow_html=True)

        from sklearn.preprocessing import StandardScaler
        scaler_std = StandardScaler()
        df_standard = pd.DataFrame(
            scaler_std.fit_transform(df_original[num_cols]),
            columns=[f'{c}_std' for c in num_cols]
        )

        st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])
        """, language='python')

        st.dataframe(df_standard.describe().T.round(3), use_container_width=True)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        for i, col_name in enumerate(df_standard.columns):
            sns.histplot(df_standard[col_name], kde=True, ax=axes[i], color='#2d6a4f', bins=30)
            axes[i].set_title(f'{col_name}\nMedia={df_standard[col_name].mean():.3f}', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("d) MinMaxScaler (Normalizare)")
        st.markdown("""
        <div class="formula-box">
        x_norm = (x - x_min) / (x_max - x_min)<br>
        Rezultat: valori în intervalul [0, 1]<br>
        Potrivit pentru: KMeans, KNN, Rețele neurale
        </div>
        """, unsafe_allow_html=True)

        from sklearn.preprocessing import MinMaxScaler
        scaler_mm = MinMaxScaler()
        df_minmax = pd.DataFrame(
            scaler_mm.fit_transform(df_original[num_cols]),
            columns=[f'{c}_norm' for c in num_cols]
        )

        st.code("""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df[numeric_cols])
        """, language='python')

        st.dataframe(df_minmax.describe().T.round(3), use_container_width=True)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        for i, col_name in enumerate(df_minmax.columns):
            sns.histplot(df_minmax[col_name], kde=True, ax=axes[i], color='#3a0ca3', bins=30)
            axes[i].set_title(f'{col_name}\nMin={df_minmax[col_name].min():.2f}, Max={df_minmax[col_name].max():.2f}',
                              fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    interpretation_box("Interpretare economică",
        "StandardScaler este preferat pentru regresie (coeficienții devin comparabili între variabile). "
        "MinMaxScaler este preferat pentru KMeans (distanțele euclidiene sunt echitabile). "
        "IMPORTANT: scalarea se face DOAR pe setul de antrenament, iar transformarea se aplică "
        "pe setul de test — pentru a evita data leakage.")


# =============================================================================
# 6. PRELUCRĂRI STATISTICE & GRUPĂRI
# =============================================================================
elif section == SECTIONS[5]:
    section_header("Prelucrări Statistice, Grupări și Agregări",
                   "Pandas: groupby, agg, funcții de grup, merge, pivot", "#2d6a4f")

    # --- 6a: Statistici descriptive ---
    st.subheader("a) Statistici descriptive de bază")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vânzări Totale", f"${df_original['Sales'].sum():,.0f}")
    col2.metric("Profit Total", f"${df_original['Profit'].sum():,.0f}")
    col3.metric("Vânzare Medie", f"${df_original['Sales'].mean():,.2f}")
    col4.metric("Discount Mediu", f"{df_original['Discount'].mean()*100:.1f}%")

    # --- 6b: Grupări simple ---
    st.subheader("b) Grupări cu groupby — vânzări și profit pe categorie")

    group_cat = df_original.groupby('Category').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Avg_Discount=('Discount', 'mean'),
        Nr_Tranzactii=('Sales', 'count')
    ).round(2)

    st.code("""
# Grupare pe categorie cu funcții de agregare multiple
group_cat = df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Avg_Discount=('Discount', 'mean'),
    Nr_Tranzactii=('Sales', 'count')
)
    """, language='python')
    st.dataframe(group_cat, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(group_cat.reset_index(), x='Category', y=['Total_Sales', 'Total_Profit'],
                     barmode='group', title='Vânzări și Profit pe Categorie',
                     color_discrete_sequence=['#e94560', '#2d6a4f'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(group_cat.reset_index(), values='Nr_Tranzactii', names='Category',
                     title='Distribuția tranzacțiilor pe categorie',
                     color_discrete_sequence=['#e94560', '#3a0ca3', '#2d6a4f'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- 6c: Grupări multiple ---
    st.subheader("c) Grupări complexe — Region × Category")
    group_multi = df_original.groupby(['Region', 'Category']).agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum'),
        Avg_Quantity=('Quantity', 'mean')
    ).round(2)

    st.code("""
# Grupare pe mai multe coloane
group_multi = df.groupby(['Region', 'Category']).agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum'),
    Avg_Quantity=('Quantity', 'mean')
)
    """, language='python')
    st.dataframe(group_multi, use_container_width=True)

    # --- 6d: Funcții de grup multiple pe aceeași coloană ---
    st.subheader("d) Funcții de grup multiple pe aceeași coloană")
    group_funcs = df_original.groupby('Sub-Category').agg(
        Sales_Sum=('Sales', 'sum'),
        Sales_Mean=('Sales', 'mean'),
        Sales_Min=('Sales', 'min'),
        Sales_Max=('Sales', 'max'),
        Sales_Std=('Sales', 'std'),
        Profit_Sum=('Profit', 'sum')
    ).round(2).sort_values('Sales_Sum', ascending=False)

    st.code("""
# Funcții multiple pe aceeași coloană
group_funcs = df.groupby('Sub-Category').agg(
    Sales_Sum=('Sales', 'sum'),
    Sales_Mean=('Sales', 'mean'),
    Sales_Min=('Sales', 'min'),
    Sales_Max=('Sales', 'max'),
    Sales_Std=('Sales', 'std'),
    Profit_Sum=('Profit', 'sum')
).sort_values('Sales_Sum', ascending=False)
    """, language='python')
    st.dataframe(group_funcs, use_container_width=True)

    fig = px.bar(group_funcs.reset_index().head(10), x='Sub-Category', y='Sales_Sum',
                 color='Profit_Sum', color_continuous_scale='RdYlGn',
                 title='Top 10 Sub-Categorii după Vânzări (colorat după Profit)')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # --- 6e: Pivot Table ---
    st.subheader("e) Pivot Table — Segment × Region")
    pivot = pd.pivot_table(df_original, values='Sales', index='Segment',
                           columns='Region', aggfunc='sum').round(2)
    st.code("""
pivot = pd.pivot_table(df, values='Sales', index='Segment',
                       columns='Region', aggfunc='sum')
    """, language='python')
    st.dataframe(pivot, use_container_width=True)

    fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='YlOrRd',
                    title='Heatmap Vânzări: Segment × Region')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # --- 6f: Corelație ---
    st.subheader("f) Matricea de corelație")
    corr = df_original[['Sales', 'Quantity', 'Discount', 'Profit', 'Delivery_Days']].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title('Corelația Pearson între variabilele numerice', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    interpretation_box("Interpretare economică",
        "Technology are cele mai mari vânzări dar și cea mai mare variabilitate a profitului. "
        "Office Supplies generează cele mai multe tranzacții dar cu valoare medie mai mică. "
        "Discountul este corelat negativ cu profitul (-0.22), confirmând că reducerile agresive "
        "erodează marjele. Vânzările și profitul sunt puternic corelate (0.48), dar nu perfect — "
        "indicând că unele produse au marje semnificativ diferite.")


# =============================================================================
# 7. VIZUALIZARE GEOSPAȚIALĂ (GEOPANDAS)
# =============================================================================
elif section == SECTIONS[6]:
    section_header("Vizualizare Geospațială cu Geopandas",
                   "Harta vânzărilor și profitului pe statele din SUA", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    Vizualizarea geospațială permite identificarea rapidă a regiunilor performante
    și a celor cu potențial de creștere. Utilizăm **geopandas** pentru a crea hărți
    coroplethe (colorate după intensitatea valorii).
    """)

    import geopandas as gpd
    from shapely.geometry import Point
    import json

    # --- Aggregate by state ---
    state_data = df_original.groupby('State').agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum'),
        Nr_Orders=('Order ID', 'nunique'),
        Avg_Discount=('Discount', 'mean')
    ).reset_index()
    state_data['Profit_Margin'] = (state_data['Profit'] / state_data['Sales'] * 100).round(2)

    # --- Load state coordinates ---
    with open('us_states_coords.json') as f:
        state_coords = json.load(f)

    # Create GeoDataFrame with points
    geometry = []
    matched_states = []
    for _, row in state_data.iterrows():
        if row['State'] in state_coords:
            coords = state_coords[row['State']]
            geometry.append(Point(coords['lon'], coords['lat']))
            matched_states.append(True)
        else:
            geometry.append(Point(0, 0))
            matched_states.append(False)

    gdf = gpd.GeoDataFrame(state_data, geometry=geometry, crs="EPSG:4326")
    gdf = gdf[matched_states]

    st.code("""
import geopandas as gpd
from shapely.geometry import Point

# Creăm GeoDataFrame din coordonatele statelor
state_data = df.groupby('State').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum'),
    Nr_Orders=('Order ID', 'nunique')
).reset_index()

# Adăugăm geometrie (coordonate GPS) pentru fiecare stat
geometry = [Point(lon, lat) for lon, lat in state_coordinates]
gdf = gpd.GeoDataFrame(state_data, geometry=geometry, crs="EPSG:4326")
    """, language='python')

    st.subheader("b) Harta vânzărilor pe state")
    metric_choice = st.selectbox("Selectează metrica:", ['Sales', 'Profit', 'Nr_Orders', 'Profit_Margin'])

    fig = px.scatter_geo(
        gdf.drop(columns='geometry'),
        lat=gdf.geometry.y,
        lon=gdf.geometry.x,
        size=gdf['Sales'].clip(lower=100),
        color=metric_choice,
        hover_name='State',
        hover_data=['Sales', 'Profit', 'Nr_Orders', 'Profit_Margin'],
        color_continuous_scale='RdYlGn' if metric_choice in ['Profit', 'Profit_Margin'] else 'YlOrRd',
        scope='usa',
        title=f'{metric_choice} pe State',
        size_max=40
    )
    fig.update_layout(height=500, geo=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("c) Top 10 State după vânzări și profit")
    col1, col2 = st.columns(2)
    with col1:
        top_sales = state_data.nlargest(10, 'Sales')
        fig = px.bar(top_sales, x='Sales', y='State', orientation='h',
                     color='Sales', color_continuous_scale='YlOrRd',
                     title='Top 10 State — Vânzări')
        fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_profit = state_data.nlargest(10, 'Profit')
        fig = px.bar(top_profit, x='Profit', y='State', orientation='h',
                     color='Profit', color_continuous_scale='RdYlGn',
                     title='Top 10 State — Profit')
        fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("d) State cu pierderi")
    loss_states = state_data[state_data['Profit'] < 0].sort_values('Profit')
    if len(loss_states) > 0:
        st.dataframe(loss_states, use_container_width=True, hide_index=True)
    else:
        st.info("Nu există state cu profit total negativ.")

    interpretation_box("Interpretare economică",
        "California și New York sunt cele mai mari piețe ca volum de vânzări. "
        "Unele state cu vânzări mari au marje de profit scăzute sau negative — indicând "
        "politici de discount agresive sau costuri logistice ridicate. "
        "Vizualizarea geospațială evidențiază oportunitățile de extindere în statele "
        "cu penetrare scăzută dar potențial economic ridicat (ex: statele din Midwest).")


# =============================================================================
# 8. CLUSTERING K-MEANS
# =============================================================================
elif section == SECTIONS[7]:
    section_header("Clustering K-Means (scikit-learn)",
                   "Segmentarea clienților pe baza comportamentului de cumpărare", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    **Obiectiv:** Identificarea segmentelor naturale de clienți pe baza
    vânzărilor totale, profitului și frecvenței comenzilor. K-Means
    grupează clienții în clustere cu comportament similar.
    """)

    # --- Pregătire date la nivel de client ---
    customer_data = df_original.groupby('Customer ID').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Nr_Orders=('Order ID', 'nunique'),
        Avg_Discount=('Discount', 'mean'),
        Avg_Quantity=('Quantity', 'mean')
    ).reset_index()

    st.subheader("b) Pregătirea datelor — agregare la nivel de client")
    st.code("""
# Agregare la nivel de client
customer_data = df.groupby('Customer ID').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Nr_Orders=('Order ID', 'nunique'),
    Avg_Discount=('Discount', 'mean'),
    Avg_Quantity=('Quantity', 'mean')
).reset_index()
    """, language='python')
    st.dataframe(customer_data.head(10), use_container_width=True)
    st.caption(f"Total clienți: {len(customer_data)}")

    # --- Scalare ---
    st.subheader("c) Scalarea datelor (StandardScaler)")
    from sklearn.preprocessing import StandardScaler

    features = ['Total_Sales', 'Total_Profit', 'Nr_Orders']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_data[features])

    st.code("""
from sklearn.preprocessing import StandardScaler
features = ['Total_Sales', 'Total_Profit', 'Nr_Orders']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data[features])
    """, language='python')

    # --- Metoda Elbow ---
    st.subheader("d) Metoda Elbow — determinarea numărului optim de clustere")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    wcss = []
    sil_scores = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=wcss, mode='lines+markers',
                                  marker=dict(size=8, color='#e94560'),
                                  line=dict(color='#e94560', width=2)))
        fig.update_layout(title='Metoda Elbow (WCSS)', xaxis_title='Număr clustere (K)',
                          yaxis_title='WCSS (Inerție)', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=sil_scores, mode='lines+markers',
                                  marker=dict(size=8, color='#2d6a4f'),
                                  line=dict(color='#2d6a4f', width=2)))
        fig.update_layout(title='Silhouette Score per K', xaxis_title='Număr clustere (K)',
                          yaxis_title='Silhouette Score', height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.code("""
# Metoda Elbow + Silhouette Score
wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    """, language='python')

    # --- Fit K-Means cu K optim ---
    optimal_k = 3
    st.subheader(f"e) Aplicare K-Means cu K={optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, customer_data['Cluster'])
    st.metric("Silhouette Score", f"{sil:.4f}",
              help="Între -1 și 1. Mai aproape de 1 = clustere bine definite.")

    st.code(f"""
kmeans = KMeans(n_clusters={optimal_k}, init='k-means++', random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)
silhouette = silhouette_score(X_scaled, customer_data['Cluster'])
print(f"Silhouette Score: {{silhouette:.4f}}")
    """, language='python')

    # --- Vizualizare clustere ---
    st.subheader("f) Vizualizarea clusterelor")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(customer_data, x='Total_Sales', y='Total_Profit',
                         color='Cluster', size='Nr_Orders',
                         color_discrete_sequence=['#e94560', '#2d6a4f', '#3a0ca3'],
                         title='Clustere: Vânzări vs Profit',
                         labels={'Total_Sales': 'Vânzări Totale ($)',
                                 'Total_Profit': 'Profit Total ($)'})
        # Adaugă centroids
        centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
        for i, c in enumerate(centroids_orig):
            fig.add_trace(go.Scatter(x=[c[0]], y=[c[1]], mode='markers',
                marker=dict(size=18, symbol='x', color='black', line=dict(width=2)),
                name=f'Centroid {i}', showlegend=True))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(customer_data, x='Nr_Orders', y='Total_Sales',
                         color='Cluster', size=customer_data['Total_Profit'].clip(lower=0),
                         color_discrete_sequence=['#e94560', '#2d6a4f', '#3a0ca3'],
                         title='Clustere: Nr. Comenzi vs Vânzări',
                         labels={'Nr_Orders': 'Număr Comenzi',
                                 'Total_Sales': 'Vânzări Totale ($)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Statistici per cluster ---
    st.subheader("g) Statistici descriptive per cluster")
    cluster_stats = customer_data.groupby('Cluster')[features + ['Avg_Discount']].agg(
        ['mean', 'median', 'count']
    ).round(2)
    st.dataframe(cluster_stats, use_container_width=True)

    # --- Boxplot per cluster ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, feat in enumerate(features):
        sns.boxplot(data=customer_data, x='Cluster', y=feat, ax=axes[i],
                    palette=['#e94560', '#2d6a4f', '#3a0ca3'])
        axes[i].set_title(feat, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    interpretation_box("Interpretare economică",
        "K-Means a identificat 3 segmente naturale de clienți: "
        "(1) Clienți cu vânzări mici și puține comenzi — segment mass-market, volum mare de clienți; "
        "(2) Clienți cu vânzări medii și comenzi frecvente — segment loial, baza stabilă de venituri; "
        "(3) Clienți premium cu vânzări mari — segment VIP, necesită atenție personalizată. "
        "Strategia comercială poate fi diferențiată pe segmente: retenție pentru VIP, "
        "up-selling pentru segmentul mediu, achiziție eficientă pentru mass-market.")


# =============================================================================
# 9. REGRESIE LOGISTICĂ (scikit-learn)
# =============================================================================
elif section == SECTIONS[8]:
    section_header("Clasificare — Regresie Logistică (scikit-learn)",
                   "Predicția profitabilității tranzacțiilor", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    **Obiectiv:** Predicția dacă o tranzacție va fi profitabilă (Profit > 0) sau nu,
    pe baza caracteristicilor tranzacției: Sales, Quantity, Discount, Delivery_Days.
    Aceasta este o problemă de **clasificare binară**.
    """)

    # --- Pregătire date ---
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, roc_auc_score, roc_curve, f1_score)

    df_clf = df_original.copy()
    df_clf['Is_Profitable'] = (df_clf['Profit'] > 0).astype(int)

    # Label encode categorice
    le = LabelEncoder()
    df_clf['Ship_Mode_enc'] = le.fit_transform(df_clf['Ship Mode'])
    df_clf['Segment_enc'] = le.fit_transform(df_clf['Segment'])
    df_clf['Category_enc'] = le.fit_transform(df_clf['Category'])
    df_clf['Region_enc'] = le.fit_transform(df_clf['Region'])

    feature_cols = ['Sales', 'Quantity', 'Discount', 'Delivery_Days',
                    'Ship_Mode_enc', 'Segment_enc', 'Category_enc', 'Region_enc']
    X = df_clf[feature_cols]
    y = df_clf['Is_Profitable']

    st.subheader("b) Distribuția variabilei target")
    col1, col2 = st.columns(2)
    with col1:
        vc = y.value_counts()
        st.dataframe(pd.DataFrame({
            'Clasă': ['Profitabil (1)', 'Neprofitabil (0)'],
            'Nr.': [vc[1], vc[0]],
            '%': [f"{vc[1]/len(y)*100:.1f}%", f"{vc[0]/len(y)*100:.1f}%"]
        }), use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        vc.plot.bar(color=['#2d6a4f', '#e94560'], ax=ax)
        ax.set_title('Distribuție Target', fontsize=10)
        ax.set_xticklabels(['Profitabil', 'Neprofitabil'], rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Train/Test Split ---
    st.subheader("c) Separare Train/Test și scalare")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                         random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # DOAR transform, nu fit_transform!
    """, language='python')
    st.info(f"Train: {len(X_train)} exemple | Test: {len(X_test)} exemple")

    # --- Antrenare model ---
    st.subheader("d) Antrenarea modelului de Regresie Logistică")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

    st.code("""
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    """, language='python')

    # --- Metrici ---
    st.subheader("e) Evaluarea modelului")
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("AUC-ROC", f"{auc:.4f}")
    m3.metric("F1-Score", f"{f1:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matricea de Confuzie:**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Neprofitabil', 'Profitabil'],
                    yticklabels=['Neprofitabil', 'Profitabil'])
        ax.set_xlabel('Prezis')
        ax.set_ylabel('Real')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Curba ROC:**")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  name=f'Reg. Logistică (AUC={auc:.3f})',
                                  line=dict(color='#e94560', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Random (AUC=0.5)',
                                  line=dict(color='gray', dash='dash')))
        fig.update_layout(title='Curba ROC', xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate', height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("f) Raportul de clasificare")
    report = classification_report(y_test, y_pred, target_names=['Neprofitabil', 'Profitabil'],
                                    output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.subheader("g) Importanța variabilelor (coeficienți)")
    coef_df = pd.DataFrame({
        'Variabilă': feature_cols,
        'Coeficient': lr.coef_[0]
    }).sort_values('Coeficient', key=abs, ascending=False)

    fig = px.bar(coef_df, x='Coeficient', y='Variabilă', orientation='h',
                 color='Coeficient', color_continuous_scale='RdBu_r',
                 title='Coeficienții Regresiei Logistice')
    fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    interpretation_box("Interpretare economică",
        "Discountul este cel mai puternic predictor negativ al profitabilității — "
        "tranzacțiile cu discount mare au probabilitate semnificativ mai mare de a fi neprofitabile. "
        "Sales are un coeficient pozitiv — vânzările mari sunt de obicei profitabile. "
        "Modelul poate fi folosit de echipa comercială pentru a evalua riscul de pierdere "
        "înainte de a aplica discounturi agresive.")


# =============================================================================
# 10. REGRESIE MULTIPLĂ (statsmodels)
# =============================================================================
elif section == SECTIONS[9]:
    section_header("Regresie Multiplă (statsmodels)",
                   "Modelarea relației dintre variabilele explicative și Profit", "#2d6a4f")

    st.subheader("a) Definirea problemei")
    st.markdown("""
    **Obiectiv:** Construirea unui model de regresie liniară multiplă pentru a explica
    variația Profitului în funcție de Sales, Quantity, Discount și Delivery_Days.
    Utilizăm pachetul **statsmodels** care oferă un raport statistic complet
    (coeficienți, p-values, R², intervale de încredere).
    """)

    import statsmodels.api as sm

    # --- Pregătire date ---
    df_reg = df_original.copy()
    df_reg = df_reg.dropna(subset=['Sales', 'Quantity', 'Discount', 'Profit', 'Delivery_Days'])

    # Variabilele explicative
    feature_cols_reg = ['Sales', 'Quantity', 'Discount', 'Delivery_Days']
    X_reg = df_reg[feature_cols_reg]
    y_reg = df_reg['Profit']

    st.subheader("b) Analiza relațiilor — Scatter plots")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, col_name in enumerate(feature_cols_reg):
        ax = axes[i // 2][i % 2]
        ax.scatter(df_reg[col_name], df_reg['Profit'], alpha=0.3, s=10, color='#3a0ca3')
        ax.set_xlabel(col_name, fontsize=10)
        ax.set_ylabel('Profit', fontsize=10)
        ax.set_title(f'Profit vs {col_name}', fontsize=11)
        # Linie de trend
        z = np.polyfit(df_reg[col_name], df_reg['Profit'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_reg[col_name].min(), df_reg[col_name].max(), 100)
        ax.plot(x_line, p(x_line), color='#e94560', linewidth=2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Construirea modelului OLS ---
    st.subheader("c) Modelul OLS (Ordinary Least Squares)")
    X_reg_const = sm.add_constant(X_reg)  # Adaugă interceptul

    st.code("""
import statsmodels.api as sm

X = df[['Sales', 'Quantity', 'Discount', 'Delivery_Days']]
X = sm.add_constant(X)  # adaugă coloana de intercept (constanta)
y = df['Profit']

model = sm.OLS(y, X).fit()
print(model.summary())
    """, language='python')

    model_ols = sm.OLS(y_reg, X_reg_const).fit()

    st.subheader("d) Raportul complet al modelului")
    st.text(model_ols.summary().as_text())

    # --- Metrici principale ---
    st.subheader("e) Metrici de evaluare")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{model_ols.rsquared:.4f}",
              help="Proporția varianței explicată de model")
    m2.metric("R² Ajustat", f"{model_ols.rsquared_adj:.4f}",
              help="R² corectat pentru numărul de variabile")
    m3.metric("F-statistic", f"{model_ols.fvalue:.2f}",
              help="Semnificanța generală a modelului")
    m4.metric("AIC", f"{model_ols.aic:.2f}",
              help="Criteriu informațional Akaike (mai mic = mai bun)")

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    y_pred_ols = model_ols.predict(X_reg_const)
    mse = mean_squared_error(y_reg, y_pred_ols)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_reg, y_pred_ols)

    m1, m2, m3 = st.columns(3)
    m1.metric("MSE", f"{mse:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("MAE", f"{mae:.2f}")

    # --- Coeficienți ---
    st.subheader("f) Coeficienții modelului și semnificația statistică")
    coef_table = pd.DataFrame({
        'Variabilă': model_ols.params.index,
        'Coeficient': model_ols.params.values.round(4),
        'Std Error': model_ols.bse.values.round(4),
        't-value': model_ols.tvalues.values.round(4),
        'P-value': model_ols.pvalues.values.round(6),
        'Semnificativ?': ['Da ' if p < 0.05 else 'Nu ' for p in model_ols.pvalues.values]
    })
    st.dataframe(coef_table, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="formula-box">
    Profit = {:.2f} + {:.4f} × Sales + {:.4f} × Quantity + {:.4f} × Discount + {:.4f} × Delivery_Days
    </div>
    """.format(model_ols.params['const'], model_ols.params['Sales'],
               model_ols.params['Quantity'], model_ols.params['Discount'],
               model_ols.params['Delivery_Days']), unsafe_allow_html=True)

    # --- Analiza rezidualurilor ---
    st.subheader("g) Analiza rezidualurilor")
    residuals = model_ols.resid

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_pred_ols, residuals, alpha=0.3, s=10, color='#3a0ca3')
        ax.axhline(y=0, color='#e94560', linewidth=2)
        ax.set_xlabel('Valori prezise', fontsize=10)
        ax.set_ylabel('Rezidualuri', fontsize=10)
        ax.set_title('Rezidualuri vs Valori Prezise', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=50, color='#3a0ca3', ax=ax)
        ax.set_title('Distribuția Rezidualurilor', fontsize=11)
        ax.set_xlabel('Rezidualuri')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Predicted vs Actual ---
    st.subheader("h) Valori prezise vs valori reale")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_reg.values[:500], y=y_pred_ols.values[:500],
                              mode='markers', marker=dict(size=4, color='#3a0ca3', opacity=0.4),
                              name='Observații'))
    min_val = min(y_reg.min(), y_pred_ols.min())
    max_val = max(y_reg.max(), y_pred_ols.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode='lines', line=dict(color='#e94560', dash='dash', width=2),
                              name='Linie ideală (y=x)'))
    fig.update_layout(title='Profit Real vs Profit Prezis (primele 500 observații)',
                      xaxis_title='Profit Real ($)', yaxis_title='Profit Prezis ($)',
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Predicție individuală ---
    st.subheader("i) Predicție individuală")
    st.markdown("Introduceți valorile pentru a obține o predicție de profit:")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        p_sales = st.number_input("Sales ($)", value=200.0, step=10.0)
    with pc2:
        p_qty = st.number_input("Quantity", value=3, step=1)
    with pc3:
        p_disc = st.number_input("Discount (%)", value=10.0, step=5.0, min_value=0.0, max_value=80.0)
    with pc4:
        p_days = st.number_input("Delivery Days", value=4, step=1)

    pred_input = [1, p_sales, p_qty, p_disc / 100, p_days]
    predicted_profit = model_ols.predict(pred_input)[0]
    st.success(f"**Profit estimat: ${predicted_profit:.2f}**")

    interpretation_box("Interpretare economică",
        f"Modelul explică {model_ols.rsquared*100:.1f}% din variația profitului (R²={model_ols.rsquared:.4f}). "
        "Sales este cel mai puternic predictor pozitiv — fiecare dolar vândut adaugă aproximativ "
        f"{model_ols.params['Sales']:.4f}$ la profit. "
        "Discountul are un efect negativ puternic — fiecare punct procentual de discount "
        f"reduce profitul cu ${abs(model_ols.params['Discount']):.2f}. "
        "Aceste rezultate confirmă că strategia de discount trebuie gestionată atent: "
        "discounturile peste 20% sunt asociate cu pierderi frecvente de profit.")
