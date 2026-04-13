# Proiect Pachete Software — Python (Streamlit)
## Superstore Sales Analytics

### Cum se rulează:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn geopandas statsmodels plotly openpyxl
streamlit run app.py
```

### Fișiere necesare (în același folder):
- `app.py` — aplicația Streamlit
- `Superstore.xlsx` — setul de date
- `us_states_coords.json` — coordonate GPS state SUA (pentru geopandas)

### Cerințe acoperite (10 funcționalități):
1. **Streamlit** — navigare sidebar, widget-uri interactive, grafice, metrici
2. **Geopandas** — hartă geospațială a vânzărilor pe state SUA
3. **Valori lipsă** — detectare, simulare, tratare (fillna cu medie/mod, dropna)
4. **Valori extreme** — detectare IQR + boxplot, tratare prin capping/winsorizing
5. **Codificare date** — LabelEncoder, One-Hot Encoding (get_dummies), codificare binară
6. **Scalare date** — StandardScaler (Z-score), MinMaxScaler (normalizare [0,1])
7. **Prelucrări statistice pandas** — describe, corr, pivot_table
8. **Funcții de grup** — groupby, agg cu funcții multiple, merge, pivot
9. **Scikit-learn: KMeans** — segmentare clienți, Elbow, Silhouette Score
10. **Scikit-learn: Regresie Logistică** — clasificare binară, Confusion Matrix, ROC-AUC
11. **Statsmodels: Regresie Multiplă OLS** — coeficienți, p-values, R², analiza rezidualurilor

### Structura fiecărei secțiuni:
- a) Definirea problemei
- b) Informații necesare pentru rezolvare
- c) Metode de calcul / algoritmi / formule
- d) Prezentarea rezultatelor
- e) Interpretarea economică a rezultatelor
