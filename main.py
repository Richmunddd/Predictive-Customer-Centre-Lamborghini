import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
import sys
import mplcursors

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# 1. LOAD DATA

df = pd.read_csv('lamborghini_sales_2020_2025.csv')

df_clean = df.drop(columns=['Turbo (Yes/No)'], errors='ignore')

# 2. K-MEANS CLUSTERING
cluster_features = ['Base Price (USD)', 'Horsepower']
scaler = StandardScaler()
scaled = scaler.fit_transform(df_clean[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(scaled)
sil_score = silhouette_score(scaled, df_clean['Cluster'])

# 3. MACHINE LEARNING MODEL PIPELINE
features = ['Year', 'Model', 'Region', 'Base Price (USD)', 'Horsepower', 'Cluster']
X = df_clean[features]
y = np.log1p(df_clean['Sales Volume']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical = ['Model', 'Region']
numeric = ['Year', 'Base Price (USD)', 'Horsepower', 'Cluster']

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numeric)
])

model = Pipeline([
    ('prep', preprocess),
    ('reg', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
])

print("Training the predictive model...")
model.fit(X_train, y_train)

# 4. VALIDATION

#ML Optimizer 
regressor = model.named_steps['reg']
opt_info = {
    "Algorithm": "Gradient Descent",
    "Loss Function": regressor.loss,
    "Learning Rate": regressor.learning_rate,
    "Iterations (n_estimators)": regressor.n_estimators
}

print("\n--- Optimizer Identified ---")
for key, value in opt_info.items():
    print(f"{key}: {value}")
print("----------------------------\n")

pred = np.expm1(model.predict(X_test))
actual = np.expm1(y_test)
r2 = r2_score(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100

# 5. 2026 FORECAST 
df_2026 = df_clean.sort_values('Year').groupby(['Model', 'Region']).last().reset_index()
df_2026['Year'] = 2026
df_2026['Base Price (USD)'] *= 1.02 

scaled_2026 = scaler.transform(df_2026[cluster_features])
df_2026['Cluster'] = kmeans.predict(scaled_2026)

pred_log = model.predict(df_2026[features])
df_2026['Predicted_Sales'] = np.expm1(pred_log).round(0)

# 6. UI 
def on_closing():
    root.quit()
    root.destroy()

root = tk.Tk()
root.title("Lamborghini Forecast Dashboard")
root.geometry("1100x900")
root.protocol("WM_DELETE_WINDOW", on_closing)

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scroll.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

# 7. PLOTS
fig = plt.figure(figsize=(10, 25))
plt.subplots_adjust(hspace=0.45)

# Graph 1: Market Segmentation 
ax1 = fig.add_subplot(5, 1, 1)
scatter = ax1.scatter(df_clean['Horsepower'], df_clean['Base Price (USD)'], c=df_clean['Cluster'], cmap='viridis', s=90)
ax1.set_title("1. Market Segmentation (K-Means Clustering)", fontweight='bold')
ax1.set_xlabel("Horsepower")
ax1.set_ylabel("Price (USD)")

cursor1 = mplcursors.cursor(scatter, hover=True)
@cursor1.connect("add")
def hover1(sel):
    idx = sel.index
    row = df_clean.iloc[idx]
    sel.annotation.set_text(f"Model: {row['Model']}\nHP: {row['Horsepower']}\nPrice: ${row['Base Price (USD)']:,.0f}")

# Graph 2: 2026 Regional Forecast Breakdown 
ax2 = fig.add_subplot(5, 1, 2)
plot_data = df_2026.pivot_table(index='Model', columns='Region', values='Predicted_Sales', aggfunc='sum').fillna(0)
plot_data.plot(kind='bar', stacked=True, ax=ax2, color=['#003366', '#FFD700', '#555555'], edgecolor='black')
ax2.set_title("2. 2026 Forecast Breakdown (All Car Models)", fontweight='bold')
ax2.set_ylabel("Predicted Units")
ax2.set_xticklabels(plot_data.index, rotation=20)

cursor2 = mplcursors.cursor(ax2, hover=True)
@cursor2.connect("add")
def hover2(sel):
    label = sel.artist.get_label()
    model_idx = int(round(sel.target[0]))
    if 0 <= model_idx < len(plot_data.index):
        model_name = plot_data.index[model_idx]
        val = plot_data.loc[model_name, label]
        sel.annotation.set_text(f"Model: {model_name}\nRegion: {label}\nPredicted: {int(val)}")

# Graph 3: Key Predictive Indicators 
ax3 = fig.add_subplot(5, 1, 3)
reg = model.named_steps['reg']
pre = model.named_steps['prep']
cat_names = pre.named_transformers_['cat'].get_feature_names_out(['Model', 'Region'])
feature_names = np.concatenate([cat_names, numeric])
importance = pd.Series(reg.feature_importances_, index=feature_names).sort_values()
bars_imp = ax3.barh(importance.index, importance.values, color='skyblue')
ax3.set_title("3. Data Logic Anchors (Predictive Weight)", fontweight='bold')

cursor3 = mplcursors.cursor(bars_imp, hover=True)

# Graph 4: Total Sales by Year  
ax4 = fig.add_subplot(5, 1, 4)
hist_sales = df_clean.groupby('Year')['Sales Volume'].sum()

active_models = ['Revuelto', 'Huracán (Temerario)', 'Urus SE']
future_total = df_2026[df_2026['Model'].isin(active_models)]['Predicted_Sales'].sum()

years = [str(y) for y in hist_sales.index] + ["2026 (Pred)"]
sales_values = list(hist_sales.values) + [future_total]
colors = ['#808080'] * len(hist_sales) + ['#D4AF37'] 

bars_yearly = ax4.bar(years, sales_values, color=colors, edgecolor='black')
ax4.set_title("4. Total Sales Volume: Growth Projection", fontweight='bold')
ax4.set_ylabel("Total Units Sold")

for bar in bars_yearly:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, yval + 500, f'{int(yval):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

cursor4 = mplcursors.cursor(bars_yearly, hover=True)
@cursor4.connect("add")
def hover4(sel):
    year_label = years[sel.index]
    val = sales_values[sel.index]
    sel.annotation.set_text(f"Year: {year_label}\nTotal: {int(val):,}")

#  Table 5: Metrics 
ax5 = fig.add_subplot(5, 1, 5); ax5.axis('off')
table_data = [['Scientific Metric', 'Evaluation Score'],
              ['Model Accuracy (R²)', f"{r2:.3f}"],
              ['Margin of Error (RMSE)', f"{rmse:.2f} units"],
              ['Error Percentage (MAPE)', f"{mape:.2f}%"],
              ['Cluster Quality (Silhouette)', f"{sil_score:.3f}"],
              ['Optimizer Algorithm', opt_info['Algorithm']]]
tbl = ax5.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.3])
tbl.scale(1, 4)

# 8. RENDER
canvas_chart = FigureCanvasTkAgg(fig, frame)
canvas_chart.draw()
canvas_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)

print("main.py running...")
root.mainloop()