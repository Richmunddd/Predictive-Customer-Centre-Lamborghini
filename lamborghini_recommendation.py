import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import sys

import main


# DATA PREPARATION
df = main.df_clean
unique_prices = sorted(df['Base Price (USD)'].unique())
unique_regions = sorted(df['Region'].unique())
unique_colors = sorted(df['Color'].unique())

# UI 
def update_hp_options(event=None):
    """Limits horsepower options based on the selected budget."""
    try:
        selected_budget = float(budget_var.get())

        valid_cars = df[df['Base Price (USD)'] <= selected_budget]

        available_hp = sorted(valid_cars['Horsepower'].unique())
        
        hp_combo['values'] = available_hp
        if available_hp:
            hp_combo.set(available_hp[0])
        else:
            hp_combo.set("No HP available")
    except ValueError:
        pass

def get_recommendation():
    """Uses ML Trend prediction to recommend the best match."""
    try:
        budget = float(budget_var.get())
        hp = float(hp_var.get())
        region = region_var.get()
        color = color_var.get()
        fuel = fuel_var.get()

        matches = df[(df['Base Price (USD)'] <= budget) & (df['Horsepower'] >= hp)].copy()
        
        if fuel != "Any":
            matches = matches[matches['Fuel Type'] == fuel]
        
        if matches.empty:
            label_result.config(text="No direct match found. Try increasing your budget.", foreground="red")
            return

        recommendations = []
        for model_name in matches['Model'].unique():
            spec_row = matches[matches['Model'] == model_name].iloc[-1]
            
            # Create AI Scenario for 2026
            test_row = pd.DataFrame([{
                'Year': 2026,
                'Model': model_name,
                'Region': region,
                'Base Price (USD)': spec_row['Base Price (USD)'],
                'Horsepower': spec_row['Horsepower'],
                'Cluster': spec_row['Cluster']
            }])
            
            # Predict Trendiness 
            trend_prediction = np.expm1(main.model.predict(test_row[main.features]))[0]
            recommendations.append((model_name, trend_prediction, spec_row))

        # Sort by highest Trend Prediction
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_pick, score, details = recommendations[0]

        res_text = (f"🏆 TOP PICK: {top_pick}\n\n"
                    f"MARKET TREND: High (Top choice for {region} in 2026)\n"
                    f"SPECS: {int(details['Horsepower'])} HP | ${int(details['Base Price (USD)']):,}\n"
                    f"COLOR PREFERENCE: Available in {color}\n"
                    f"ENGINE: {details['Fuel Type']}")
        label_result.config(text=res_text, foreground="#D4AF37")

    except Exception as e:
        messagebox.showerror("Selection Error", "Please ensure all fields are selected.")

# ---------------------------------------------------------
root = tk.Tk()
root.title("Lamborghini AI Consultant")
root.geometry("500x750")
root.configure(bg="#000000")

# Header
tk.Label(root, text="LAMBORGHINI", font=("Impact", 30), bg="#000000", fg="#D4AF37").pack(pady=(20,0))
tk.Label(root, text="RECOMMENDATION SYSTEM", font=("Arial", 10, "bold"), bg="#000000", fg="white").pack()

# Input Container
frame = tk.Frame(root, bg="#000000", padx=40)
frame.pack(fill="both", expand=True, pady=20)

# Budget 
tk.Label(frame, text="Select Max Budget (USD):", bg="#000000", fg="#AAAAAA").pack(anchor="w")
budget_var = tk.StringVar()
budget_combo = ttk.Combobox(frame, textvariable=budget_var, values=unique_prices, state="readonly")
budget_combo.pack(fill="x", pady=5)
budget_combo.bind("<<ComboboxSelected>>", update_hp_options)

# Horsepower
tk.Label(frame, text="Min Horsepower (Filtered by Budget):", bg="#000000", fg="#AAAAAA").pack(anchor="w", pady=(10,0))
hp_var = tk.StringVar()
hp_combo = ttk.Combobox(frame, textvariable=hp_var, state="readonly")
hp_combo.pack(fill="x", pady=5)

# Region
tk.Label(frame, text="Your Region:", bg="#000000", fg="#AAAAAA").pack(anchor="w", pady=(10,0))
region_var = tk.StringVar()
region_combo = ttk.Combobox(frame, textvariable=region_var, values=unique_regions, state="readonly")
region_combo.set(unique_regions[0])
region_combo.pack(fill="x", pady=5)

# Preferred Color
tk.Label(frame, text="Preferred Color:", bg="#000000", fg="#AAAAAA").pack(anchor="w", pady=(10,0))
color_var = tk.StringVar()
color_combo = ttk.Combobox(frame, textvariable=color_var, values=unique_colors, state="readonly")
color_combo.set(unique_colors[0])
color_combo.pack(fill="x", pady=5)

# Fuel Type
tk.Label(frame, text="Engine Preference:", bg="#000000", fg="#AAAAAA").pack(anchor="w", pady=(10,0))
fuel_var = tk.StringVar()
fuel_combo = ttk.Combobox(frame, textvariable=fuel_var, values=["Any", "Gasoline", "Hybrid"], state="readonly")
fuel_combo.set("Any")
fuel_combo.pack(fill="x", pady=5)

#  Button
btn = tk.Button(root, text="FIND MATCH", command=get_recommendation, 
                bg="#D4AF37", fg="black", font=("Arial", 12, "bold"), height=2, cursor="hand2")
btn.pack(pady=20, padx=40, fill="x")

# Result Display
label_result = tk.Label(root, text="Select your criteria to begin...", bg="#000000", 
                       fg="#888888", font=("Arial", 11, "italic"), wraplength=400)
label_result.pack(pady=20)

root.mainloop()