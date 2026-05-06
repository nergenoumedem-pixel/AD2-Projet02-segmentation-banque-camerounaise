"""
==========================================================
PROJET #02 — Segmentation des Clients d'une Banque Camerounaise
ENSP Douala — AD2 2025/2026 — Enseignant : M. FOTSO Valdez
==========================================================
PERSONNE B : Prétraitement, Feature Engineering & AFD
Fichier   : preprocessing_features_B.py
Entrée    : data/dataset_clean.csv  (livré par Personne A)
Sorties   : data/X_preprocessed.csv
            data/X_resampled.npy
            data/y_resampled.npy
            data/y_labels.npy
            models/scaler.pkl
            models/le_geo.pkl
            models/le_gen.pkl
            models/lda_model.pkl
            figures/fig_B1_smote_equilibre.png
            figures/fig_B2_afd_plan_discriminant.png
            figures/fig_B3_afd_coefficients.png
            figures/fig_B4_features_engineering.png
==========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Création des dossiers nécessaires ─────────────────────
os.makedirs('data',    exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('figures', exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 13
COLORS = {'churn': '#A32D2D', 'non_churn': '#185FA5'}

print("=" * 60)
print("PROJET #02 — Prétraitement, Feature Engineering & AFD")
print("Personne B")
print("=" * 60)

# ==========================================================
# CHARGEMENT DU DATASET PROPRE (livré par A)
# ==========================================================
print("\n📂 Chargement du dataset_clean.csv de la Personne A...")
df = pd.read_csv('data/dataset_clean.csv')
print(f"  ➤ Shape chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  ➤ Colonnes     : {list(df.columns)}")
print(f"  ➤ Variable cible 'Exited' : {df['Exited'].value_counts().to_dict()}")

# ==========================================================
# ÉTAPE 1 — ENCODAGE DES VARIABLES CATÉGORIELLES
# ==========================================================
print("\n🔧 Étape 1 : Encodage des variables catégorielles")

from sklearn.preprocessing import LabelEncoder

df_enc = df.copy()

# LabelEncoder sur Geography et Gender
le_geo = LabelEncoder()
le_gen = LabelEncoder()

df_enc['Geography'] = le_geo.fit_transform(df_enc['Geography'])
df_enc['Gender']    = le_gen.fit_transform(df_enc['Gender'])

# Sauvegarder les encodeurs pour la démo Streamlit
joblib.dump(le_geo, 'models/le_geo.pkl')
joblib.dump(le_gen, 'models/le_gen.pkl')

print(f"  ➤ Geography encodé : {dict(zip(le_geo.classes_, le_geo.transform(le_geo.classes_)))}")
print(f"  ➤ Gender encodé    : {dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_)))}")
print("  ✅ Encodeurs sauvegardés : le_geo.pkl, le_gen.pkl")

# ==========================================================
# ÉTAPE 2 — FEATURE ENGINEERING
# ==========================================================
print("\n⚙️  Étape 2 : Feature Engineering — Création de nouvelles variables")

# Nouvelles variables créées AVANT la standardisation
df_enc['balance_par_produit']   = df_enc['Balance'] / (df_enc['NumOfProducts'] + 1)
df_enc['ratio_salaire_balance'] = df_enc['Balance'] / (df_enc['EstimatedSalary'] + 1)
df_enc['anciennete_par_age']    = df_enc['Tenure'] / df_enc['Age']
df_enc['est_inactif_riche']     = (
    (df_enc['IsActiveMember'] == 0) & (df_enc['Balance'] > 100000)
).astype(int)
df_enc['client_premium']        = (
    (df_enc['CreditScore'] > 750) & (df_enc['Balance'] > 150000)
).astype(int)

nouvelles_vars = [
    'balance_par_produit',
    'ratio_salaire_balance',
    'anciennete_par_age',
    'est_inactif_riche',
    'client_premium'
]
print(f"  ➤ 5 nouvelles variables créées : {nouvelles_vars}")
print(f"  ➤ Clients inactifs riches détectés : {df_enc['est_inactif_riche'].sum()}")
print(f"  ➤ Clients premium détectés         : {df_enc['client_premium'].sum()}")

# Vérification : taux de churn par segment créé
for var in ['est_inactif_riche', 'client_premium']:
    taux = df_enc.groupby(var)['Exited'].mean()
    print(f"  ➤ Taux churn par {var} : {taux.to_dict()}")

# ── Figure B4 : Impact des nouvelles variables sur le churn ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Feature Engineering — Impact sur le Churn",
             fontsize=14, fontweight='bold')

# balance_par_produit distribution
for label, color in [(0, COLORS['non_churn']), (1, COLORS['churn'])]:
    axes[0].hist(df_enc[df_enc['Exited']==label]['balance_par_produit'],
                 bins=40, alpha=0.6, color=color,
                 label='Non-Churn' if label==0 else 'Churn')
axes[0].set_title('Balance par Produit')
axes[0].set_xlabel('Balance / NumOfProducts')
axes[0].legend()

# ratio_salaire_balance
for label, color in [(0, COLORS['non_churn']), (1, COLORS['churn'])]:
    axes[1].hist(df_enc[df_enc['Exited']==label]['ratio_salaire_balance'],
                 bins=40, alpha=0.6, color=color,
                 label='Non-Churn' if label==0 else 'Churn')
axes[1].set_title('Ratio Salaire / Balance')
axes[1].set_xlabel('Balance / EstimatedSalary')
axes[1].legend()

# est_inactif_riche et client_premium
cats = ['est_inactif_riche', 'client_premium']
churn_rates = [df_enc[df_enc[c]==1]['Exited'].mean()*100 for c in cats]
non_churn_rates = [df_enc[df_enc[c]==0]['Exited'].mean()*100 for c in cats]
x = range(len(cats))
bars1 = axes[2].bar([i-0.2 for i in x], non_churn_rates, 0.35,
                    color=COLORS['non_churn'], alpha=0.85, label='Flag = 0')
bars2 = axes[2].bar([i+0.2 for i in x], churn_rates, 0.35,
                    color=COLORS['churn'], alpha=0.85, label='Flag = 1')
axes[2].set_xticks(list(x))
axes[2].set_xticklabels(['Inactif\nRiche', 'Client\nPremium'], fontsize=10)
axes[2].set_ylabel('Taux de Churn (%)')
axes[2].set_title('Taux Churn par Segment Créé')
axes[2].legend()
for bar, val in zip(bars1, non_churn_rates):
    axes[2].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, churn_rates):
    axes[2].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig_B4_features_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_B4_features_engineering.png")

# ==========================================================
# ÉTAPE 3 — SÉPARATION X / y ET STANDARDISATION
# ==========================================================
print("\n📐 Étape 3 : Séparation X / y et Standardisation")

from sklearn.preprocessing import StandardScaler

# Séparation
X = df_enc.drop('Exited', axis=1)
y = df_enc['Exited'].values
feature_names = X.columns.tolist()

print(f"  ➤ Variables X ({len(feature_names)}) : {feature_names}")
print(f"  ➤ Variable cible y : {pd.Series(y).value_counts().to_dict()}")

# Standardisation (APRÈS feature engineering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Vérification
print(f"  ➤ Moyenne après scaling  : {X_scaled.mean():.6f} (doit être ≈ 0)")
print(f"  ➤ Std après scaling      : {X_scaled.std():.6f}  (doit être ≈ 1)")

# Sauvegardes
joblib.dump(scaler, 'models/scaler.pkl')
np.save('data/y_labels.npy', y)
pd.DataFrame(X_scaled, columns=feature_names).to_csv('data/X_preprocessed.csv', index=False)

print("  ✅ scaler.pkl sauvegardé")
print("  ✅ X_preprocessed.csv exporté")
print("  ✅ y_labels.npy exporté")

# ==========================================================
# ÉTAPE 4 — GESTION DU DÉSÉQUILIBRE DES CLASSES (SMOTE)
# ==========================================================
print("\n⚖️  Étape 4 : Gestion du déséquilibre — SMOTE")

from imblearn.over_sampling import SMOTE

print(f"  Avant SMOTE : {pd.Series(y).value_counts().to_dict()}")
print(f"  Ratio Churn/Non-Churn : {y.sum()}/{len(y)-y.sum()} = {y.mean():.2%}")

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

print(f"  Après SMOTE : {pd.Series(y_res).value_counts().to_dict()}")
print(f"  Nouvelles observations synthétiques créées : {len(X_res) - len(X_scaled)}")

# Sauvegardes SMOTE
np.save('data/X_resampled.npy', X_res)
np.save('data/y_resampled.npy', y_res)
print("  ✅ X_resampled.npy exporté")
print("  ✅ y_resampled.npy exporté")

# ── Figure B1 : Avant / Après SMOTE ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Gestion du Déséquilibre des Classes — SMOTE",
             fontsize=13, fontweight='bold')

labels_plot = ['Non-Churn (0)', 'Churn (1)']
avant = [len(y) - y.sum(), y.sum()]
apres = [len(y_res) - y_res.sum(), y_res.sum()]

axes[0].bar(labels_plot, avant, color=[COLORS['non_churn'], COLORS['churn']], alpha=0.85)
axes[0].set_title('Avant SMOTE')
axes[0].set_ylabel("Nombre de clients")
for i, v in enumerate(avant):
    axes[0].text(i, v + 50, f'{v}\n({v/sum(avant):.1%})',
                 ha='center', va='bottom', fontweight='bold')

axes[1].bar(labels_plot, apres, color=[COLORS['non_churn'], COLORS['churn']], alpha=0.85)
axes[1].set_title('Après SMOTE (classes équilibrées)')
axes[1].set_ylabel("Nombre de clients")
for i, v in enumerate(apres):
    axes[1].text(i, v + 50, f'{v}\n({v/sum(apres):.1%})',
                 ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig_B1_smote_equilibre.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_B1_smote_equilibre.png")

# ==========================================================
# ÉTAPE 5 — AFD : ANALYSE FACTORIELLE DISCRIMINANTE
# ==========================================================
print("\n📊 Étape 5 : AFD — Analyse Factorielle Discriminante")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# On applique LDA sur X_scaled (données originales, non-SMOTE)
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

print(f"  ➤ Ratio Fisher (séparabilité) : {lda.explained_variance_ratio_}")
print(f"  ➤ Variance expliquée par l'axe discriminant : {lda.explained_variance_ratio_[0]:.2%}")

# Sauvegarder le modèle LDA
joblib.dump(lda, 'models/lda_model.pkl')
print("  ✅ lda_model.pkl sauvegardé")

# Statistiques sur l'axe discriminant
scores_non_churn = X_lda[y == 0].flatten()
scores_churn     = X_lda[y == 1].flatten()
print(f"\n  Scores discriminants — Non-Churn : μ={scores_non_churn.mean():.3f}, σ={scores_non_churn.std():.3f}")
print(f"  Scores discriminants — Churn     : μ={scores_churn.mean():.3f}, σ={scores_churn.std():.3f}")
print(f"  Séparation (diff des moyennes)   : {abs(scores_churn.mean() - scores_non_churn.mean()):.3f}")

# ── Figure B2 : Plan discriminant ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("AFD — Plan Discriminant Churn / Non-Churn",
             fontsize=14, fontweight='bold')

# Histogramme des scores discriminants
axes[0].hist(scores_non_churn, bins=60, alpha=0.6,
             color=COLORS['non_churn'], label=f'Non-Churn (μ={scores_non_churn.mean():.2f})', density=True)
axes[0].hist(scores_churn, bins=60, alpha=0.6,
             color=COLORS['churn'], label=f'Churn (μ={scores_churn.mean():.2f})', density=True)
axes[0].axvline(scores_non_churn.mean(), color=COLORS['non_churn'],
                linestyle='--', linewidth=2, label='Moyenne Non-Churn')
axes[0].axvline(scores_churn.mean(), color=COLORS['churn'],
                linestyle='--', linewidth=2, label='Moyenne Churn')
axes[0].set_xlabel("Score Discriminant LDA")
axes[0].set_ylabel("Densité")
axes[0].set_title("Distribution sur l'Axe Discriminant")
axes[0].legend(fontsize=8)

# Scatter plot : 200 points de chaque classe pour la lisibilité
np.random.seed(42)
idx_0 = np.random.choice(np.where(y==0)[0], size=300, replace=False)
idx_1 = np.random.choice(np.where(y==1)[0], size=300, replace=False)
scatter_x0 = X_lda[idx_0, 0]
scatter_x1 = X_lda[idx_1, 0]
jitter0 = np.random.normal(0, 0.05, len(idx_0))
jitter1 = np.random.normal(0, 0.05, len(idx_1))
axes[1].scatter(scatter_x0, jitter0, alpha=0.4, s=12,
                color=COLORS['non_churn'], label='Non-Churn')
axes[1].scatter(scatter_x1, jitter1, alpha=0.4, s=12,
                color=COLORS['churn'], label='Churn')
axes[1].axvline(0, color='gray', linestyle='--', linewidth=1.5, label='Frontière LDA')
axes[1].set_xlabel("Score Discriminant LDA")
axes[1].set_yticks([])
axes[1].set_title("Nuage de Points — Séparation des Groupes")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig_B2_afd_plan_discriminant.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_B2_afd_plan_discriminant.png")

# ── Figure B3 : Coefficients discriminants ───────────────
coef_df = pd.DataFrame({
    'Variable': feature_names,
    'Coefficient_abs': np.abs(lda.coef_[0]),
    'Coefficient'    : lda.coef_[0]
}).sort_values('Coefficient_abs', ascending=True)

plt.figure(figsize=(9, 7))
colors_bar = [COLORS['churn'] if c > 0 else COLORS['non_churn']
              for c in coef_df['Coefficient'].values]
bars = plt.barh(coef_df['Variable'], coef_df['Coefficient_abs'],
                color=colors_bar, alpha=0.85)
plt.axvline(0, color='gray', linewidth=0.5)
for bar, val in zip(bars, coef_df['Coefficient_abs'].values):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2.,
             f'{val:.3f}', va='center', fontsize=8.5)
plt.xlabel("Coefficient Discriminant LDA (valeur absolue)")
plt.title("Variables les plus Discriminantes — AFD\n(Rouge = coefficient positif, Bleu = négatif)",
          fontweight='bold')
legend_elem = [
    mpatches.Patch(color=COLORS['churn'],     label='Favorise le Churn (+)'),
    mpatches.Patch(color=COLORS['non_churn'], label='Favorise la Fidélité (−)'),
]
plt.legend(handles=legend_elem, fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig_B3_afd_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_B3_afd_coefficients.png")

# ==========================================================
# SYNTHÈSE FINALE
# ==========================================================
print("\n" + "=" * 60)
print("SYNTHÈSE — PERSONNE B")
print("=" * 60)
print(f"\n  Dataset reçu de A  : {df.shape[0]} clients × {df.shape[1]} variables")
print(f"  Après encoding     : Geography et Gender encodés")
print(f"  Après feat. eng.   : +5 nouvelles variables → {len(feature_names)} variables au total")
print(f"  Après SMOTE        : {len(X_res)} observations ({pd.Series(y_res).value_counts().to_dict()})")
print(f"\n  TOP 3 variables discriminantes (AFD) :")
top3_afd = coef_df.tail(3)
for i, (_, row) in enumerate(top3_afd.iterrows(), 1):
    print(f"    {i}. {row['Variable']} — coeff = {row['Coefficient_abs']:.4f}")
print(f"\n  Fichiers exportés pour C et D :")
print(f"    ✅ data/X_preprocessed.csv")
print(f"    ✅ data/X_resampled.npy")
print(f"    ✅ data/y_resampled.npy")
print(f"    ✅ data/y_labels.npy")
print(f"    ✅ models/scaler.pkl")
print(f"    ✅ models/le_geo.pkl, le_gen.pkl, lda_model.pkl")
print(f"\n  Figures générées :")
print(f"    ✅ fig_B1_smote_equilibre.png")
print(f"    ✅ fig_B2_afd_plan_discriminant.png")
print(f"    ✅ fig_B3_afd_coefficients.png")
print(f"    ✅ fig_B4_features_engineering.png")
print("\n  ✅ Personne B — Travail terminé. X_preprocessed.csv prêt pour C et D.")
print("=" * 60)
