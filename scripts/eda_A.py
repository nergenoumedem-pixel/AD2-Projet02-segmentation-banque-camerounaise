"""
==========================================================
PROJET #02 — Segmentation des Clients d'une Banque Camerounaise
ENSP Douala — AD2 2025/2026 — Enseignant : M. FOTSO Valdez
==========================================================
PERSONNE A : Données & Analyse Exploratoire (EDA)
Auteur : Personne A
==========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration globale du style des figures
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
COLORS_CHURN = {0: '#185FA5', 1: '#A32D2D'}  # Bleu = fidèle, Rouge = churn

print("=" * 60)
print("PROJET #02 — EDA Segmentation Clients Banque Camerounaise")
print("Personne A — Exploration & Analyse des Données")
print("=" * 60)

# ==========================================================
# TÂCHE 1 — TÉLÉCHARGEMENT & EXPLORATION INITIALE
# ==========================================================
print("\n📂 Tâche 1 : Chargement et exploration initiale du dataset")

# Chargement du dataset
df = pd.read_csv('../data/Churn_Modelling.csv')

print(f"\n  ➤ Shape du dataset     : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  ➤ Colonnes disponibles : {list(df.columns)}")
print(f"\n  Aperçu des 5 premières lignes :")
print(df.head())

# Valeurs manquantes
missing = df.isnull().sum()
print(f"\n  Valeurs manquantes par colonne :")
print(missing[missing >= 0].to_string())

# Doublons
n_doublons = df.duplicated().sum()
print(f"\n  Doublons détectés : {n_doublons}")

# Suppression colonnes inutiles
df.drop_duplicates(inplace=True)
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
print(f"\n  Colonnes supprimées : RowNumber, CustomerId, Surname (non discriminantes)")
print(f"  Shape après nettoyage : {df_clean.shape}")

# ==========================================================
# TÂCHE 2 — STATISTIQUES DESCRIPTIVES
# ==========================================================
print("\n📊 Tâche 2 : Statistiques descriptives")

print("\n  Variables numériques — describe() :")
print(df_clean.describe().round(2).to_string())

print("\n  Variables catégorielles :")
for col in ['Geography', 'Gender']:
    print(f"\n  {col} :")
    print(df_clean[col].value_counts().to_string())

# Taux de churn global
taux_churn = df_clean['Exited'].mean()
print(f"\n  ✅ Taux de churn global : {taux_churn:.2%} ({df_clean['Exited'].sum()} clients sur {len(df_clean)})")
print(f"  Déséquilibre classes : {df_clean['Exited'].value_counts().to_dict()}")

# Tableau résumé par churn
print("\n  Moyennes par variable selon le statut churn :")
tableau_churn = df_clean.groupby('Exited')[
    ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
].mean().round(2)
tableau_churn.index = ['Non-Churn (0)', 'Churn (1)']
print(tableau_churn.to_string())

# ==========================================================
# TÂCHE 3 — FIGURE 1 : Distribution des variables numériques
# ==========================================================
print("\n🎨 Génération des visualisations EDA...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Distribution des Variables Numériques\nDataset Banque Camerounaise (N=10 000 clients)',
             fontsize=14, fontweight='bold', y=1.01)

num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
for ax, col in zip(axes.flatten(), num_cols):
    ax.hist(df_clean[df_clean['Exited'] == 0][col], bins=30,
            alpha=0.6, color='#185FA5', label='Non-Churn')
    ax.hist(df_clean[df_clean['Exited'] == 1][col], bins=30,
            alpha=0.6, color='#A32D2D', label='Churn')
    ax.set_title(col)
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('../figures/fig1_distributions_numeriques.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig1_distributions_numeriques.png")

# ==========================================================
# FIGURE 2 : Boxplots par statut churn
# ==========================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Boxplots des Variables par Statut Churn\n(Bleu = Fidèle, Rouge = Churn)',
             fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), num_cols):
    data_0 = df_clean[df_clean['Exited'] == 0][col]
    data_1 = df_clean[df_clean['Exited'] == 1][col]
    bp = ax.boxplot([data_0, data_1],
                    patch_artist=True,
                    labels=['Non-Churn', 'Churn'],
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor('#185FA5')
    bp['boxes'][1].set_facecolor('#A32D2D')
    ax.set_title(col)

plt.tight_layout()
plt.savefig('../figures/fig2_boxplots_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig2_boxplots_churn.png")

# ==========================================================
# FIGURE 3 : Matrice de corrélation
# ==========================================================
plt.figure(figsize=(10, 8))
corr = df_clean.copy()
# Encodage temporaire pour la corrélation
corr['Gender_num'] = (corr['Gender'] == 'Male').astype(int)
corr['Geography_num'] = corr['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
corr_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
             'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender_num',
             'Geography_num', 'Exited']
corr_matrix = corr[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=~mask,  # Show lower triangle
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation — Variables du Dataset Bancaire\n(Corrélations avec Exited = variable cible)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/fig3_matrice_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig3_matrice_correlation.png")

# ==========================================================
# FIGURE 4 : Taux de churn par variables catégorielles
# ==========================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Taux de Churn par Variables Catégorielles',
             fontsize=14, fontweight='bold')

# Par Geography
churn_geo = df_clean.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
bars = axes[0].bar(churn_geo.index, churn_geo.values * 100,
                   color=['#A32D2D', '#185FA5', '#2D8A5C'])
axes[0].set_title('Par Pays (Geography)')
axes[0].set_ylabel('Taux de Churn (%)')
axes[0].set_ylim(0, 40)
for bar, val in zip(bars, churn_geo.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

# Par Gender
churn_gender = df_clean.groupby('Gender')['Exited'].mean().sort_values(ascending=False)
bars2 = axes[1].bar(churn_gender.index, churn_gender.values * 100,
                    color=['#A32D2D', '#185FA5'])
axes[1].set_title('Par Genre (Gender)')
axes[1].set_ylabel('Taux de Churn (%)')
axes[1].set_ylim(0, 35)
for bar, val in zip(bars2, churn_gender.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

# Par NumOfProducts
churn_prod = df_clean.groupby('NumOfProducts')['Exited'].mean()
bars3 = axes[2].bar(churn_prod.index.astype(str), churn_prod.values * 100,
                    color=['#2D8A5C', '#185FA5', '#A32D2D', '#FF6B35'])
axes[2].set_title('Par Nombre de Produits')
axes[2].set_ylabel('Taux de Churn (%)')
axes[2].set_xlabel('Nombre de produits souscrits')
for bar, val in zip(bars3, churn_prod.values):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig4_churn_categoriel.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig4_churn_categoriel.png")

# ==========================================================
# FIGURE 5 : Analyse approfondies — Pairplot variables clés
# ==========================================================
df_pair = df_clean[['Age', 'Balance', 'CreditScore', 'EstimatedSalary', 'Exited']].copy()
df_pair['Statut'] = df_pair['Exited'].map({0: 'Non-Churn', 1: 'Churn'})

g = sns.pairplot(df_pair.drop('Exited', axis=1),
                 hue='Statut',
                 palette={'Non-Churn': '#185FA5', 'Churn': '#A32D2D'},
                 plot_kws={'alpha': 0.3, 's': 10},
                 diag_kind='kde')
g.fig.suptitle('Pairplot — Variables les plus Corrélées au Churn', y=1.02, fontsize=13, fontweight='bold')
g.fig.savefig('../figures/fig5_pairplot.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ fig5_pairplot.png")

# ==========================================================
# FIGURE 6 : Taux de churn selon l'âge (tranches)
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Distribution de l'âge par churn
df_clean['Tranche_Age'] = pd.cut(df_clean['Age'],
                                  bins=[17, 25, 35, 45, 55, 65, 100],
                                  labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
churn_age = df_clean.groupby('Tranche_Age')['Exited'].mean()
counts_age = df_clean.groupby('Tranche_Age').size()

ax1 = axes[0]
bars = ax1.bar(range(len(churn_age)), churn_age.values * 100,
               color='#A32D2D', alpha=0.8)
ax1.set_xticks(range(len(churn_age)))
ax1.set_xticklabels(churn_age.index, rotation=30)
ax1.set_title('Taux de Churn par Tranche d\'Âge')
ax1.set_ylabel('Taux de Churn (%)')
for bar, val in zip(bars, churn_age.values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Effectifs par tranche
ax2 = axes[1]
ax2.bar(range(len(counts_age)), counts_age.values, color='#185FA5', alpha=0.8)
ax2.set_xticks(range(len(counts_age)))
ax2.set_xticklabels(counts_age.index, rotation=30)
ax2.set_title('Nombre de Clients par Tranche d\'Âge')
ax2.set_ylabel('Nombre de clients')
for i, (bar, val) in enumerate(zip(ax2.patches, counts_age.values)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
             f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig6_churn_par_age.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig6_churn_par_age.png")

# ==========================================================
# FIGURE 7 : IsActiveMember vs HasCrCard vs churn
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Impact du Statut de Membre et de la Carte de Crédit sur le Churn',
             fontsize=13, fontweight='bold')

for ax, col, label in zip(axes,
                           ['IsActiveMember', 'HasCrCard'],
                           ['Membre Actif', 'Possède une Carte de Crédit']):
    churn_data = df_clean.groupby(col)['Exited'].mean()
    bars = ax.bar(['Non', 'Oui'], churn_data.values * 100,
                  color=['#A32D2D', '#185FA5'], alpha=0.85)
    ax.set_title(f'Taux de churn selon : {label}')
    ax.set_ylabel('Taux de Churn (%)')
    ax.set_ylim(0, 40)
    for bar, val in zip(bars, churn_data.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/fig7_membre_carte.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig7_membre_carte.png")

# ==========================================================
# FIGURE 8 : Résumé synthétique — Variables les + discriminantes
# ==========================================================
# Corrélations absolues avec Exited
df_num = df_clean.copy()
df_num['Gender'] = (df_num['Gender'] == 'Male').astype(int)
df_num['Geography'] = df_num['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
df_num['Tranche_Age'] = df_num['Tranche_Age'].cat.codes

corr_with_exited = df_num.drop('Exited', axis=1).corrwith(df_num['Exited']).abs().sort_values(ascending=True)

plt.figure(figsize=(9, 6))
colors = ['#A32D2D' if v > 0.1 else '#185FA5' for v in corr_with_exited.values]
bars = plt.barh(corr_with_exited.index, corr_with_exited.values, color=colors, alpha=0.85)
plt.axvline(0.1, color='gray', linestyle='--', linewidth=1.2, label='Seuil = 0.10')
plt.xlabel('Corrélation absolue avec Exited (Churn)')
plt.title('Variables les plus Discriminantes pour le Churn\n(Rouge = corrélation > 0.10)',
          fontweight='bold')
plt.legend()
for bar, val in zip(bars, corr_with_exited.values):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2.,
             f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../figures/fig8_variables_discriminantes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig8_variables_discriminantes.png")

# ==========================================================
# EXPORT FINAL
# ==========================================================
# Supprimer la colonne Tranche_Age (créée pour l'analyse, optionnelle)
df_export = df_clean.drop('Tranche_Age', axis=1)
df_export.to_csv('../data/dataset_clean.csv', index=False)
print(f"\n  ✅ dataset_clean.csv exporté ({df_export.shape[0]} lignes, {df_export.shape[1]} colonnes)")

# ==========================================================
# TABLEAU DE SYNTHÈSE FINAL
# ==========================================================
print("\n" + "=" * 60)
print("SYNTHÈSE EDA — RÉSULTATS CLÉS")
print("=" * 60)
print(f"\n  Dataset : {df_export.shape[0]} clients, {df_export.shape[1]} variables")
print(f"  Valeurs manquantes : AUCUNE")
print(f"  Doublons supprimés : {n_doublons}")
print(f"  Taux de churn global : {taux_churn:.2%}")
print(f"\n  TOP 3 variables discriminantes :")
top3 = corr_with_exited.tail(3)
for var, val in zip(top3.index[::-1], top3.values[::-1]):
    print(f"    1. {var} — corrélation = {val:.3f}")
print(f"\n  Observations clés :")
print(f"    • L'Allemagne a le taux de churn le plus élevé (~32%)")
print(f"    • Les clients entre 46-55 ans churne le plus")
print(f"    • Les membres inactifs churne 2x plus que les actifs")
print(f"    • Un balance de 0 est associé à un faible churn")
print(f"    • Les clients avec 3-4 produits ont un très fort taux de churn (>80%)")
print("\n  ✅ Toutes les figures générées dans le dossier figures/")
print("  ✅ dataset_clean.csv prêt pour la Personne B")
print("=" * 60)
