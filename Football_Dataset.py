import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load Data
file_path = "FIFA_Dataset.xlsx"
df = pd.read_excel(file_path)
df.columns = [c.strip() for c in df.columns]

metric_cols = [
    'Goals Scored', 'Assists Provided', 'Dribbles per 90',
    'Interceptions per 90', 'Tackles per 90', 'Total Duels Won per 90'
]

def clean_numeric(x):
    if pd.isna(x): return 0.0
    if isinstance(x, str):
        x = x.strip()
        if x in ['-', 'N.A', 'NA', 'nan']:
            return 0.0
        try:
            return float(x.replace(',', ''))
        except:
            return 0.0
    return float(x)

for col in metric_cols:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].apply(clean_numeric)

df[metric_cols] = df[metric_cols].fillna(0)

scaler_norm = MinMaxScaler()
df_norm = pd.DataFrame(scaler_norm.fit_transform(df[metric_cols]), columns=metric_cols)

df['performance_score'] = df_norm.mean(axis=1) * 100

# Labeling based on performance score
q33, q66 = df['performance_score'].quantile([0.33, 0.66])

df['rating_class'] = df['performance_score'].apply(
    lambda x: 'Low' if x <= q33 else ('Medium' if x <= q66 else 'High')
)

X = df[metric_cols]
y = df['rating_class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm_clf = SVC(kernel='rbf', C=10, gamma='scale')
svm_clf.fit(X_scaled, y)

df['predicted_class'] = svm_clf.predict(X_scaled)

df['predicted_score'] = df['predicted_class'].map({
    'Low': 0,
    'Medium': 50,
    'High': 100
})

df['final_score'] = 0.5 * df['performance_score'] + 0.5 * df['predicted_score']

print("\nSVM CLASSIFICATION REPORT:")
print(classification_report(y, df['predicted_class']))

cm = confusion_matrix(y, df['predicted_class'])
acc = accuracy_score(y, df['predicted_class'])

print("\nCONFUSION MATRIX:")
print(cm)
print(f"\nACCURACY: {acc*100:.2f}%")
print(f"ERROR RATE: {(1-acc)*100:.2f}%")

# Plot Confusion Matrix
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

def plot_parameter_distributions():
    for col in metric_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=20)

        # BLUE LINE = Mean
        avg = df[col].mean()

        # GREEN LINE = Max
        max_val = df[col].max()

        # RED LINE = 10th highest value
        sorted_vals = df[col].sort_values(ascending=False).values
        if len(sorted_vals) >= 10:
            top10_cutoff = sorted_vals[9]
        else:
            top10_cutoff = sorted_vals[-1]

        plt.axvline(avg, color='blue', linestyle='--', linewidth=2,
                    label=f"Average (Mean) = {avg:.2f}")

        plt.axvline(top10_cutoff, color='red', linestyle='--', linewidth=2,
                    label=f"Top 10 Cutoff = {top10_cutoff:.2f}")

        plt.axvline(max_val, color='green', linestyle='--', linewidth=2,
                    label=f"Best Player (Max) = {max_val:.2f}")

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("\nGenerating parameter distribution graphs...")
plot_parameter_distributions()

def animate_top10(position_code, title_name):
    if position_code == 'All':
        top10 = df.sort_values('final_score', ascending=False).head(10)
    else:
        top10 = df[df['Position'] == position_code].sort_values('final_score', ascending=False).head(10)

    if top10.empty:
        print(f"No players found for {title_name}")
        return

    top10 = top10.sort_values('final_score')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10['Player Name'], [0]*len(top10),
                   color=sns.color_palette("coolwarm", len(top10)))

    ax.set_xlim(0, 100)
    ax.set_xlabel("Final Combined Score")
    ax.set_title(f"Top 10 {title_name} (Simulated)")

    def update(frame):
        for bar, score in zip(bars, top10['final_score']):
            bar.set_width(score * (frame / 100))
        return bars

    ani = animation.FuncAnimation(
        fig, update, frames=np.linspace(0, 100, 60),
        interval=25, blit=False, repeat=False
    )

    plt.tight_layout()
    plt.show()

animate_top10('FW', 'Forwards')
animate_top10('MF', 'Midfielders')
animate_top10('DF', 'Defenders')
animate_top10('GK', 'Goalkeepers')
animate_top10('All', 'Overall Players')

print("TOP 10 FORWARDS (FW)")
print(df[df['Position']=='FW'].sort_values('final_score', ascending=False).head(10)[[
    'Player Name', 'final_score', 'performance_score', 'predicted_class'
]])

print("TOP 10 MIDFIELDERS (MF)")
print(df[df['Position']=='MF'].sort_values('final_score', ascending=False).head(10)[[
    'Player Name', 'final_score', 'performance_score', 'predicted_class'
]])

print("TOP 10 DEFENDERS (DF)")
print(df[df['Position']=='DF'].sort_values('final_score', ascending=False).head(10)[[
    'Player Name', 'final_score', 'performance_score', 'predicted_class'
]])

print("TOP 10 GOALKEEPERS (GK)")
print(df[df['Position']=='GK'].sort_values('final_score', ascending=False).head(10)[[
    'Player Name', 'final_score', 'performance_score', 'predicted_class'
]])

print("TOP 10 OVERALL PLAYERS")
print(df.sort_values('final_score', ascending=False).head(10)[[
    'Player Name', 'Position', 'final_score', 'performance_score', 'predicted_class'
]])

