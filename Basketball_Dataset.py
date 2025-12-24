import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#Load Data
file_path = "NBA_Dataset.csv"
df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

metric_cols = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%']

def clean(x):
    if pd.isna(x): return 0
    if isinstance(x, str):
        x = x.replace('%', '').replace(',', '')
        return float(x) if x.replace('.', '').isdigit() else 0
    return float(x)

for col in metric_cols:
    df[col] = df[col].apply(clean)

# Convert FG%
if df['FG%'].max() <= 1:
    df['FG%'] *= 100

norm = MinMaxScaler()
df_norm = pd.DataFrame(norm.fit_transform(df[metric_cols]), columns=metric_cols)
df['performance_score'] = df_norm.mean(axis=1) * 100

q1, q2 = df['performance_score'].quantile([0.33, 0.66])
df['rating_class'] = df['performance_score'].apply(
    lambda x: 'Low' if x <= q1 else ('Medium' if x <= q2 else 'High')
)

X = df[metric_cols]
y = df['rating_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_s, y_train)

df['predicted_class'] = svm.predict(sc.transform(X))

df['predicted_score'] = df['predicted_class'].map({'Low':20, 'Medium':60, 'High':100})
df['final_score'] = 0.7*df['performance_score'] + 0.3*df['predicted_score']

print("\nSVM CLASSIFICATION REPORT:")
y_pred = svm.predict(X_test_s)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX:")
print(cm)

print(f"\nACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.show()

def plot_nba_distributions():
    for col in metric_cols:
        plt.figure(figsize=(10,5))
        sns.histplot(df[col], kde=True, bins=20)

        avg = df[col].mean()
        max_val = df[col].max()

        sorted_vals = df[col].sort_values(ascending=False).values
        if len(sorted_vals) >= 10:
            top10_cut = sorted_vals[9]
        else:
            top10_cut = sorted_vals[-1]

        plt.axvline(avg, color='blue', linestyle='--', linewidth=2,
                    label=f"Average = {avg:.2f}")

        plt.axvline(top10_cut, color='red', linestyle='--', linewidth=2,
                    label=f"Top 10 Cutoff = {top10_cut:.2f}")

        plt.axvline(max_val, color='green', linestyle='--', linewidth=2,
                    label=f"Best Player (Max) = {max_val:.2f}")

        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("\n Generating NBA parameter distribution graphs...")
plot_nba_distributions()

def pos_map(p):
    p = str(p).upper()
    if any(x in p for x in ['PG','SG','G']): return 'Guard'
    if any(x in p for x in ['SF','PF','F']): return 'Forward'
    if 'C' in p: return 'Center'
    return 'Unknown'

df['Group'] = df['Pos'].apply(pos_map)
df['Player Name'] = df['Player']

def animate_top10(group, title):
    sub = df if group == 'All' else df[df['Group']==group]
    top10 = sub.sort_values('final_score', ascending=False).head(10)

    if top10.empty:
        print(f"No players found: {group}")
        return

    top10 = top10.sort_values('final_score')

    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(top10['Player Name'], [0]*len(top10))

    ax.set_xlim(0, 100)
    ax.set_title(f"Top 10 {title}")

    def update(frame):
        ratio = frame/100
        for bar, score in zip(bars, top10['final_score']):
            bar.set_width(score * ratio)
        return bars

    ani = animation.FuncAnimation(
        fig, update,
        frames=np.linspace(0,100,60),
        interval=25,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()
    return ani

animate_top10('Guard', "Guards")
animate_top10('Forward', "Forwards")
animate_top10('Center', "Centers")
animate_top10('All', "Overall Players")


print("TOP 10 GUARDS")
print(df[df['Group']=='Guard'].sort_values('final_score',ascending=False).head(10)[[
    'Player Name','final_score','performance_score','predicted_class'
]])

print("TOP 10 FORWARDS")
print(df[df['Group']=='Forward'].sort_values('final_score',ascending=False).head(10)[[
    'Player Name','final_score','performance_score','predicted_class'
]])

print("TOP 10 CENTERS")
print(df[df['Group']=='Center'].sort_values('final_score',ascending=False).head(10)[[
    'Player Name','final_score','performance_score','predicted_class'
]])

print("TOP 10 OVERALL NBA PLAYERS")
print(df.sort_values('final_score', ascending=False).head(10)[[
    'Player Name','Group','final_score','performance_score','predicted_class'
]])
