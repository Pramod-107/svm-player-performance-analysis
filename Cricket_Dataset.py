import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load Data
file_path = "IPL_Dataset.csv"
df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

batting_cols = ['Runs_Scored', 'Batting_Average', 'Batting_Strike_Rate']
bowling_cols = ['Wickets_Taken', 'Economy_Rate', 'Bowling_Strike_Rate']
keeping_cols = ['Stumpings']

all_metrics = batting_cols + bowling_cols + keeping_cols


def clean(x):
    if pd.isna(x): return 0
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        try:
            return float(x)
        except:
            return 0
    return float(x)


for col in all_metrics:
    df[col] = df[col].apply(clean)

df['Economy_Rate_inv'] = df['Economy_Rate'].max() - df['Economy_Rate']
df['Bowling_Strike_Rate_inv'] = df['Bowling_Strike_Rate'].max() - df['Bowling_Strike_Rate']

overall_metric_cols = [
    'Runs_Scored', 'Batting_Average', 'Batting_Strike_Rate',
    'Wickets_Taken', 'Economy_Rate_inv', 'Bowling_Strike_Rate_inv'
]

norm = MinMaxScaler()
df_norm = pd.DataFrame(norm.fit_transform(df[overall_metric_cols]), columns=overall_metric_cols)
df['performance_score'] = df_norm.mean(axis=1) * 100

q1, q2 = df['performance_score'].quantile([0.33, 0.66])
df['rating_class'] = df['performance_score'].apply(
    lambda x: 'Low' if x <= q1 else ('Medium' if x <= q2 else 'High')
)

X = df[overall_metric_cols]
y = df['rating_class']

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_scaled, y)

df['predicted_class'] = svm.predict(X_scaled)
df['predicted_score'] = df['predicted_class'].map({'Low': 20, 'Medium': 60, 'High': 100})

df['final_score'] = 0.7 * df['performance_score'] + 0.3 * df['predicted_score']

# Batting score
bat_norm = MinMaxScaler()
df_bat = pd.DataFrame(bat_norm.fit_transform(df[batting_cols]), columns=batting_cols)
df['batting_score'] = df_bat.mean(axis=1) * 100

# Bowling score
bowl_norm = MinMaxScaler()
df_bowl = pd.DataFrame(bowl_norm.fit_transform(df[['Wickets_Taken', 'Economy_Rate_inv', 'Bowling_Strike_Rate_inv']]),
                       columns=['Wickets_Taken', 'Economy_Rate_inv', 'Bowling_Strike_Rate_inv'])
df['bowling_score'] = df_bowl.mean(axis=1) * 100


df['keeping_score'] = df['Stumpings']

def detect_role(row):
    if row['Stumpings'] > 0:
        return "Wicketkeeper"

    if row['batting_score'] > row['bowling_score']:
        return "Batsman"

    if row['bowling_score'] > row['batting_score']:
        return "Bowler"

    return "All-Rounder"


df['Role'] = df.apply(detect_role, axis=1)

df['Player_Name'] = df['Player_Name']

print("\nSVM CLASSIFICATION REPORT:")
print(classification_report(y, df['predicted_class']))

cm = confusion_matrix(y, df['predicted_class'])
print("\nCONFUSION MATRIX:")
print(cm)
print(f"\nACCURACY: {accuracy_score(y, df['predicted_class']) * 100:.2f}%")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - IPL Overall")
plt.show()

def plot_distributions():
    for col in overall_metric_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=20)

        avg = df[col].mean()
        max_val = df[col].max()

        sorted_vals = df[col].sort_values(ascending=False).values
        top10_cutoff = sorted_vals[9] if len(sorted_vals) >= 10 else sorted_vals[-1]

        plt.axvline(avg, color='blue', linestyle='--', linewidth=2, label=f"Average = {avg:.2f}")
        plt.axvline(top10_cutoff, color='red', linestyle='--', linewidth=2, label=f"Top 10 Cutoff = {top10_cutoff:.2f}")
        plt.axvline(max_val, color='green', linestyle='--', linewidth=2, label=f"Best Player (Max) = {max_val:.2f}")

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()


print("\nGenerating metric distribution graphs...")
plot_distributions()

def animate_top10(scores_col, title):
    top10 = df.sort_values(scores_col, ascending=False).head(10)

    top10 = top10.sort_values(scores_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10['Player_Name'], [0] * len(top10))

    ax.set_xlim(0, 100)
    ax.set_title(f"Top 10 {title}")

    def update(frame):
        for bar, score in zip(bars, top10[scores_col]):
            bar.set_width(score * (frame / 100))
        return bars

    ani = animation.FuncAnimation(
        fig, update, frames=np.linspace(0, 100, 60), interval=25, repeat=False
    )

    plt.tight_layout()
    plt.show()
    return ani

animate_top10("batting_score", "Batsmen")
animate_top10("bowling_score", "Bowlers")
animate_top10("keeping_score", "Wicketkeepers")
animate_top10("final_score", "Overall Players")

def print_top(scores_col, title):
    print(f"TOP 10 {title}")
    print(df.sort_values(scores_col, ascending=False).head(10)[[
        'Player_Name', 'Role', scores_col, 'performance_score', 'predicted_class'
    ]])


print_top("batting_score", "BATSMEN")
print_top("bowling_score", "BOWLERS")
print_top("keeping_score", "WICKETKEEPERS")
print_top("final_score", "OVERALL PLAYERS")
