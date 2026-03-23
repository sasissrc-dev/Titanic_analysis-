import pandas as pd
import matplotlib.pyplot as plt

# Load & Overview
df = pd.read_csv("1c02c00c-cab2-4667-ae9a-b4030f7df087.csv")
  print("Shape:", df.shape, "\n\nMissing Values:\n", df.isnull().sum(), "\n\nBasic Stats:\n", df.describe())

# Clean
df["Age"] = df["Age"].fillna(df["Age"].median())
df.drop(columns=["Cabin"], inplace=True)
df.dropna(inplace=True)

# Feature Engineering
df["AgeGroup"] = pd.cut(df["Age"], bins=[0,12,18,35,60,100], labels=["Child","Teen","Adult","Middle-Age","Senior"])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Analysis
sr = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {sr:.2f}%")
print("\nBy Gender (%):\n",      df.groupby("Sex")["Survived"].mean() * 100)
print("\nBy Class (%):\n",       df.groupby("Pclass")["Survived"].mean() * 100)
print("\nBy Age Group (%):\n",   df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100)
print("\nAvg Fare by Class:\n",  df.groupby("Pclass")["Fare"].mean())
print("\nBy Port (%):\n",        df.groupby("Embarked")["Survived"].mean() * 100)
print("\nBy Family Size (%):\n", df.groupby("FamilySize")["Survived"].mean() * 100)
print("\nSummary:\n", pd.DataFrame({"Total":[len(df)],"Survived":[df['Survived'].sum()],"Rate%":[round(sr,2)],"AvgAge":[round(df['Age'].mean(),1)],"AvgFare":[round(df['Fare'].mean(),2)]}).to_string(index=False))

# Graphs
B,M,L,R,G,BG = "#1F3864","#2E75B6","#A8C8E8","#C0392B","#27AE60","#F8FBFE"
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor(BG); fig.suptitle("Titanic Passenger Data Analysis", fontsize=18, fontweight='bold', color=B, y=1.01)

def bar(a, x, y, cols, title, ylabel="Survival Rate (%)", ylim=90, prefix="", suffix="%"):
    a.set_facecolor(BG); bars = a.bar(x, y, color=cols, width=0.5, edgecolor='white', linewidth=1.5)
    [a.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{prefix}{b.get_height():.1f}{suffix}", ha='center', fontsize=10, fontweight='bold', color=B) for b in bars]
    a.set_ylim(0, ylim); a.set_ylabel(ylabel, color=B, fontsize=9); a.set_title(title, fontsize=11, fontweight='bold', color=B)
    a.spines[['top','right']].set_visible(False); a.yaxis.grid(True, linestyle='--', alpha=0.5); a.set_axisbelow(True)

v = [df["Survived"].sum(), (df["Survived"]==0).sum()]
w, _ = ax[0,0].pie(v, colors=[G,R], startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2))
ax[0,0].legend(w, [f"Survived\n{v[0]} ({v[0]/sum(v)*100:.1f}%)", f"Did Not Survive\n{v[1]} ({v[1]/sum(v)*100:.1f}%)"], loc="lower center", fontsize=8, bbox_to_anchor=(0.5,-0.12), frameon=False)
ax[0,0].set_facecolor(BG); ax[0,0].set_title("Figure 1: Overall Survival Distribution", fontsize=11, fontweight='bold', color=B)

bar(ax[0,1], ["Female","Male"], df.groupby("Sex")["Survived"].mean().values*100, [G,R], "Figure 2: Survival Rate by Gender")
bar(ax[0,2], ["1st","2nd","3rd"], df.groupby("Pclass")["Survived"].mean().values*100, [M,L,"#5B9BD5"], "Figure 3: Survival by Class", ylim=80)
bar(ax[1,0], df.groupby("AgeGroup",observed=True)["Survived"].mean().index.tolist(), df.groupby("AgeGroup",observed=True)["Survived"].mean().values*100, [M,"#5B9BD5",L,"#A8C8E8","#D0E8F5"], "Figure 4: Survival by Age Group", ylim=75)
bar(ax[1,1], df.groupby("FamilySize")["Survived"].mean().index.astype(str), df.groupby("FamilySize")["Survived"].mean().values*100, M, "Figure 5: Survival by Family Size", ylim=100)
bar(ax[1,2], ["1st","2nd","3rd"], df.groupby("Pclass")["Fare"].mean().values, [M,"#5B9BD5",L], "Figure 6: Avg Fare by Class", ylabel="Avg Fare (£)", ylim=110, prefix="£", suffix="")

plt.tight_layout()
plt.savefig("titanic_graphs.png", dpi=150, bbox_inches="tight")
print("\nGraphs saved as titanic_graphs.png")
