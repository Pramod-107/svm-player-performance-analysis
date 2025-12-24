# Player Performance Evaluation Using Support Vector Machines

> A comparative machine learning framework for evaluating player performance across multiple sports using Support Vector Machines (SVM).

---

## 📖 Overview
This project presents a **unified machine learning approach** to evaluate and compare player performance across **cricket, football, and basketball** using **Support Vector Machines (SVM)**.  
The framework converts raw performance statistics into normalized scores and classifies players into **Low, Medium, and High** performance tiers, enabling objective and data-driven analysis across different sports domains.

---

## 🎯 Objectives
- Develop a common evaluation framework across multiple sports
- Reduce subjectivity in player performance assessment
- Apply SVM with RBF kernel for non-linear classification
- Generate interpretable performance scores and rankings

---

## ✨ Key Features
- 📊 Multi-sport performance analysis (Cricket, Football, Basketball)
- 🧮 Statistical normalization and feature scaling
- 🤖 SVM-based classification (Low / Medium / High)
- 📈 Hybrid scoring using statistical and ML-based predictions
- 🎥 Visual analytics including distributions, heatmaps, and animations
- 🏅 Role-based and position-wise player ranking

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Model:** Support Vector Machine (RBF Kernel)

---

## 🧠 Methodology
1. Data cleaning and preprocessing  
2. Feature normalization using Min-Max and Standard scaling  
3. Performance score calculation from normalized metrics  
4. SVM-based classification into Low, Medium, and High tiers  
5. Final score computation using hybrid scoring  
6. Visualization and ranking of players  

---

## 📂 Datasets Used
The datasets used in this project were obtained from **publicly available sports analytics sources**:

### 🏏 Cricket (IPL)
- **Dataset:** IPL Player Performance Dataset  
- **Source:** Kaggle  
- **Metrics:** Runs scored, batting average, strike rate, wickets taken, economy rate, bowling strike rate, stumpings  

### ⚽ Football (FIFA)
- **Dataset:** FIFA Player Statistics Dataset  
- **Source:** Kaggle  
- **Metrics:** Goals scored, assists, dribbles per 90, tackles per 90, interceptions per 90, duels won per 90  

### 🏀 Basketball (NBA)
- **Dataset:** NBA Player Performance Dataset  
- **Source:** Kaggle  
- **Metrics:** Points (PTS), assists (AST), rebounds (TRB), steals (STL), blocks (BLK), field goal percentage (FG%)  

---

## 📈 Results
- High classification accuracy across all three sports  
- Effective separation of players into performance tiers  
- Role-aware and position-wise ranking of top performers  
- Consistent evaluation across heterogeneous sports data  

---

## 🚀 Future Enhancements
- Include age, experience, and fitness-related features  
- Time-series and match-by-match performance tracking  
- Hyperparameter tuning and model comparison  
- Interactive dashboards for real-time analytics  
- Extension to additional sports and leagues  

