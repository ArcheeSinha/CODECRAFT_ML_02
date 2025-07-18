# 🧠 Customer Segmentation using K-Means Clustering

This project uses the K-Means clustering algorithm to segment customers of a retail store based on their annual income and spending score. The goal is to help the business better understand customer behavior and target marketing strategies accordingly.

---

## 📊 Dataset

- Source: [Kaggle - Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- File Used: `Mall_Customers.csv`

---

## 📌 Features Used for Clustering

- **Annual Income (k$)** – Customer's yearly income
- **Spending Score (1–100)** – Score assigned by the store based on customer behavior

---

## 🛠️ Technologies Used

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🔧 Installation & Setup

1. Clone the repository or download the project folder.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate    # Windows
Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Download and place Mall_Customers.csv into your project directory.

Run the script:

bash
Copy
Edit
python kmeans_model.py
🚀 Output
Elbow Method Plot to determine the optimal number of clusters.

Scatter Plot showing customer segments based on income and spending.

📁 File Structure
Copy
Edit
customer_segmentation/
├── kmeans_model.py
├── Mall_Customers.csv
├── README.md
└── requirements.txt
📌 Author
Archee Sinha
2nd Year B.Tech CSE (AI) Student
📫 Let’s connect on LinkedIn
