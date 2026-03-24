import pandas as pd
import plotly.express as px

# 1. Load Data
df = pd.read_csv('kc_house_data.csv')

# 2. Check Data
print("--- Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Plot Analysis (จะเปิดหน้าต่างเบราว์เซอร์โชว์กราฟ)
# กราฟ 1: พื้นที่ vs ราคา
fig1 = px.scatter(df, x="sqft_living", y="price", color="floors", 
                 trendline="ols", title="Living Area vs Price Analysis")
fig1.show()

# กราฟ 2: การกระจายตัวของราคา
fig2 = px.histogram(df, x="price", nbins=50, title="Price Distribution")
fig2.show()

print("\nEDA Success! Graphs are shown in your browser.")