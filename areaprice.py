import pickle
import streamlit as st
import pandas as pd
import numpy as np
model1=pickle.load(open("area.pkl","rb"))
st.set_page_config(page_title="🏠 Area Price Predictor", layout="centered")

def myf1():

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20230715/pngtree-palantir-crystal-ball-a-3d-rendering-of-middle-earth-s-future-image_3856097.jpg");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Area Price Prediction")
    st.markdown("Predict the **price** of a property based on its **area** using a trained ML model.")
    area=st.number_input("Enter the area value...")
    pred=st.button("Predict Price")

    if pred:
        op=model1.predict([[area]])
        st.write("price of the area is : ",op[0])
        # --- Generate data for visualizations ---
        area_range = np.arange(500, 5001, 100)
        predicted_prices = [model1.predict([[a]])[0] for a in area_range]

        df = pd.DataFrame({
            "Area (sq ft)": area_range,
            "Predicted Price (₹)": predicted_prices
        })
 
        # --- Visualizations ---
        st.markdown("---")
        st.subheader("📉 Area Chart")
        st.area_chart(df.rename(columns={"Area (sq ft)": "index"}).set_index("index"))

        df1 = pd.read_csv("areaprice071.csv")

        df1.columns = df1.columns.str.strip()  # Remove extra spaces

        print(df1.columns)  # Debug print
        print(df1.head())   # Check data


        st.subheader("📊 Bar Chart - Avg Price by Area Range")
        df1["Area Bucket"] = pd.cut(df1["area"], bins=5)
        bar_data = df1.groupby("Area Bucket")["price"].mean().reset_index()
        bar_data['Area Bucket'] = bar_data['Area Bucket'].astype(str) 
        st.bar_chart(bar_data.rename(columns={"Area Bucket": "index"}).set_index("index"))

        st.subheader("📈 Line Chart - Price Trend")
        st.line_chart(df.rename(columns={"Area (sq ft)": "index"}).set_index("index"))

myf1()



