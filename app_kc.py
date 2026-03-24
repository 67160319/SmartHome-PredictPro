import streamlit as st
import pandas as pd
import joblib
import os

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# --- โหลดข้อมูลเพื่อดึงชื่อเขต ---
@st.cache_data
def load_data():
    return pd.read_csv('kc_house_data.csv')

df = load_data()

# --- Dictionary แปลง Zipcode เป็นชื่อย่าน/เขตพื้นที่ ---
zip_to_district = {
    98101: "Downtown Seattle (ใจกลางเมือง/ย่านธุรกิจ)",
    98004: "Bellevue (ย่านหรู/ศูนย์กลางเศรษฐกิจ)",
    98052: "Redmond (ย่านเทคโนโลยี/Microsoft)",
    98033: "Kirkland (ย่านริมทะเลสาบ)",
    98001: "Auburn (ย่านที่พักอาศัยตอนใต้)",
    98105: "University District (ย่านมหาวิทยาลัย)",
    98040: "Mercer Island (ย่านหรูบนเกาะ)",
    98115: "Green Lake (ย่านพักผ่อน/สวนสาธารณะ)",
    98027: "Issaquah (ย่านธรรมชาติใกล้ภูเขา)",
    98103: "Fremont/Wallingford (ย่านศิลปะและวัฒนธรรม)"
}

# --- โหลดโมเดล AI ---
MODEL_PATH = 'model/house_model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณารัน train_kc.py ก่อน")
    st.stop()

model = joblib.load(MODEL_PATH)

# --- ส่วนหน้าจอหลัก ---
st.title("🏠 ระบบพยากรณ์ราคาบ้าน (SmartHome Predictor)")
st.subheader("กรอกข้อมูลด้านล่างเพื่อประเมินราคาบ้านด้วย AI (จากทำเล/โครงการที่ไหน)")
st.markdown("---")

# =========================================
# ส่วนที่ 1: ข้อมูลทำเล (เปลี่ยนจากเลข Zipcode เป็นชื่อเขต)
# =========================================
st.subheader("📍 ข้อมูลทำเล / เขตพื้นที่")
available_zips = sorted(df['zipcode'].unique())
option_list = [f"{zip_to_district.get(z, f'Area Code {z}')} [{z}]" for z in available_zips]

selected_option = st.selectbox("เลือกทำเลที่ตั้งของบ้าน:", option_list)
st.success(f"📌 เขตพื้นที่ที่เลือกคือ: **{selected_option.split(' [')[0]}**")

# --- หมายเหตุเชิงลึก (Deep Notes) ---
with st.expander("🔍 อ่านหมายเหตุเชิงลึกและข้อจำกัดของโมเดล"):
    st.info("""
    **ข้อกำหนดการใช้งานและข้อสังเกตสำคัญ:**
    1. **ที่มาของข้อมูล:** AI นี้ฝึกสอนจากฐานข้อมูล King County, USA จึงอ้างอิงราคาตามตลาดสหรัฐฯ เป็นหลัก
    2. **โครงการที่ไหน:** ชื่อทำเลอ้างอิงตามเขตพื้นที่ทางภูมิศาสตร์ ปัจจัยด้าน 'แบรนด์โครงการ' ในไทยยังไม่ได้ถูกนำมาคำนวณ
    3. **ความแม่นยำ:** AI ประเมินจาก 5 ปัจจัยหลักด้านล่าง (ห้อง, พื้นที่, ชั้น) ซึ่งมีผลต่อราคาโดยตรง
    4. **การแปลงสกุลเงิน:** คำนวณเป็นบาท (THB) โดยอ้างอิงเรทมาตรฐาน 35 THB/USD
    """)

st.markdown("---")

# =========================================
# ส่วนที่ 2: รายละเอียดตัวบ้าน (พร้อมระบบแปลงหน่วย ตร.ม.)
# =========================================
st.subheader("📊 รายละเอียดตัวบ้าน")
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("🛏️ จำนวนห้องนอน", 1, 10, 3)
    bathrooms = st.number_input("🚿 จำนวนห้องน้ำ", 1.0, 8.0, 2.0, 0.5)
    floors = st.number_input("🏢 จำนวนชั้น", 1.0, 3.5, 1.0, 0.5)

with col2:
    # พื้นที่ใช้สอย + คำนวณเป็น ตร.ม.
    sqft_living = st.number_input("📐 พื้นที่ใช้สอย (ตารางฟุต)", 500, 10000, 1800)
    st.caption(f"➡️ คิดเป็นพื้นที่ประมาณ: **{sqft_living * 0.0929:,.2f} ตารางเมตร**")
    
    # ขนาดที่ดิน + คำนวณเป็น ตร.ม.
    sqft_lot = st.number_input("🌳 ขนาดที่ดิน (ตารางฟุต)", 500, 50000, 5000)
    st.caption(f"➡️ คิดเป็นที่ดินประมาณ: **{sqft_lot * 0.0929:,.2f} ตารางเมตร**")

# =========================================
# ส่วนที่ 3: ปุ่มกดคำนวณ
# =========================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 พยากรณ์ราคา"):
    input_df = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors]], 
                            columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors'])
    
    prediction = model.predict(input_df)[0]
    st.balloons()
    st.success(f"### 💰 ราคาประเมิน: {prediction * 35:,.2f} บาท")