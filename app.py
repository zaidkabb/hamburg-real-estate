import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Hamburg Real Estate ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    h1 {
        color: #1e3a8a;
        font-size: 3em !important;
        font-weight: 800 !important;
        margin-bottom: 0.5em !important;
    }
    h2 {
        color: #667eea;
        font-weight: 600 !important;
    }
    h3 {
        color: #4a5568;
    }
    .price-result-sales {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3em 2em;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin: 2em 0;
    }
    .price-result-rental {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3em 2em;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
        margin: 2em 0;
    }
    .price-main {
        font-size: 4em;
        font-weight: bold;
        color: white;
        margin: 0.3em 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .price-label {
        font-size: 1.3em;
        color: #e0e7ff;
        margin-bottom: 0;
    }
    .price-subtitle {
        font-size: 1.1em;
        color: #c7d2fe;
        margin-top: 1em;
    }
    .info-box {
        background: white;
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1em 0;
    }
    .property-card-sales {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2em;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    .property-card-rental {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2em;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 25px rgba(245, 87, 108, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_sales_model():
    try:
        model = joblib.load('best_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except:
        return None, None

@st.cache_resource
def load_rental_model():
    try:
        model = joblib.load('rental_model.pkl')
        label_encoder = joblib.load('rental_label_encoder.pkl')
        return model, label_encoder
    except:
        return None, None

sales_model, sales_encoder = load_sales_model()
rental_model, rental_encoder = load_rental_model()

# Header
st.markdown("""
    <div style='text-align: center; padding: 2em 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 15px; margin-bottom: 2em; box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);'>
        <h1 style='color: white; margin: 0; font-size: 3em;'>🏠 Hamburg Real Estate ML</h1>
        <h2 style='color: #e0e7ff; margin-top: 0.5em; font-weight: 400;'>Smart Price Predictions for Sales & Rentals</h2>
        <p style='color: #e0e7ff; font-size: 1.1em; margin-top: 1em;'>
            Powered by advanced machine learning algorithms
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main tabs
tab1, tab2 = st.tabs(["💰 Buy Property (Sales)", "🏘️ Rent Property (Monthly)"])

# ==================== SALES TAB ====================
with tab1:
    if sales_model is None:
        st.error("⚠️ Sales model not found! Please run train_model.py first.")
    else:
        left_col, right_col = st.columns([1.2, 1], gap="large")
        
        with left_col:
            st.markdown("### 📋 Property Details")
            
            # Location
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #667eea;'>📍 Location</h4>
                </div>
            """, unsafe_allow_html=True)
            
            neighborhoods = sorted(sales_encoder.classes_)
            neighborhood_sales = st.selectbox(
                "Select Neighborhood",
                neighborhoods,
                help="Choose from 40 Hamburg neighborhoods",
                label_visibility="collapsed",
                key="sales_neighborhood"
            )
            
            # Dimensions
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #667eea;'>📐 Dimensions</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_size, col_rooms = st.columns(2)
            with col_size:
                size_sqm_sales = st.number_input(
                    "Size (m²)",
                    min_value=30,
                    max_value=200,
                    value=80,
                    step=5,
                    key="sales_size"
                )
            
            with col_rooms:
                rooms_sales = st.number_input(
                    "Rooms",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    key="sales_rooms"
                )
            
            # Building info
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #667eea;'>🏢 Building Info</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_year, col_floor = st.columns(2)
            with col_year:
                year_built_sales = st.number_input(
                    "Year Built",
                    min_value=1950,
                    max_value=2024,
                    value=2000,
                    step=1,
                    key="sales_year"
                )
            
            with col_floor:
                floor_sales = st.number_input(
                    "Floor",
                    min_value=0,
                    max_value=9,
                    value=2,
                    step=1,
                    key="sales_floor"
                )
            
            # Amenities
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #667eea;'>✨ Amenities</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_bal, col_park, col_elev = st.columns(3)
            with col_bal:
                has_balcony_sales = st.checkbox("🌿 Balcony", value=True, key="sales_balcony")
            with col_park:
                has_parking_sales = st.checkbox("🚗 Parking", value=False, key="sales_parking")
            with col_elev:
                has_elevator_sales = st.checkbox("🛗 Elevator", value=True, key="sales_elevator")
        
        with right_col:
            st.markdown("### 💡 Property Preview")
            
            st.markdown(f"""
                <div class='property-card-sales'>
                    <h2 style='color: white; margin-top: 0;'>{neighborhood_sales}</h2>
                    <hr style='border-color: rgba(255,255,255,0.3);'>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin-top: 1em;'>
                        <div>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>📐 Size:</b> {size_sqm_sales} m²</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🚪 Rooms:</b> {rooms_sales}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>📅 Built:</b> {year_built_sales}</p>
                        </div>
                        <div>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🏢 Floor:</b> {floor_sales}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🌿 Balcony:</b> {'Yes' if has_balcony_sales else 'No'}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🚗 Parking:</b> {'Yes' if has_parking_sales else 'No'}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn_sales = st.button("🔮 Predict Sale Price", type="primary", key="sales_predict")
        
        # Prediction results
        if predict_btn_sales:
            neighborhood_encoded = sales_encoder.transform([neighborhood_sales])[0]
            input_data = np.array([[
                size_sqm_sales, rooms_sales, year_built_sales,
                1 if has_balcony_sales else 0,
                1 if has_parking_sales else 0,
                floor_sales,
                1 if has_elevator_sales else 0,
                neighborhood_encoded
            ]])
            
            predicted_price = sales_model.predict(input_data)[0]
            price_per_sqm = predicted_price / size_sqm_sales
            lower = predicted_price * 0.9
            upper = predicted_price * 1.1
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='price-result-sales'>
                    <p class='price-label'>💰 Predicted Property Price</p>
                    <h1 class='price-main'>€{predicted_price:,.0f}</h1>
                    <p class='price-subtitle'>€{price_per_sqm:,.0f} per m² • Range: €{lower:,.0f} - €{upper:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class='info-box'>
                        <h4 style='color: #667eea; margin-top: 0;'>📊 Price Breakdown</h4>
                        <p style='margin: 0.5em 0;'><b>Total Price:</b> €{predicted_price:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Per m²:</b> €{price_per_sqm:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Per Room:</b> €{predicted_price/rooms_sales:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='info-box'>
                        <h4 style='color: #667eea; margin-top: 0;'>🎯 Confidence Range</h4>
                        <p style='margin: 0.5em 0;'><b>Lower Estimate:</b> €{lower:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Upper Estimate:</b> €{upper:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Range:</b> ±10%</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class='info-box'>
                        <h4 style='color: #667eea; margin-top: 0;'>🏠 Property Summary</h4>
                        <p style='margin: 0.5em 0;'><b>Location:</b> {neighborhood_sales}</p>
                        <p style='margin: 0.5em 0;'><b>Size:</b> {size_sqm_sales}m² • {rooms_sales} rooms</p>
                        <p style='margin: 0.5em 0;'><b>Built:</b> {year_built_sales} • Floor {floor_sales}</p>
                    </div>
                """, unsafe_allow_html=True)

# ==================== RENTAL TAB ====================
with tab2:
    if rental_model is None:
        st.error("⚠️ Rental model not found! Please run train_rental_model.py first.")
    else:
        left_col, right_col = st.columns([1.2, 1], gap="large")
        
        with left_col:
            st.markdown("### 📋 Property Details")
            
            # Location
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #f5576c;'>📍 Location</h4>
                </div>
            """, unsafe_allow_html=True)
            
            neighborhoods_rental = sorted(rental_encoder.classes_)
            neighborhood_rental = st.selectbox(
                "Select Neighborhood",
                neighborhoods_rental,
                help="Choose from 40 Hamburg neighborhoods",
                label_visibility="collapsed",
                key="rental_neighborhood"
            )
            
            # Dimensions
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #f5576c;'>📐 Dimensions</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_size, col_rooms = st.columns(2)
            with col_size:
                size_sqm_rental = st.number_input(
                    "Size (m²)",
                    min_value=30,
                    max_value=200,
                    value=80,
                    step=5,
                    key="rental_size"
                )
            
            with col_rooms:
                rooms_rental = st.number_input(
                    "Rooms",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    key="rental_rooms"
                )
            
            # Building info
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #f5576c;'>🏢 Building Info</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_year, col_floor = st.columns(2)
            with col_year:
                year_built_rental = st.number_input(
                    "Year Built",
                    min_value=1950,
                    max_value=2024,
                    value=2000,
                    step=1,
                    key="rental_year"
                )
            
            with col_floor:
                floor_rental = st.number_input(
                    "Floor",
                    min_value=0,
                    max_value=9,
                    value=2,
                    step=1,
                    key="rental_floor"
                )
            
            # Amenities
            st.markdown("""
                <div class='feature-box'>
                    <h4 style='margin-top: 0; color: #f5576c;'>✨ Amenities</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col_bal, col_park, col_elev, col_furn = st.columns(4)
            with col_bal:
                has_balcony_rental = st.checkbox("🌿 Balcony", value=True, key="rental_balcony")
            with col_park:
                has_parking_rental = st.checkbox("🚗 Parking", value=False, key="rental_parking")
            with col_elev:
                has_elevator_rental = st.checkbox("🛗 Elevator", value=True, key="rental_elevator")
            with col_furn:
                is_furnished = st.checkbox("🛋️ Furnished", value=False, key="rental_furnished")
        
        with right_col:
            st.markdown("### 💡 Property Preview")
            
            st.markdown(f"""
                <div class='property-card-rental'>
                    <h2 style='color: white; margin-top: 0;'>{neighborhood_rental}</h2>
                    <hr style='border-color: rgba(255,255,255,0.3);'>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin-top: 1em;'>
                        <div>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>📐 Size:</b> {size_sqm_rental} m²</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🚪 Rooms:</b> {rooms_rental}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>📅 Built:</b> {year_built_rental}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🛋️ Furnished:</b> {'Yes' if is_furnished else 'No'}</p>
                        </div>
                        <div>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🏢 Floor:</b> {floor_rental}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🌿 Balcony:</b> {'Yes' if has_balcony_rental else 'No'}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🚗 Parking:</b> {'Yes' if has_parking_rental else 'No'}</p>
                            <p style='margin: 0.5em 0; font-size: 1.1em;'><b>🛗 Elevator:</b> {'Yes' if has_elevator_rental else 'No'}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn_rental = st.button("🔮 Predict Monthly Rent", type="primary", key="rental_predict")
        
        # Prediction results
        if predict_btn_rental:
            neighborhood_encoded = rental_encoder.transform([neighborhood_rental])[0]
            input_data = np.array([[
                size_sqm_rental, rooms_rental, year_built_rental,
                1 if has_balcony_rental else 0,
                1 if has_parking_rental else 0,
                floor_rental,
                1 if has_elevator_rental else 0,
                1 if is_furnished else 0,
                neighborhood_encoded
            ]])
            
            predicted_rent = rental_model.predict(input_data)[0]
            rent_per_sqm = predicted_rent / size_sqm_rental
            utilities = size_sqm_rental * 2.5  # Estimate
            total_rent = predicted_rent + utilities
            lower = predicted_rent * 0.9
            upper = predicted_rent * 1.1
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='price-result-rental'>
                    <p class='price-label'>🏘️ Predicted Monthly Rent</p>
                    <h1 class='price-main'>€{predicted_rent:,.0f}</h1>
                    <p class='price-subtitle'>€{rent_per_sqm:,.2f} per m² • Range: €{lower:,.0f} - €{upper:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class='info-box'>
                        <h4 style='color: #f5576c; margin-top: 0;'>💵 Rent Breakdown</h4>
                        <p style='margin: 0.5em 0;'><b>Base Rent:</b> €{predicted_rent:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Utilities (est.):</b> €{utilities:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Total/Month:</b> €{total_rent:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='info-box'>
                        <h4 style='color: #f5576c; margin-top: 0;'>🎯 Confidence Range</h4>
                        <p style='margin: 0.5em 0;'><b>Lower Estimate:</b> €{lower:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Upper Estimate:</b> €{upper:,.0f}</p>
                        <p style='margin: 0.5em 0;'><b>Range:</b> ±10%</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class='info-box'>
                        <h4 style='color: #f5576c; margin-top: 0;'>🏠 Property Summary</h4>
                        <p style='margin: 0.5em 0;'><b>Location:</b> {neighborhood_rental}</p>
                        <p style='margin: 0.5em 0;'><b>Size:</b> {size_sqm_rental}m² • {rooms_rental} rooms</p>
                        <p style='margin: 0.5em 0;'><b>Type:</b> {'Furnished' if is_furnished else 'Unfurnished'}</p>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9ca3af;'>Made with ❤️ By ZAID KABBA</p>", unsafe_allow_html=True)