from predict.predict_model import LGBMPredictor
import streamlit as st
import base64
import os


def run_app(
    artifacts_dir: str
    ):

    """
    Streamlit app to interact and visualize the 
    results in a user interface
    
    """


    st.set_page_config(
        page_title="Customer Churn System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
        
    def set_background(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        page_bg = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        .stApp {{
            background-color: rgba(0, 0, 64, 0.85); /* dark blue overlay */
            color: #FFFFFF;
        }}
        .css-18e3th9 {{
            background-color: transparent !important;
        }}s
        </style>
        """

        st.markdown(page_bg, unsafe_allow_html=True)


    background_image = os.path.join("dataset", "background.png")

    if os.path.exists(background_image):
        set_background(background_image)
        
        st.markdown(
            """
            <h1 style='text-align:center; color:#D3D3D3; font-size:42px; text-shadow: 2px 2px 8px #000000;'>
                Customer Churn Prediction 💡
            </h1>
            <div style='background-color:rgba(255,255,255,0.1);
                        border-radius:10px;
                        padding:10px 20px;
                        margin:auto;
                        width:70%;
                        text-align:center;'>
                <p style='font-size:18px; color:#00FFFF; font-weight:400; text-shadow:1px 1px 3px #000000;'>
                    Enter customer details below to predict whether they are likely to churn.
                </p>
            </div>
            """,
            unsafe_allow_html=True
            
            )

        try:

            predictor = LGBMPredictor(artifacts_dir=artifacts_dir)
        
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🔧 Input Customer Information")
       
        col1, col2 = st.columns(2)

        with col1:


            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", [0, 1])
            dependents = st.selectbox("Dependents", [0, 1])

            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperlessbilling = st.selectbox("Paperless Billing", [0, 1])
            payment_method = st.selectbox(
                "Payment Method",
                ['Mailed check', 'Electronic check', 'Automatic']
            )

            monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
            total_charges = st.number_input("Total Charges", min_value=0.0)
            
            fibre_stream_pref = st.selectbox("Fibre Stream Pref", [0, 1])
            dsl_security_pref = st.selectbox("DSL Security Pref", [0, 1])
            has_phone = st.selectbox("Has Phone", [0, 1])
            has_multipleline = st.selectbox("Has Multiple Line", [0, 1])

        
        with col2:

            tenure_bucket = st.selectbox("Tenure Bucket", ["0-3", "4-12", "13-24", "25-48", "49-72", "72+"])
            avg_monthly_charge = total_charges / tenure if tenure > 0 else 0.0
            total_addons = st.number_input("Total Addons", min_value=0, value=1)
            contract_payment = f"{contract}_{payment_method}"
            security_bins = st.number_input("Security Addons Bin", min_value=0, value=1, step=1)
            streaming_bins = st.number_input("Streaming Addons Bin", min_value=0, value=1, step=1)
            new_customers = 0 if tenure > 3 else 1
            spend_per_addon = monthly_charges / total_addons if total_addons > 0 else 0.0

       
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Predict Churn", use_container_width=True):
            
            try:

                inputs = {

                    'SeniorCitizen': senior_citizen,
                    'Partner': partner,
                    'Dependents':dependents,
                    'tenure':tenure,
                    'InternetService':internet_service,
                    "Contract": contract,
                    "PaperlessBilling": paperlessbilling,
                    "PaymentMethod": payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges,
                    'Fibre_stream_pref': fibre_stream_pref,
                    'DSL_security_pref': dsl_security_pref,
                    'has_phone': has_phone,
                    'has_multipleline': has_multipleline,
                    "tenure_bucket": tenure_bucket,
                    "avg_monthly_charge": avg_monthly_charge,
                    'total_addons': total_addons,
                    "contract_payment": contract_payment,
                    "security_bins": security_bins,
                    'streaming_bins': streaming_bins,
                    'new_customers': new_customers,
                    'spend_per_addon': spend_per_addon

                }


                preds, probs = predictor.predict_from_dict(inputs)
                probs = probs[0] * 100

                prediction_label = "Customer will likely Stay" if preds[0] == 0 else "Customer will likely Churn"
                probability = round(float(probs), 2)
                
                bg_color = "#228B22" if preds[0] == 0 else "#8B0000"

                st.markdown(

                  f"""
                    <div style="
                        background-color: {bg_color};
                        border-radius: 12px;
                        padding: 14px 20px;
                        text-align: center;
                        box-shadow: 0px 3px 10px rgba(0,0,0,0.3);
                        margin-top: 20px;
                        width: 55%;
                        margin-left: auto;
                        margin-right: auto;
                    ">
                        <h2 style="color: #FFD700; text-shadow: 1px 1px 2px #000; font-size: 26px; margin-bottom: 10px;">
                            🧠 Prediction Result
                        </h2>
                        <h3 style="color: #FFFFFF; font-size: 22px; margin-bottom: 5px;">
                            {prediction_label}
                        </h3>
                        <p style="color: #B0C4DE; font-size: 16px; margin-top: 0;">
                            Confidence: <b style="color: #00FFCC;">{probability}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True

                )


    
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")


st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 Customer Churn Predictor | Built with Streamlit</p>",
    unsafe_allow_html=True
)

