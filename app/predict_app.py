from predict.predict_model import LGBMPredictor
import streamlit as st


def run_app(
    artifacts_dir: str
    ):

    """
    Streamlit app to interact and visualize the 
    results in a user interface
    
    """

    try:

        predictor = LGBMPredictor(artificats_dir=artifacts_dir)

        pred_class = {0: "No Churn", 1: "Churn"}

        st.set_page_config(
            page_title="📊 Customer Churn Prediction App"
            )
        
        st.write("Enter customer details below to predict whether they are likely to churn")

        with st.form("Churn form"):

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

            
            tenure_bucket = st.selectbox("Tenure Bucket", ["0-3", "4-12", "13-24", "25-48", "49-72", "72+"])
            avg_monthly_charge = total_charges / tenure if tenure > 0 else 0.0
            total_addons = st.number_input("Total Addons", min_value=0, value=1)
            contract_payment = f"{contract}_{payment_method}"
            security_bins = st.slider("Security Addons Bin", 0, 1, 2, 3, 4, 5)
            streaming_bins = st.slider("Streaming Addons Bin", 0, 1, 2, 3, 4, 5)
            new_customers = st.selectbox("New Customer", [0, 1])
            spend_per_addon = st.number_input("Spend per Addon", min_value=0.0, value=20.12)

            submitted = st.form_submit_button("Predict")

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

            prediction_label = pred_class[int(preds[0])]
            probability = float(probs[0])

            st.subheader("🔮 Prediction Result")
            st.write(f"**Prediction**: {prediction_label}")
            st.progress(probability)
            st.write(f"**Probability Score**: {probability:.2f}")

    
    except Exception as e:
        print("Error occurred during streamlit app building as:", str(e))
        raise e

if __name__ == "__main__":
    run_app(artifacts_dir="models_artifact/")