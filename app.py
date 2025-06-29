import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit.components.v1 as components

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("""
This app uses a trained XGBoost model to classify the overall healthiness of a ready-to-eat food into five levels (HSR grades A to E).  
**Input variables explanation**:
- `Protein`, `Sodium`, `Total fat`, `Energy`: Nutrient values per 100g  
- `weight`: Total package weight (g)  
- `procef_4`: 1 = ultra-processed, 0 = not  
- `ifclaim`: Whether any nutrition/health/other claim exists (1/0)  
- `ifnurclaim`: Whether a nutrition claim is present  
- `nutclaim3`: Specific type of nutrient claim
""")

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    try:
        return joblib.load("XGBoost_final_model_selected_9.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please upload 'XGBoost_final_model_selected_9.pkl'.")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler2.pkl")
    except FileNotFoundError:
        st.error("❌ Scaler file not found. Please upload 'scaler2.pkl'.")
        st.stop()

@st.cache_resource
def load_background_data():
    try:
        return np.load("background_data.npy")
    except FileNotFoundError:
        st.error("❌ Background data not found. Please upload 'background_data.npy'.")
        st.stop()

model = load_model()
scaler = load_scaler()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# ===== Nutri-Score 风格图像绘制函数 =====
def draw_nutriscore_final(predicted_label):
    labels = ['A', 'B', 'C', 'D', 'E']
    colors = ['#00843D', '#A8C92D', '#FECB00', '#EF7D00', '#E60012']
    fig, ax = plt.subplots(figsize=(6.5, 2.5))

    background = patches.FancyBboxPatch((0, 0), 5, 1.2,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        edgecolor='gray', facecolor='white', linewidth=2)
    ax.add_patch(background)

    for i, (label, color) in enumerate(zip(labels, colors)):
        rect = patches.FancyBboxPatch((i, 0), 1, 1,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

        if label == predicted_label.upper():
            circle = patches.Circle((i + 0.5, 0.5), radius=0.55,
                facecolor='white', alpha=0.25, edgecolor=None, zorder=2)
            ax.add_patch(circle)
            ax.text(i + 0.5, 0.5, label,
                ha='center', va='center', fontsize=36,
                weight='bold', color='white', zorder=3)
        else:
            ax.text(i + 0.5, 0.5, label,
                ha='center', va='center', fontsize=26,
                weight='bold', color='white', alpha=0.3, zorder=2)

    ax.text(2.5, 1.15, 'PREDICTED HEALTHINESS',
        ha='center', va='bottom', fontsize=14,
        weight='bold', color='black')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.4)
    ax.axis('off')
    plt.tight_layout()
    return fig

# ===== 输入栏 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)
total_fat = st.sidebar.number_input("Total Fat (g/100g)", min_value=0.0, step=0.1)
weight = st.sidebar.number_input("Weight (g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Ultra-Processed? (procef_4)", [0, 1])
ifclaim = st.sidebar.selectbox("Any Claim Present? (ifclaim)", [0, 1])
ifnurclaim = st.sidebar.selectbox("Nutrition Claim Present? (ifnurclaim)", [0, 1])
nutclaim3 = st.sidebar.selectbox("Specific Nutrient Claim (nutclaim3)", [0, 1])

# ===== 预测逻辑 =====
if st.sidebar.button("🧮 Predict"):
    scaled_columns = ['Sodium', 'Protein', 'Energy', 'Total fat', 'weight',
                      'ifclaim', 'ifnurclaim', 'nutclaim3']
    final_columns = scaled_columns + ['procef_4']

    input_dict = {
        "Sodium": sodium,
        "Protein": protein,
        "Energy": energy,
        "Total fat": total_fat,
        "weight": weight,
        "ifclaim": ifclaim,
        "ifnurclaim": ifnurclaim,
        "nutclaim3": nutclaim3
    }

    user_input_for_scaler = pd.DataFrame([[input_dict[feat] for feat in scaled_columns]], columns=scaled_columns)
    user_scaled_part = scaler.transform(user_input_for_scaler)
    user_scaled_df = pd.DataFrame(user_scaled_part, columns=scaled_columns)

    user_scaled_df["procef_4"] = procef_4
    user_scaled_df = user_scaled_df[final_columns]

    prediction = model.predict(user_scaled_df)[0]
    prob_array = model.predict_proba(user_scaled_df)[0]
    label_map = {0: 'E', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
    predicted_label = label_map.get(prediction, f"Class {prediction}")

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** `{predicted_label}`")
    st.pyplot(draw_nutriscore_final(predicted_label))

    st.subheader("📊 Probability Table")
    prob_df = pd.DataFrame({
        "HSR Class": [label_map[i] for i in range(len(prob_array))],
        "Probability": [f"{p:.2f}" for p in prob_array]
    })
    st.dataframe(prob_df, use_container_width=True)

    st.subheader("📈 SHAP Force Plot (Model Explanation)")
    with st.expander("Click to view SHAP force plot"):
        shap_values = explainer(user_scaled_df)
        shap_for_sample = shap_values[0]
        force_html = shap.force_plot(
            explainer.expected_value,
            shap_for_sample.values,
            shap_for_sample.data,
            feature_names=shap_for_sample.feature_names,
            matplotlib=False
        )
        components.html(shap.getjs() + force_html.html(), height=400)



# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
