import streamlit as st
import os
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import pickle
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

try:
    from model import ImageClassifier
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

st.set_page_config(
    page_title="🐾 Anidex - Animal Pokédex",
    page_icon="🐾",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86C1;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .pokedex-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #3498db;
        margin: 15px 0;
    }
    .stat-item {
        background: rgba(52, 152, 219, 0.1);
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

ANIMAL_DATA = {
    'butterfly': {
        'emoji': '🦋', 
        'name': 'Butterfly',
        'habitat': 'Gardens, meadows, forests',
        'taille': '2-10 cm wingspan',
        'poids': '0.1-0.5 grams',
        'regime': 'Flower nectar',
        'caracteristiques': 'Complete metamorphosis, delicate flight',
        'fait_interessant': 'Can see ultraviolet colors invisible to the human eye'
    },
    'cat': {
        'emoji': '🐱', 
        'name': 'Cat',
        'habitat': 'Domestic, urban and rural',
        'taille': '23-25 cm (height)',
        'poids': '3-7 kg',
        'regime': 'Strict carnivore',
        'caracteristiques': 'Excellent night vision, agility',
        'fait_interessant': 'Purrs at a frequency that promotes bone healing'
    },
    'chicken': {
        'emoji': '🐔', 
        'name': 'Chicken',
        'habitat': 'Farms, chicken coops',
        'taille': '35-45 cm',
        'poids': '1.5-4 kg',
        'regime': 'Omnivore (seeds, insects)',
        'caracteristiques': 'Complex communication, social hierarchy',
        'fait_interessant': 'Can recognize more than 100 different faces'
    },
    'cow': {
        'emoji': '🐄', 
        'name': 'Cow',
        'habitat': 'Meadows, pastures',
        'taille': '120-150 cm (height)',
        'poids': '400-800 kg',
        'regime': 'Ruminant herbivore',
        'caracteristiques': '4-compartment stomach, social life',
        'fait_interessant': 'Can produce up to 40 liters of milk per day'
    },
    'dog': {
        'emoji': '🐕', 
        'name': 'Dog',
        'habitat': 'Domestic, all environments',
        'taille': '15-90 cm (depending on breed)',
        'poids': '1-90 kg (depending on breed)',
        'regime': 'Omnivore with carnivorous tendency',
        'caracteristiques': 'Loyalty, intelligence, developed sense of smell',
        'fait_interessant': 'Can detect certain diseases through smell'
    },
    'elephant': {
        'emoji': '🐘', 
        'name': 'Elephant',
        'habitat': 'African/Asian savannas, forests',
        'taille': '2.5-4 m (height)',
        'poids': '4000-7000 kg',
        'regime': 'Herbivore (300 kg of vegetation/day)',
        'caracteristiques': 'Exceptional memory, versatile trunk',
        'fait_interessant': 'Can hear infrasounds from kilometers away'
    },
    'horse': {
        'emoji': '🐴', 
        'name': 'Horse',
        'habitat': 'Meadows, stables, ranches',
        'taille': '140-180 cm (height at withers)',
        'poids': '380-900 kg',
        'regime': 'Herbivore (grass, hay, oats)',
        'caracteristiques': 'Speed, endurance, panoramic vision',
        'fait_interessant': 'Can sleep standing up thanks to a leg-locking system'
    },
    'sheep': {
        'emoji': '🐑', 
        'name': 'Sheep',
        'habitat': 'Pastures, hills, mountains',
        'taille': '60-100 cm (height)',
        'poids': '45-160 kg',
        'regime': 'Ruminant herbivore',
        'caracteristiques': 'Insulating wool, herd instinct',
        'fait_interessant': 'Can recognize up to 50 different faces for 2 years'
    },
    'spider': {
        'emoji': '🕷️', 
        'name': 'Spider',
        'habitat': 'Everywhere (8 legs = 8 habitats)',
        'taille': '0.5-30 cm (depending on species)',
        'poids': '0.1g-175g (depending on species)',
        'regime': 'Carnivore (insects, small animals)',
        'caracteristiques': 'Silk web, 8 eyes, venom',
        'fait_interessant': 'Spider silk is stronger than steel at equal weight'
    },
    'squirrel': {
        'emoji': '🐿️', 
        'name': 'Squirrel',
        'habitat': 'Forests, parks, urban gardens',
        'taille': '15-25 cm + tail 15-25 cm',
        'poids': '300-700 grams',
        'regime': 'Omnivore (nuts, seeds, fruits)',
        'caracteristiques': 'Acrobatic agility, spatial memory',
        'fait_interessant': 'Hides up to 10,000 nuts per season and remembers 80% of hiding spots'
    }
}

@st.cache_resource
def load_classifier():
    try:
        return ImageClassifier()
    except Exception as e:
        st.error(f"Model error: {e}")
        return None

def create_simple_chart(probabilities):
    classes = list(probabilities.keys())
    probs = [probabilities[class_name] * 100 for class_name in classes]
    
    sorted_data = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    classes_sorted, probs_sorted = zip(*sorted_data)
    
    top_classes = classes_sorted[:5]
    top_probs = probs_sorted[:5]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.barh(range(len(top_classes)), top_probs, color='skyblue')
    
    labels = [f"{ANIMAL_DATA[cls]['emoji']} {ANIMAL_DATA[cls]['name']}" 
              for cls in top_classes]
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Probability (%)')
    ax.set_title('🎯 Top 5 Predictions')
    
    for i, prob in enumerate(top_probs):
        ax.text(prob + 1, i, f'{prob:.1f}%', va='center')
    
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def save_image_to_dataset(image, predicted_class, original_filename):
    try:
        dataset_path = "data/"
        class_folder = os.path.join(dataset_path, predicted_class)
        
        if not os.path.exists(class_folder):
            os.makedirs(class_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_validated_{timestamp}_{original_filename}"
        save_path = os.path.join(class_folder, filename)
        
        image.save(save_path)
        
        log_file = os.path.join(dataset_path, "user_feedback.log")
        with open(log_file, "a") as f:
            f.write(f"{datetime.now()}: {predicted_class} - {filename} - VALIDATED\n")
        
        return True, save_path
        
    except Exception as e:
        return False, str(e)

def update_dataset_pickle():
    try:
        log_file = "../data/dataset_updates.log"
        with open(log_file, "a") as f:
            f.write(f"{datetime.now()}: Dataset update requested\n")
        return True
    except:
        return False

def display_pokedex_info(predicted_class):
    animal_info = ANIMAL_DATA[predicted_class]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 20px; border-radius: 15px; border: 2px solid #3498db; margin: 15px 0;">
        <h2 style="text-align: center; color: #2c3e50; margin-bottom: 5px;">
            {animal_info['emoji']} ANIDEX #{list(ANIMAL_DATA.keys()).index(predicted_class) + 1:03d}
        </h2>
        <h3 style="text-align: center; color: #34495e; margin-bottom: 20px;">
            {animal_info['name'].upper()}
        </h3>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"🏠 **Habitat**\n{animal_info['habitat']}")
        st.info(f"⚖️ **Weight**\n{animal_info['poids']}")
        st.info(f"⭐ **Characteristics**\n{animal_info['caracteristiques']}")
    
    with col2:
        st.info(f"📏 **Size**\n{animal_info['taille']}")
        st.info(f"🍽️ **Diet**\n{animal_info['regime']}")
        st.info(f"💡 **Interesting Fact**\n{animal_info['fait_interessant']}")

def main():
    st.markdown('<h1 class="main-title">🐾 ANIDEX</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Animal Pokédex</p>', unsafe_allow_html=True)
    
    if not MODEL_AVAILABLE:
        st.error("❌ Model not found!")
        return
    
    classifier = load_classifier()
    if classifier is None:
        st.error("❌ Unable to load model.")
        return
    
    st.success("✅ Model ready!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            help="Formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)
            
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("🧠 Analysis in progress..."):
                    try:
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        result = classifier.predict(temp_path, show_probabilities=False, show_image=False)
                        
                        os.remove(temp_path)
                        
                        st.session_state.result = result
                        st.session_state.image = image
                        st.session_state.filename = uploaded_file.name
                        st.session_state.feedback_given = False
                        st.session_state.show_correction = False
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    with col2:
        st.header("🎯 Result")
        
        if hasattr(st.session_state, 'result') and st.session_state.result:
            result = st.session_state.result
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            animal = ANIMAL_DATA.get(predicted_class, {'emoji': '❓', 'name': 'Unknown'})
            
            st.success(f"**{animal['emoji']} {animal['name'].upper()}**")
            st.metric("🎯 Confidence", f"{confidence:.1%}")
            
            st.progress(confidence)
            if confidence < 0.95:
                st.markdown(f"""
                <div class="warning-box">
                    <h4 style="color: #856404; margin: 0 0 10px 0;">⚠️ Warning - Low Confidence</h4>
                    <p style="color: #856404; margin: 0;">
                        The confidence is {confidence:.1%}, which is below 95%. 
                        This prediction might be incorrect. Please carefully verify 
                        the result and provide your feedback to improve the model.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            if not hasattr(st.session_state, 'feedback_given') or not st.session_state.feedback_given:
                st.markdown("---")
                st.subheader("📝 Is this prediction correct?")
                
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("✅ Yes, it's correct!", type="primary", use_container_width=True):
                        save_image_to_dataset(
                            st.session_state.image, 
                            predicted_class, 
                            st.session_state.filename
                        )
                        update_dataset_pickle()
                        st.session_state.feedback_given = True
                        st.success("✅ Thank you for your feedback! Image added to dataset.")
                        st.rerun()
                
                with col_no:
                    if st.button("❌ No, it's wrong", use_container_width=True):
                        st.session_state.show_correction = True
                    
                    if hasattr(st.session_state, 'show_correction') and st.session_state.show_correction:
                        st.markdown("**What is the correct class?**")
                        correct_class = st.selectbox(
                            "Choose:",
                            options=list(ANIMAL_DATA.keys()),
                            format_func=lambda x: f"{ANIMAL_DATA[x]['emoji']} {ANIMAL_DATA[x]['name']}",
                            key="correct_class_selector"
                        )
                        
                        if st.button("💾 Save with correct class"):
                            save_image_to_dataset(
                                st.session_state.image, 
                                correct_class, 
                                           st.session_state.filename
                            )
                            
                            log_file = "../data/user_feedback.log"
                            try:
                                with open(log_file, "a") as f:
                                    f.write(f"{datetime.now()}: CORRECTION - Predicted: {predicted_class}, Actual: {correct_class}\n")
                            except:
                                pass
                            
                            st.session_state.feedback_given = True
                            st.session_state.show_correction = False
                            st.success(f"✅ Thank you for the correction! Image saved as {ANIMAL_DATA[correct_class]['name']}.")
                            st.rerun()
            else:
                st.markdown("---")
                st.info("✅ Thank you for your feedback! Image added to dataset.")
            
            st.markdown("---")
            st.header(f"📚 Pokédex Card - {ANIMAL_DATA[predicted_class]['name']}")
            display_pokedex_info(predicted_class)
        
        else:
            st.info("👆 Upload an image to get started!")
    
    if hasattr(st.session_state, 'result') and st.session_state.result:
        st.markdown("---")
        st.header("📊 Probability Details")
        
        fig = create_simple_chart(st.session_state.result['probabilities'])
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("📋 All probabilities")
        
        prob_data = []
        for class_name, prob in st.session_state.result['probabilities'].items():
            animal = ANIMAL_DATA[class_name]
            prob_data.append({
                'Animal': f"{animal['emoji']} {animal['name']}",
                'Probability': f"{prob:.2%}"
            })
        
        prob_data.sort(key=lambda x: float(x['Probability'].strip('%')), reverse=True)
        
        cols = st.columns(2)
        mid = len(prob_data) // 2
        
        with cols[0]:
            for item in prob_data[:mid]:
                st.write(f"{item['Animal']}: **{item['Probability']}**")
        
        with cols[1]:
            for item in prob_data[mid:]:
                st.write(f"{item['Animal']}: **{item['Probability']}**")
    
    with st.sidebar:
        st.header("ℹ️ Information")
        st.info("🎯 **Accuracy**: 91.53%")
        st.info("🐾 **Classes**: 10 animals")
        
        st.markdown("### 🏆 Detectable animals")
        for class_name, data in ANIMAL_DATA.items():
            st.write(f"{data['emoji']} {data['name']}")
        
        st.markdown("---")
        st.markdown("### 📈 Continuous improvement")
        st.markdown("""
        Your feedback helps improve the model:
        - ✅ **Correct prediction** → Image added to dataset
        - ❌ **Incorrect prediction** → Correction recorded
        - 🔄 **Future training** → More accurate model
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🐾 Anidex - Simple Dashboard with User Feedback"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()