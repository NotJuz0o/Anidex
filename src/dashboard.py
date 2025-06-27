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
    page_title="🐾 Anidex - Pokédex Animalier",
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
</style>
""", unsafe_allow_html=True)

ANIMAL_DATA = {
    'butterfly': {
        'emoji': '🦋', 
        'name': 'Papillon',
        'habitat': 'Jardins, prairies, forêts',
        'taille': '2-10 cm d\'envergure',
        'poids': '0.1-0.5 grammes',
        'regime': 'Nectar des fleurs',
        'caracteristiques': 'Métamorphose complète, vol délicat',
        'fait_interessant': 'Peut voir les couleurs ultraviolettes invisibles à l\'œil humain'
    },
    'cat': {
        'emoji': '🐱', 
        'name': 'Chat',
        'habitat': 'Domestique, urbain et rural',
        'taille': '23-25 cm (hauteur)',
        'poids': '3-7 kg',
        'regime': 'Carnivore strict',
        'caracteristiques': 'Vision nocturne excellente, agilité',
        'fait_interessant': 'Ronronne à une fréquence qui favorise la guérison osseuse'
    },
    'chicken': {
        'emoji': '🐔', 
        'name': 'Poulet',
        'habitat': 'Fermes, basses-cours',
        'taille': '35-45 cm',
        'poids': '1.5-4 kg',
        'regime': 'Omnivore (graines, insectes)',
        'caracteristiques': 'Communication complexe, hiérarchie sociale',
        'fait_interessant': 'Peut reconnaître plus de 100 visages différents'
    },
    'cow': {
        'emoji': '🐄', 
        'name': 'Vache',
        'habitat': 'Prairies, pâturages',
        'taille': '120-150 cm (hauteur)',
        'poids': '400-800 kg',
        'regime': 'Herbivore ruminant',
        'caracteristiques': 'Estomac à 4 compartiments, vie sociale',
        'fait_interessant': 'Peut produire jusqu\'à 40 litres de lait par jour'
    },
    'dog': {
        'emoji': '🐕', 
        'name': 'Chien',
        'habitat': 'Domestique, tous environnements',
        'taille': '15-90 cm (selon race)',
        'poids': '1-90 kg (selon race)',
        'regime': 'Omnivore à tendance carnivore',
        'caracteristiques': 'Loyauté, intelligence, odorat développé',
        'fait_interessant': 'Peut détecter certaines maladies grâce à son odorat'
    },
    'elephant': {
        'emoji': '🐘', 
        'name': 'Éléphant',
        'habitat': 'Savanes, forêts africaines/asiatiques',
        'taille': '2.5-4 m (hauteur)',
        'poids': '4000-7000 kg',
        'regime': 'Herbivore (300 kg de végétaux/jour)',
        'caracteristiques': 'Mémoire exceptionnelle, trompe polyvalente',
        'fait_interessant': 'Peut entendre des infrasons à des kilomètres de distance'
    },
    'horse': {
        'emoji': '🐴', 
        'name': 'Cheval',
        'habitat': 'Prairies, écuries, ranch',
        'taille': '140-180 cm (hauteur au garrot)',
        'poids': '380-900 kg',
        'regime': 'Herbivore (herbe, foin, avoine)',
        'caracteristiques': 'Vitesse, endurance, vision panoramique',
        'fait_interessant': 'Peut dormir debout grâce à un système de verrouillage des pattes'
    },
    'sheep': {
        'emoji': '🐑', 
        'name': 'Mouton',
        'habitat': 'Pâturages, collines, montagnes',
        'taille': '60-100 cm (hauteur)',
        'poids': '45-160 kg',
        'regime': 'Herbivore ruminant',
        'caracteristiques': 'Laine isolante, instinct grégaire',
        'fait_interessant': 'Peut reconnaître jusqu\'à 50 visages différents pendant 2 ans'
    },
    'spider': {
        'emoji': '🕷️', 
        'name': 'Araignée',
        'habitat': 'Partout (8 pattes = 8 habitats)',
        'taille': '0.5-30 cm (selon espèce)',
        'poids': '0.1g-175g (selon espèce)',
        'regime': 'Carnivore (insectes, petits animaux)',
        'caracteristiques': 'Toile de soie, 8 yeux, venin',
        'fait_interessant': 'La soie d\'araignée est plus résistante que l\'acier à poids égal'
    },
    'squirrel': {
        'emoji': '🐿️', 
        'name': 'Écureuil',
        'habitat': 'Forêts, parcs, jardins urbains',
        'taille': '15-25 cm + queue 15-25 cm',
        'poids': '300-700 grammes',
        'regime': 'Omnivore (noix, graines, fruits)',
        'caracteristiques': 'Agilité acrobatique, mémoire spatiale',
        'fait_interessant': 'Cache jusqu\'à 10 000 noix par saison et se souvient de 80% des cachettes'
    }
}

@st.cache_resource
def load_classifier():
    try:
        return ImageClassifier()
    except Exception as e:
        st.error(f"Erreur modèle: {e}")
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
    ax.set_xlabel('Probabilité (%)')
    ax.set_title('🎯 Top 5 Prédictions')
    
    for i, prob in enumerate(top_probs):
        ax.text(prob + 1, i, f'{prob:.1f}%', va='center')
    
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def save_image_to_dataset(image, predicted_class, original_filename):
    try:
        dataset_path = "../data"
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
        st.info(f"⚖️ **Poids**\n{animal_info['poids']}")
        st.info(f"⭐ **Caractéristiques**\n{animal_info['caracteristiques']}")
    
    with col2:
        st.info(f"📏 **Taille**\n{animal_info['taille']}")
        st.info(f"🍽️ **Régime**\n{animal_info['regime']}")
        st.info(f"💡 **Fait intéressant**\n{animal_info['fait_interessant']}")

def main():
    st.markdown('<h1 class="main-title">🐾 ANIDEX</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pokédex Animalier Simple</p>', unsafe_allow_html=True)
    
    if not MODEL_AVAILABLE:
        st.error("❌ Modèle non trouvé!")
        return
    
    classifier = load_classifier()
    if classifier is None:
        st.error("❌ Impossible de charger le modèle.")
        return
    
    st.success("✅ Modèle prêt!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Votre Image")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['png', 'jpg', 'jpeg'],
            help="Formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
            
            if st.button("🔍 Analyser", type="primary", use_container_width=True):
                with st.spinner("🧠 Analyse en cours..."):
                    try:
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        result = classifier.predict(temp_path, show_probabilities=False, show_image=False)
                        
                        os.remove(temp_path)
                        
                        st.session_state.result = result
                        st.session_state.image = image
                        st.session_state.filename = uploaded_file.name
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
    
    with col2:
        st.header("🎯 Résultat")
        
        if hasattr(st.session_state, 'result') and st.session_state.result:
            result = st.session_state.result
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            animal = ANIMAL_DATA.get(predicted_class, {'emoji': '❓', 'name': 'Inconnu'})
            
            st.success(f"**{animal['emoji']} {animal['name'].upper()}**")
            st.metric("🎯 Confiance", f"{confidence:.1%}")
            
            st.progress(confidence)
            
            st.markdown("---")
            st.subheader("📝 Cette prédiction est-elle correcte?")
            
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                if st.button("✅ Oui, c'est correct!", type="primary", use_container_width=True):
                    save_image_to_dataset(
                        st.session_state.image, 
                        predicted_class, 
                        st.session_state.filename
                    )
                    update_dataset_pickle()
            
            with col_no:
                if st.button("❌ Non, c'est faux", use_container_width=True):
                    st.session_state.show_correction = True
                
                if hasattr(st.session_state, 'show_correction') and st.session_state.show_correction:
                    st.markdown("**Quelle est la vraie classe?**")
                    correct_class = st.selectbox(
                        "Choisissez:",
                        options=list(ANIMAL_DATA.keys()),
                        format_func=lambda x: f"{ANIMAL_DATA[x]['emoji']} {ANIMAL_DATA[x]['name']}",
                        key="correct_class_selector"
                    )
                    
                    if st.button("💾 Sauvegarder avec la bonne classe"):
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
                        
                        st.session_state.show_correction = False
            
            st.markdown("---")
            st.header(f"📚 Fiche Pokédex - {ANIMAL_DATA[predicted_class]['name']}")
            display_pokedex_info(predicted_class)
        
        else:
            st.info("👆 Téléchargez une image pour commencer!")
    
    if hasattr(st.session_state, 'result') and st.session_state.result:
        st.markdown("---")
        st.header("📊 Détail des Probabilités")
        
        fig = create_simple_chart(st.session_state.result['probabilities'])
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("📋 Toutes les probabilités")
        
        prob_data = []
        for class_name, prob in st.session_state.result['probabilities'].items():
            animal = ANIMAL_DATA[class_name]
            prob_data.append({
                'Animal': f"{animal['emoji']} {animal['name']}",
                'Probabilité': f"{prob:.2%}"
            })
        
        prob_data.sort(key=lambda x: float(x['Probabilité'].strip('%')), reverse=True)
        
        cols = st.columns(2)
        mid = len(prob_data) // 2
        
        with cols[0]:
            for item in prob_data[:mid]:
                st.write(f"{item['Animal']}: **{item['Probabilité']}**")
        
        with cols[1]:
            for item in prob_data[mid:]:
                st.write(f"{item['Animal']}: **{item['Probabilité']}**")
    
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.info("🎯 **Précision**: 92.2%")
        st.info("🐾 **Classes**: 10 animaux")
        
        st.markdown("### 🏆 Animaux détectables")
        for class_name, data in ANIMAL_DATA.items():
            st.write(f"{data['emoji']} {data['name']}")
        
        st.markdown("---")
        st.markdown("### 📈 Amélioration continue")
        st.markdown("""
        Vos feedbacks aident à améliorer le modèle:
        - ✅ **Prédiction correcte** → Image ajoutée au dataset
        - ❌ **Prédiction incorrecte** → Correction enregistrée
        - 🔄 **Futur entraînement** → Modèle plus précis
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🐾 Anidex - Dashboard Simple avec Feedback Utilisateur"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
