# 🐾 Anidex - Attrape-les tous (en image) !

Une application de classification d'animaux par intelligence artificielle inspirée de l'univers Pokémon.
![image](https://github.com/user-attachments/assets/72cfac24-f3a0-491b-964f-61e0209b7e85)
![image](https://github.com/user-attachments/assets/c320bade-2778-4a63-acd7-dd9a81c14ba5)
![image](https://github.com/user-attachments/assets/eeeaf660-2502-4eed-941f-f0f7adcb28a7)
![image](https://github.com/user-attachments/assets/7b1ce775-7939-4538-be4f-652875fa0488)

## 📋 Description

Anidex est une application web interactive qui utilise l'intelligence artificielle pour identifier automatiquement les animaux présents dans vos images. Grâce à un modèle de deep learning basé sur MobileNetV2 et le transfer learning, l'application peut reconnaître **10 espèces d'animaux** avec une grande précision.

### 🎯 Fonctionnalités principales

- **Classification automatique** : Upload d'image et prédiction instantanée
- **Niveau de confiance** : Pourcentage de certitude de la prédiction
- **Interface Pokédex** : Fiches détaillées des animaux détectés
- **Système de feedback** : Validation des prédictions pour améliorer le modèle
- **Visualisations avancées** : Graphiques des probabilités et statistiques
- **Design intuitif** : Interface web moderne et responsive

### 🦁 Animaux reconnus

L'application peut identifier les animaux suivants :
- 🐘 **Éléphant** - Majesteux mammifère des savanes
- 🐄 **Vache** - Bovin domestique
- 🐑 **Mouton** - Ovin producteur de laine
- 🐕 **Chien** - Fidèle compagnon de l'homme
- 🐎 **Cheval** - Noble équidé
- 🕷️ **Araignée** - Arachnide tisseuse
- 🦋 **Papillon** - Insecte aux ailes colorées
- 🐱 **Chat** - Félin domestique
- 🐔 **Poulet** - Volaille de basse-cour
- 🐿️ **Écureuil** - Petit rongeur agile

## 🚀 Aperçu de l'application

### Interface de classification
L'interface principale permet d'uploader une image par glisser-déposer et affiche instantanément la prédiction avec le niveau de confiance.

### Fiche Pokédex
Chaque animal détecté dispose d'une fiche détaillée contenant :
- **Habitat** : Environnement naturel
- **Taille** : Dimensions moyennes
- **Poids** : Masse corporelle
- **Régime alimentaire** : Habitudes nutritionnelles
- **Caractéristiques** : Traits distinctifs
- **Faits intéressants** : Anecdotes surprenantes

### Analyses statistiques
- Graphique en barres des top 5 prédictions
- Vue détaillée de toutes les probabilités

## 🧠 Technologie

### Architecture du modèle
- **Base** : MobileNetV2 pré-entraîné sur ImageNet
- **Technique** : Transfer Learning avec fine-tuning
- **Optimisation** : Callbacks intelligents (EarlyStopping, ReduceLROnPlateau)
- **Augmentation** : Data augmentation pour améliorer la robustesse

### Stack technique
- **Backend ML** : TensorFlow/Keras
- **Interface** : Streamlit
- **Traitement d'images** : PIL, numpy
- **Visualisations** : matplotlib, seaborn
- **Documentation** : Jupyter Notebooks

## 📊 Performances

Le modèle atteint une **précision de 92.3%** sur l'ensemble de test, avec des prédictions particulièrement fiables pour les grands mammifères comme les éléphants (>99% de confiance typique).

## 🏗️ Structure du projet

```
Anidex/
├── data/                        # Dataset organisé par classes
│   ├── butterfly/               # Images de papillons
│   ├── cat/                     # Images de chats
│   ├── chicken/                 # Images de poulets
│   ├── cow/                     # Images de vaches
│   ├── dog/                     # Images de chiens
│   ├── elephant/                # Images d'éléphants
│   ├── horse/                   # Images de chevaux
│   ├── sheep/                   # Images de moutons
│   ├── spider/                  # Images d'araignées
│   └── squirrel/                # Images d'écureuils
├── models/                      # Notebooks et modèles ML
│   └── MobileNetV2.ipynb        # Notebook d'entraînement du modèle
├── src/                         # Code source principal
│   ├── dashboard.py             # Application Streamlit principale
│   ├── data_preprocessing.ipynb # Notebook de prétraitement
│   └── model.py                 # Fonctions du modèle ML
├── model_classification.h5      # Modèle entraîné sauvegardé
├── requirements.txt             # Dépendances Python
├── .gitignore                   # Fichiers à ignorer par Git
└── README.md                    # Documentation du projet
```

## 📖 Utilisation

### Interface web

1. Accédez à l'application Streamlit
2. Cliquez sur "Browse files" ou glissez-déposez votre image
3. Attendez la prédiction automatique
4. Consultez la fiche Pokédex de l'animal détecté
5. Validez ou corrigez la prédiction via les boutons de feedback

### Formats supportés
- **Images** : PNG, JPG, JPEG
- **Taille maximum** : 200MB
- **Résolution** : Optimale entre 224x224 et 1024x1024 pixels

## 🛠️ Installation

### Prérequis

- **Système d'exploitation** : Ubuntu 18.04+ ou Debian 10+
- **Make** : Pour automatiser l'installation
- **Accès sudo** : Pour installer Python 3.11
- **Git** : Pour cloner le projet

### Installation rapide

1. **Cloner le repository**
   ```bash
   git clone https://github.com/NotJuz0o/Anidex.git
   cd Anidex
   ```

2. **Installation automatique**
   ```bash
   # Voir les commandes disponibles
   make help
   
   # Installation complète (Python 3.11 + dépendances + environnement virtuel)
   make install
   ```

3. **Lancer l'application**
   ```bash
   make run
   ```

4. **Accéder à l'interface**
   
   Ouvrez votre navigateur et allez sur : `http://localhost:8501`

### Installation manuelle (optionnelle)

Si vous préférez installer manuellement :

```bash
# 1. Installer Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# 2. Créer l'environnement virtuel
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer l'application
python -m streamlit run src/dashboard.py
```

### Commandes Makefile disponibles

```bash
make help        # Afficher toutes les commandes disponibles
make install     # Installation complète du projet
make dataset     # Exécuter le prétraitement des données
make model       # Entraîner le modèle MobileNetV2
make run         # Lancer l'application Streamlit
make clean       # Nettoyer les caches Python uniquement
make fclean      # Nettoyage complet (venv + cache + modèles générés)
```

### Problèmes courants

**Erreur de permissions :**
```bash
# S'assurer que make a les permissions
chmod +x Makefile
```

**Port 8501 déjà utilisé :**
```bash
# Lancer sur un autre port
.venv/bin/python -m streamlit run src/dashboard.py --server.port 8502
```

### Vérification de l'installation

Une fois l'installation terminée, vous devriez voir :
- ✅ Interface Streamlit accessible sur `http://localhost:8501`
- ✅ Upload d'image fonctionnel
- ✅ Prédictions en temps réel
- ✅ Fiches Pokédex complètes

### Désinstallation

Pour supprimer complètement le projet :
```bash
make fclean         # Nettoyage complet (recommandé)
cd ..
rm -rf Anidex       # Supprimer le dossier du projet
```

**Ou nettoyage partiel :**
```bash
make clean          # Garder l'environnement, supprimer juste les caches
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Le système de feedback intégré permet d'améliorer continuellement le modèle.

### Feedback utilisateur
- Utilisez les boutons "✅ Oui, c'est correct!" et "❌ Non, c'est faux" après chaque prédiction
- Vos retours sont automatiquement sauvegardés pour améliorer le modèle
- Les données anonymisées contribuent à l'entraînement futur

## 📈 Roadmap

### Version actuelle (1.0)
- ✅ Classification de 10 animaux
- ✅ Interface Streamlit complète
- ✅ Système de feedback
- ✅ Fiches Pokédex

### Versions futures
- 🔄 **v1.1** : Géolocalisation des animaux
- 🔄 **v1.2** : API REST pour intégrations
- 🔄 **v2.0** : Classification multi-animaux par image

## 👨‍💻 Auteur

**NotJuz0o** - Etudiant EPITECH Promo 2029  
*Projet réalisé dans le cadre du développement de compétences en machine learning*

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🎯 Objectifs pédagogiques

Ce projet couvre l'ensemble du cycle de développement ML :
- Collecte et analyse de données
- Prétraitement d'images
- Transfer learning et fine-tuning
- Déploiement d'application ML
- Interface utilisateur interactive
- Amélioration continue par feedback

---

**🌟 Testez dès maintenant Anidex et découvrez la magie de l'IA appliquée à la reconnaissance animale !**
