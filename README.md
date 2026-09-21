# Reconocimiento-de-se-as-b-sicas-del-alfabeto-LSM
Sistema de visión por computadora para reconocimiento y traducción de Lengua de Señas en tiempo real. Evaluación comparativa entre una CNN (Sign Language MNIST) y Random Forest sobre landmarks articulares con MediaPipe Hands, desplegado interactivamente en Streamlit.
# 🤟 Sistema de Reconocimiento de Lengua de Señas (LSM / ASL) en Tiempo Real

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-teal)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-yellow)](https://www.gradio.app/)

## 📌 Descripción del Proyecto

Este proyecto desarrolla e implementa un sistema integral de **visión artificial y aprendizaje automático** enfocado en la clasificación estática de señas del alfabeto de Lengua de Señas (LSM / ASL). 

El sistema implementa un enfoque de **evaluación comparativa dual**:
1. **Red Neuronal Convolucional (CNN):** Entrenada con el dataset *Sign Language MNIST* para inferencia directa sobre representaciones tensoriales de $28 \times 28$ píxeles en escala de grises.
2. **Clasificador Basado en Coordenadas Esqueléticas:** Extracción y normalización de 21 landmarks articulares tridimensionales por mano (vector de 126 características) mediante *MediaPipe Hands*, clasificados con un ensamble de *Random Forest*.

Incluye despliegues interactivos desacoplados mediante **Streamlit**
