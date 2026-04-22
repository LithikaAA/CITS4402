"""
CITS4402 Computer Vision Project - Face Detection and Matching
Group Members:
  - Jason Rarey - 22681384
  - Lithika Senthil Kumar - 23812347

Description:
    A GUI application that performs face detection, facial landmark detection,
    face alignment, and identity clustering on images.
"""
 
import os
import sys
import time
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
 
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Optional deep-learning imports ────────────────────────────────────────────
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
 
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
 
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
 
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False