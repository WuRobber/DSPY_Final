import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

@st.cache_resource
def face_cascade_load(path):
    return cv2.CascadeClassifier(path)

def FaceDetection(image):
    face_cascade = face_cascade_load('./haarcascade_frontalface_default.xml')
    scaleFactor = 1.02
    minNeighbors = 4
    faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
    face_cascade=None
    return image, faces

def CropImage(image):
    _, faces = FaceDetection(image)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face
    else:
        return None

def ResizeImage(image, target_size):
    original_size = (image.shape[1], image.shape[0])
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    resized_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    resized_image = resized_image[:target_size[1], :target_size[0]]
    return resized_image


def GrayImage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return gray_image


def ReadImage(img):
    target_size = (256, 256)
    total_faces = 0
    img = ResizeImage(img, target_size)
    img = GrayImage(img)
    img, _ = FaceDetection(img)
    _, faces = FaceDetection(img)
    total_faces += len(faces)
    cropped_face = CropImage(img)
    return cropped_face


def showSaliencyMap(img, model, pred_position):
    images = tf.Variable(img, dtype=float)
    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        with pred_position:
            st.write("")
            st.write("")
            st.write("Prediction:")
            st.write(f"{pred.numpy()[0][0]*100:.2f} % like female")
            st.write(f"{pred.numpy()[0][1]*100:.2f} % like male")
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
        pred = None
    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    # normalize to range between 0 and 1
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    img_np = img.numpy()
    img_3d = img_np.reshape(img_np.shape[1], img_np.shape[2], img_np.shape[3])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img_3d)
    i = axes[1].imshow(grad_eval, cmap="jet", alpha=1)
    fig.colorbar(i)
    return fig


st.markdown("<h1 style='text-align: center; color: white;'>Sexual Orientation "
            "Classifier</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your Image here...",
                                 type=['jpg', 'png', 'jpeg'])
@st.cache_resource
def loadmodel():
    return tf.keras.models.load_model('2.h5', compile=False)

if uploaded_file is None:
    st.markdown("<img src='https://media1.tenor.com/m/uZv4t9KXvCMAAAAC/"
                "rainbow-cat-rainbow.gif' "
                "style='display: block; margin-left: auto; "
                "margin-right: auto; width: 50%;'/>",
                unsafe_allow_html=True)

if uploaded_file is not None:
    left_co, right_co = st.columns(2)
    with left_co:
        w, h = Image.open(uploaded_file).size
        st.image(uploaded_file, width=300*w//h)
    original_image = Image.open(uploaded_file)
    original_image = np.array(original_image)
    face = ReadImage(original_image)
    if face is not None:
        pass
        face = face.reshape(1, face.shape[0], face.shape[1], 3)
        face = tf.image.resize(face, [224, 224])
        face = face/255.0

        # model
        model = loadmodel()
        fig = showSaliencyMap(face, model, right_co)
        face = None
        on = st.toggle("Show Saliency Map")
        
        if on:
            st.pyplot(fig)

    else:
        st.markdown("<h3 style='text-align: center; color: white;'>No face "
                    "detected. Please upload new one</h3>",
                    unsafe_allow_html=True)