import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from collections import deque
import tempfile
import os

# Function to process and save the video
def process_video(uploaded_video, threshold=0.2):
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    temp_video.close()  # Close the file to ensure it's not being used

    try:
        model = load_model('modelnew.h5')
        Q = deque(maxlen=128)
        vs = cv2.VideoCapture(temp_video.name)
        writer = None
        (W, H) = (None, None)
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        violent_frames = 0
        frame_counter = 0

        progress_bar = st.progress(0)
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')

        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break

            frame_counter += 1

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128)).astype("float32")
            frame = frame.reshape(128, 128, 3) / 255

            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            Q.append(preds)

            results = np.array(Q).mean(axis=0)
            label = (preds > 0.50)[0]

            if label:
                violent_frames += 1

            text_color = (0, 255, 0) if not label else (0, 0, 255)
            text = "Violence: {}".format(label)
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_file.name, fourcc, 30, (W, H), True)

            writer.write(output)
            progress_bar.progress(frame_counter / total_frames)

        writer.release()
        vs.release()

        violent_percentage = violent_frames / total_frames

        if violent_percentage >= threshold:
            result_message = f"This video contains violence. ({violent_percentage:.2%} of frames were violent)"
            result_type = 'warning'
        else:
            result_message = f"This video does not contain violence. ({violent_percentage:.2%} of frames were violent)"
            result_type = 'success'

    finally:
        os.remove(temp_video.name)

    return output_file.name, result_message, result_type

# Streamlit app interface
st.set_page_config(page_title="Violence Detection in Videos", layout="wide")
st.title("🎥 Violence Detection in Videos")

# Sidebar for additional settings
st.sidebar.title("Settings")
threshold = st.sidebar.slider("Violence Detection Threshold (%)", min_value=10, max_value=50, value=20, step=5) / 100

if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("Processing your video. This may take a while depending on the video length...")

    # Only process if it hasn't been processed already
    if st.session_state.processed_video_path is None:
        processed_video_path, result_message, result_type = process_video(uploaded_file, threshold)
        st.session_state.processed_video_path = processed_video_path
        st.session_state.result_message = result_message
        st.session_state.result_type = result_type
    else:
        processed_video_path = st.session_state.processed_video_path
        result_message = st.session_state.result_message
        result_type = st.session_state.result_type

    # Display the result message
    if result_type == 'warning':
        st.warning(result_message)
    else:
        st.success(result_message)

    st.success("Video processing completed!")
    
    # Provide a download link for the processed video
    with open(processed_video_path, 'rb') as f:
        st.download_button('Download Processed Video', f, file_name='processed_video.avi')

    # Display the processed video
    # st.video(processed_video_path)
else:
    st.warning("Please upload a video to start processing.")
