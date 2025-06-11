import streamlit as st
import cv2
from helper import list_models, load_model   # Your helper.py functions
from tracker import PeopleTracker            # Your tracker.py class
from auth import login                        # Your auth.py function
from alert import AlertManager                # Your alert.py class
import time

st.set_page_config(page_title='People Counter', layout='wide')

def main():
    # User authentication
    if not login():
        st.stop()

    st.sidebar.header('Configuration')

    cam_source = st.sidebar.text_input('Camera index / RTSP URL', '0')
    model_names = list_models()
    selected_model = st.sidebar.selectbox('Detection model', model_names)
    mode = st.sidebar.radio('Mode', ['Eagle‑Eye', 'Lane Counter'])
    capacity = st.sidebar.number_input('Max capacity', min_value=1, value=10)
    restricted_input = st.sidebar.text_input('Restricted classes (comma separated)', '')
    restricted_items = [x.strip().lower() for x in restricted_input.split(',') if x.strip()]

    # Initialize tracker and reset counts if mode changed
    if 'tracker' not in st.session_state or st.session_state.get('tracker_mode') != mode:
        st.session_state['tracker'] = PeopleTracker(mode=mode)
        st.session_state['tracker_mode'] = mode
        st.session_state['in_count'] = 0
        st.session_state['out_count'] = 0
        st.session_state['people_total'] = 0

    # Reset counts button for lane counter mode
    if mode == 'Lane Counter' and st.sidebar.button('Reset lane counts'):
        st.session_state['tracker'].reset_counts()
        st.session_state['in_count'] = 0
        st.session_state['out_count'] = 0
        st.session_state['people_total'] = 0

    # Load model if changed or not loaded
    if ('detector' not in st.session_state or 
        st.session_state.get('model_name') != selected_model):
        detector, names = load_model(selected_model)
        st.session_state['detector'] = detector
        st.session_state['names'] = names
        st.session_state['model_name'] = selected_model

    # Initialize AlertManager with cooldown
    alert_mgr = AlertManager(capacity=capacity, cooldown=10)

    # Initialize alert flashing state variables
    st.session_state.setdefault('custom_alert_message', "")
    st.session_state.setdefault('custom_alert_visible', True)
    st.session_state.setdefault('last_flash_toggle_time', time.time())

    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    metrics_placeholder = st.sidebar.empty()

    # Stream control
    if 'run_stream' not in st.session_state:
        st.session_state['run_stream'] = False

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('Start Stream'):
            st.session_state['run_stream'] = True
    with col2:
        if st.button('Stop Stream'):
            st.session_state['run_stream'] = False
            st.session_state['custom_alert_message'] = ""
            alert_placeholder.empty()

    if st.session_state['run_stream']:
        try:
            cam_idx = int(cam_source) if cam_source.isdigit() else cam_source
            cap = cv2.VideoCapture(cam_idx)
        except Exception as e:
            st.error(f"Error opening camera: {e}")
            st.session_state['run_stream'] = False
            cap = None

        if cap and cap.isOpened():
            st.session_state['tracker'].mode = mode

            while cap.isOpened() and st.session_state['run_stream']:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    st.session_state['run_stream'] = False
                    break

                if 'detector' not in st.session_state or 'names' not in st.session_state:
                    st.warning("Detection model not loaded.")
                    time.sleep(1)
                    continue

                # Run detector on frame
                results = st.session_state['detector'](frame, verbose=False)[0]

                detections = []
                for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    label = st.session_state['names'][int(cls)]
                    detections.append((x1, y1, x2, y2, label))

                # Update tracker with detections
                in_c, out_c, total_c, frame = st.session_state['tracker'].update(detections, frame)

                st.session_state['in_count'] = in_c
                st.session_state['out_count'] = out_c
                st.session_state['people_total'] = total_c

                # Handle alerts (frame annotated internally)
                alert_mgr.handle_capacity(frame, total_c)
                if restricted_items:
                    alert_mgr.handle_restricted(frame, detections, restricted_items)

                # Determine alert message to flash below feed
                alert_msg = ""
                if total_c > capacity:
                    alert_msg = "CAPACITY EXCEEDED!"
                elif restricted_items and any(label.lower() in restricted_items for *_, label in detections):
                    alert_msg = "UNAUTHORIZED ITEM DETECTED!"

                st.session_state['custom_alert_message'] = alert_msg

                # Flashing alert toggle every 0.7 sec
                now = time.time()
                if now - st.session_state['last_flash_toggle_time'] > 0.7:
                    st.session_state['custom_alert_visible'] = not st.session_state['custom_alert_visible']
                    st.session_state['last_flash_toggle_time'] = now

                if st.session_state['custom_alert_message'] and st.session_state['custom_alert_visible']:
                    alert_placeholder.markdown(
                        f"<h3 style='color: red; text-align: center; font-weight: bold;'>🚨 {st.session_state['custom_alert_message']} 🚨</h3>",
                        unsafe_allow_html=True)
                else:
                    alert_placeholder.empty()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels='RGB', use_container_width=True)

                with metrics_placeholder.container():
                    st.metric("People in view (Net)", total_c)
                    if mode == 'Lane Counter':
                        st.metric("Entries", in_c)
                        st.metric("Exits", out_c)

                time.sleep(0.01)

            if cap:
                cap.release()
            video_placeholder.empty()
            alert_placeholder.empty()
            metrics_placeholder.empty()
        else:
            st.error("Failed to open camera.")
            st.session_state['run_stream'] = False
            video_placeholder.empty()
            alert_placeholder.empty()
            metrics_placeholder.empty()
    else:
        video_placeholder.empty()
        alert_placeholder.empty()
        with metrics_placeholder.container():
            st.metric("People in view (Net)", st.session_state.get('people_total', 0))
            if mode == 'Lane Counter':
                st.metric("Entries", st.session_state.get('in_count', 0))
                st.metric("Exits", st.session_state.get('out_count', 0))


if __name__ == '__main__':
    main()