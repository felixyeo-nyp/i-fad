# Flask-related imports #
import flask_mail
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, send_file, session
import socket
import struct
from scapy.all import *
from scapy.layers.inet import IP, TCP
from sympy import false
from wtforms import Form, StringField, RadioField, SelectField, TextAreaField, validators, ValidationError, PasswordField
import sys
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import shelve, re
from flask_wtf import FlaskForm
from wtforms.validators import email
import win32serviceutil
import win32service
import win32event
import servicemanager
import subprocess
from Forms import configurationForm, emailForm, LoginForm, RegisterForm,updatepasswordForm, MFAForm, FeedbackForm,updateemailrole, forgetpassword , ipForm
from flask_mail import Mail, Message
import random
import secrets
import webbrowser
# Object detection & processing-related imports #
import torch
import torchvision
import cv2
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Datetime-related imports #
import datetime, time
import threading
from datetime import datetime, timedelta
from flask_apscheduler import APScheduler
import numpy as np
import pytz

# File-reader-related imports #
import openpyxl
from openpyxl.styles import Font

# OOP-related imports #
from OOP import *
import os

# Security-related imports #
from functools import wraps
from flask import abort

# System-related imports #
import socket
import psutil

# Back-end codes for object detection & processing #
if torch.cuda.is_available():
    print('you are using gpu to process the video camera')
else:
    print('no gpu is found in this python environment. using cpu to process')

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        self.condition = threading.Condition()
        self.is_running = False
        self.frame = None
        self.pellets_num = 0
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.is_running = True
        super().start()

    def stop(self, timeout=None):
        self.is_running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.is_running:
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            with self.condition:
                self.frame = img if rv else None
                self.condition.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, sequence_number=None, timeout=None):
        with self.condition:
            if wait:
                # If sequence_number is not provided, get the next sequence number
                if sequence_number is None:
                    sequence_number = self.pellets_num + 1

                if sequence_number < 1:
                    sequence_number = 1

                if (sequence_number) > 0:
                    self.pellets_num = sequence_number

                # Wait until the latest frame's sequence number is greater than or equal to sequence_number
                rv = self.condition.wait_for(lambda: self.pellets_num >= sequence_number, timeout=timeout) # if there is a pellets. should get "true"
                if not rv:
                    return (self.pellets_num, self.frame)  # Return the latest frame if timeout occurs
            return (self.pellets_num, self.frame)  # Return the latest frame

# define the id "1" for pellets
# do note that in the pth file, the pellet id also is 1
class_labels = {
    1: 'Pellets',
}

# pth file where you have defined on roboflow
model_path = './best_model5.pth'
latest_processed_frame = None  # Stores the latest processed frame
stop_event = threading.Event()  # Event to stop threads gracefully
freshest_frame = None

# Initialize variables for feeding logic
feeding = False
feeding_timer = None
showing_timer = None
line_chart_timer = None
object_count = {1: 0}
frame_data = {
    'object_count': {1: 0},  # Initialize with default values for object count
    'bounding_boxes': []  # List to store bounding boxes for the current frame
}

# Initialize locks for shared variables
latest_processed_frame_lock = threading.Lock()
frame_data_lock = threading.Lock()
object_count_lock = threading.Lock()
freshest_frame_lock = threading.Lock()
shelve_lock = threading.Lock()

def create_model(num_classes, pretrained=False, coco_model=False):
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    if not coco_model:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Function to load the custom-trained model from the .pth file
def load_model(model_path, num_classes):
    model = create_model(num_classes=num_classes, pretrained=False, coco_model=False)
    model.roi_heads.detections_per_img = 500
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# Assume these methods to load model and settings are defined
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model = load_model(model_path, num_classes=2)
model.to(device)
model.eval()

def check_feeding_time():
    global feeding, feeding_timer, showing_timer, line_chart_timer
    db = shelve.open('settings.db', 'w')
    Time_Record_dict = db['Time_Record']
    db.close()

    setting = Time_Record_dict.get('Time_Record_Info')

    hours = setting.get_first_timer()[:2]
    minutes = setting.get_first_timer()[2:]
    hours1 = setting.get_second_timer()[:2]
    minutes1 = setting.get_second_timer()[2:]

    first_feeding_time = int(hours)
    first_feeding_time_min = int(minutes)
    second_feeding_time = int(hours1)
    second_feeding_time_min = int(minutes1)
    while not stop_event.is_set():
        current_time = datetime.now()
        # Check if it's time for feeding (at the exact second)
        if (current_time.hour == first_feeding_time or current_time.hour == second_feeding_time) and \
           (current_time.minute == first_feeding_time_min or current_time.minute == second_feeding_time_min) and \
           current_time.second == 0:
            feeding = True
            feeding_timer = None
            showing_timer = None
            line_chart_timer = time.time()
        else:
            feeding = False  # If it's not time for feeding, set feeding to False
        time.sleep(1)  # Check every second

def capture_frames():
    while not stop_event.is_set():
        try:
            with shelve.open('IP.db', 'c') as ip_db:
                cam = ip_db.get('IP', {})
                ip = cam.get('camera_ip')
                user = cam.get('amcrest_username')
                pw = cam.get('amcrest_password')

                if not all([ip, user, pw]):
                    print("Camera config incomplete, waiting...")
                    time.sleep(5)
                    continue  # Try again in a few seconds

                rtsp_url = f"rtsp://{user}:{pw}@{ip}:554/cam/realmonitor?channel=1&subtype=0"
                print(f"Connecting to RTSP stream: {rtsp_url}")
        except Exception as e:
            print(f"Error reading IP.db: {e}")
            time.sleep(5)
            continue

        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print("Warning: Cannot open video stream. Retrying in 10s...")
            time.sleep(10)
            continue

        global freshest_frame
        with freshest_frame_lock:
            freshest_frame = FreshestFrame(cap)

        global latest_processed_frame
        while not stop_event.is_set():
            if not cap.isOpened():
                break

            sequence_num, frame = freshest_frame.read(wait=True)
            if frame is not None:
                with latest_processed_frame_lock:
                    latest_processed_frame = frame

            time.sleep(0.03)

        cap.release()

def validate_config_thread():
    global latest_valid_settings, latest_set_ip_settings
    print("Starting validation thread for settings and IP...")

    fallback_setting = Settings('0830', '1800', 1, 100, 1, 60)
    fallback_ip = {
        "source": "192.168.0.65",
        "destination": "192.168.0.18",
        "camera_ip": "192.168.0.52",
        "amcrest_username": "admin",
        "amcrest_password": "fyp2025Fish34535"
    }
    fallback_email = Email('iatfadteam@gmail.com', 'iatfadteam@gmail.com', 'pmtu cilz uewx xqqi', 3)

    while not stop_event.is_set():
        try:
            # Check IP.db
            ip_config_updated = False
            with shelve_lock:
                with shelve.open('IP.db', 'c') as ip_db:
                    ip_data = ip_db.get("IP", {})
                    required_keys = ["source", "destination", "camera_ip", "amcrest_username", "amcrest_password"]

                    if not all(k in ip_data and ip_data[k] for k in required_keys):
                        ip_db["IP"] = latest_set_ip_settings if latest_set_ip_settings else fallback_ip
                        ip_config_updated = True
                        print("[VALIDATE] Fallback IP configuration written.")

            # Check settings.db
            with shelve_lock:
                with shelve.open('settings.db', 'c') as settings_db:
                    time_record = settings_db.get('Time_Record', {}).get('Time_Record_Info')
                    if not time_record:
                        if latest_valid_settings:
                            settings_db['Time_Record'] = {'Time_Record_Info': latest_valid_settings}
                            print("[VALIDATE] Restored missing settings from latest valid config.")
                        else:
                            settings_db['Time_Record'] = {'Time_Record_Info': fallback_setting}
                            print("[VALIDATE] Fallback settings initialized.")

                    if 'Port' not in settings_db:
                        settings_db['Port'] = random.randint(50001, 65535)
                        print("[VALIDATE] Missing Port generated.")

                    if 'Generate_Status' not in settings_db:
                        settings_db['Generate_Status'] = False
                        print("[VALIDATE] Generate_Status set to False.")

                    if 'Email_Data' not in settings_db:
                        settings_db['Email_Data'] = {'Email_Info': fallback_email}
                        print("[VALIDATE] Fallback Email_Data written.")

                    if ip_config_updated or 'syn_ack_seq' not in settings_db or 'syn_ack_ack' not in settings_db:
                        try:
                            ip_data = None
                            with shelve_lock:
                                with shelve.open('IP.db', 'r') as ip_db:
                                    ip_data = ip_db.get("IP", {})

                            source = ip_data.get("source")
                            destination = ip_data.get("destination")

                            port = settings_db.get('Port')
                            if source and destination and port:
                                start_syn_packet_thread(source, destination, port, 50000)
                                send_udp_packet(source, "255.255.255.255", 60000, b"\x00\x00\x00\x00\x00")
                                print("[VALIDATE] SYN/ACK packets sent after IP config.")
                            else:
                                print("[VALIDATE WARNING] Could not send SYN/ACK packets due to missing source/destination/port.")

                        except Exception as e:
                            print(f"[VALIDATE ERROR] Failed to generate SYN/ACK packets: {e}")

        except Exception as e:
            print(f"[VALIDATE ERROR] {e}")

        time.sleep(2)

def video_processing_loop():
    global freshest_frame, object_count, frame_data, latest_processed_frame
    while True:
        try:
            with shelve_lock:
                with shelve.open('settings.db', 'r') as db:
                    Time_Record_dict = db.get('Time_Record', {})
                    setting = Time_Record_dict.get('Time_Record_Info')
        except Exception as e:
            print(f"[ERROR] Failed to read from settings.db: {e}")
            time.sleep(1)
            continue
        
        confidence = float(setting.get_confidence()) / 100
        with freshest_frame_lock:
            if freshest_frame is None:
                time.sleep(0.1)
                continue
            cnt, frame = freshest_frame.read(sequence_number=object_count[1] + 1)
            if frame is None:
                time.sleep(0.1)
                continue

        # Inference
        img_tensor = torchvision.transforms.ToTensor()(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(img_tensor)

        temp_object_count = {1: 0}
        bounding_boxes = []
        for i in range(len(predictions[0]['labels'])):
            label = predictions[0]['labels'][i].item()
            if label == 1 and predictions[0]['scores'][i].item() > confidence:
                box = predictions[0]['boxes'][i].cpu().numpy().astype(int)
                temp_object_count[label] += 1
                bounding_boxes.append((box, label, predictions[0]['scores'][i].item()))
        
        with object_count_lock:
            object_count[1] = temp_object_count[1]

        try:
            with shelve_lock:
                with shelve.open('currentcount.db', 'c') as db2:
                    db2['object_count'] = temp_object_count[1]
        except Exception as e:
            print(f"[ERROR] Failed to write to currentcount.db: {e}")

        for box, label, score in bounding_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'Pellet: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        with frame_data_lock:
            frame_data['object_count'] = temp_object_count
            frame_data['bounding_boxes'] = bounding_boxes
        with latest_processed_frame_lock:
            latest_processed_frame = frame

def feeding_scheduler_loop():
    global feeding, feed_start_time, total_count
    feeding = False
    feed_start_time = None
    total_count = 0
    last_check_index = -1

    port = server_isn = server_ack = source_ip = destination_ip = None

    while True:
        try:
            with shelve_lock:
                with shelve.open('settings.db', 'r') as db:
                    Time_Record_dict = db.get('Time_Record', {})
                    setting = Time_Record_dict.get('Time_Record_Info')
        except Exception as e:
            print(f"[ERROR] Feeding loop failed to read settings: {e}")
            time.sleep(1)
            continue

        tz = pytz.timezone("Asia/Singapore")  # Use your actual timezone
        now = datetime.now(tz)

        try:
            first_timer = setting.get_first_timer()
            second_timer = setting.get_second_timer()
            scheduled1 = now.replace(hour=int(first_timer[:2]), minute=int(first_timer[2:]))
            scheduled2 = now.replace(hour=int(second_timer[:2]), minute=int(second_timer[2:]))
        except Exception as e:
            print(f"[ERROR] Failed to parse feeding times: {e}")
            time.sleep(1)
            continue

        try:
            interval = max(int(setting.get_interval_seconds()), 10)
            duration = int(setting.get_seconds())
            check_count = duration // interval
        except Exception as e:
            print(f"[ERROR] Invalid interval or duration: {e}")
            time.sleep(1)
            continue

        with object_count_lock:
            current_count = object_count[1]
        threshold = (int(setting.get_pellets()) * 100) // 10

        # Check if feeding should start
        if not feeding and (now == scheduled1 or now == scheduled2):
            print(f"[FEED TRIGGER] Starting feeding session at {now.strftime('%H:%M:%S')}")
            if now == scheduled1:
                active_feeding_time_str = first_timer
            elif now == scheduled2:
                active_feeding_time_str = second_timer
            try:
                with shelve_lock:
                    with shelve.open('IP.db', 'r') as ip_db, shelve.open('settings.db', 'r') as db:
                        port = db.get('Port')
                        server_isn = db.get('syn_ack_seq')
                        server_ack = db.get('syn_ack_ack')
                        ip_data = ip_db.get('IP', {})
                        source_ip = ip_data.get('source')
                        destination_ip = ip_data.get('destination')

                feeding = True
                feed_start_time = time.time()
                last_check_index = -1
                with object_count_lock:
                    total_count += object_count[1]
            except Exception as e:
                print(f"[ERROR] Failed to retrieve database settings: {e}")

        # If in feeding mode, manage interval checks
        if feeding and feed_start_time is not None:
            elapsed = int(time.time() - feed_start_time)
            with object_count_lock:
                    total_count += object_count[1]
            current_check_index = elapsed // interval

            print(f"[DEBUG] Elapsed: {elapsed}s | Check {current_check_index}/{check_count} | Pellet count: {current_count} | Re-feed threshold: {threshold}")

            if current_check_index > last_check_index and current_check_index < check_count:
                last_check_index = current_check_index

                with object_count_lock:
                    current_count = object_count[1]

                if current_count < threshold:
                    try:
                        print("[RE-FEED] Pellet low, continuing feed")
                        try:
                            with shelve_lock:
                                with shelve.open('IP.db', 'r') as ip_db, shelve.open('settings.db', 'r') as db:
                                    port = db.get('Port')
                                    server_isn = db.get('syn_ack_seq')
                                    server_ack = db.get('syn_ack_ack')
                                    ip_data = ip_db.get('IP', {})
                                    source_ip = ip_data.get('source')
                                    destination_ip = ip_data.get('destination')
                            print("[RE-FEED] Pellet low, continuing feed")
                            start_send_manual_feed(port, server_isn, int(server_ack)+1, source_ip, destination_ip)
                            print("[RE-FEED] Feed continued successfully")
                        except Exception as e:
                            print(f"[ERROR] Failed to continue feed: {e}")
                        print("[RE-FEED] Feed continued successfully")
                    except Exception as e:
                        print(f"[ERROR] Failed to continue feed: {e}")
                else:
                    try:
                        print("[RE-FEED] Pellet level sufficient. Stopping feed.")
                        try:
                            with shelve_lock:
                                with shelve.open('IP.db', 'r') as ip_db, shelve.open('settings.db', 'r') as db:
                                    port = db.get('Port')
                                    server_isn = db.get('syn_ack_seq')
                                    server_ack = db.get('syn_ack_ack')
                                    ip_data = ip_db.get('IP', {})
                                    source_ip = ip_data.get('source')
                                    destination_ip = ip_data.get('destination')
                            print("[RE-FEED] Pellet level sufficient. Stopping feed.")
                            stop_send_manual_feed(port, server_isn, int(server_ack), source_ip, destination_ip)
                            print("[RE-FEED] Feed stopped successfully")
                        except Exception as e:
                            print(f"[ERROR] Failed to stop feed early: {e}")
                        print("[RE-FEED] Feed stopped successfully")
                    except Exception as e:
                        print(f"[ERROR] Failed to stop feed early: {e}")

            # End feeding session if duration complete
            if elapsed >= duration:
                print("[FEED END] Duration complete. Ending session.")
                try:
                    try:
                        with shelve_lock:
                            with shelve.open('IP.db', 'r') as ip_db, shelve.open('settings.db', 'r') as db:
                                port = db.get('Port')
                                server_isn = db.get('syn_ack_seq')
                                server_ack = db.get('syn_ack_ack')
                                ip_data = ip_db.get('IP', {})
                                source_ip = ip_data.get('source')
                                destination_ip = ip_data.get('destination')
                        stop_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)
                    except Exception as e:
                        print(f"[ERROR] Failed to stop final feed: {e}")
                except Exception as e:
                    print(f"[ERROR] Failed to stop final feed: {e}")
                feeding = False
                feed_start_time = None
                if active_feeding_time_str:
                    save_chart_data(total_count, active_feeding_time_str)
                    print(f"[FEED END] Feeding session ended at {now.strftime('%H:%M:%S')} with {total_count} pellets.")
                    total_count = 0

        time.sleep(1)

def save_chart_data(total_count, feeding_time_str):
    try:
        hour = int(feeding_time_str[:2])
        session = 'Morning' if 6 <= hour < 12 else 'Evening'

        with shelve_lock:
            with shelve.open('mock_chart_data.db', 'c') as db:
                date_str = datetime.today().strftime("%Y-%m-%d")

                if date_str not in db:
                    db[date_str] = {}

                day_data = db[date_str]
                day_data[session] = day_data.get(session, 0) + int(total_count)
                day_data['Total'] = day_data.get('Total', 0) + int(total_count)

                db[date_str] = day_data

        print(f"[SAVE] {session} session ({feeding_time_str}) recorded with {total_count} pellets.")
    except Exception as e:
        print(f"[ERROR] Failed to save feeding data: {e}")

@login_required
def generate_frames():
    global latest_processed_frame, frame_data
    count = 1
    while not stop_event.is_set():
        if count // 60 == 0:
            db = shelve.open('settings.db', 'c')
            if not db.get('Generate_Status', True):
                print("stopped generating")
                break
            db.close()

        with latest_processed_frame_lock:
            if latest_processed_frame is None:
                # Create a black frame fallback if no frame available
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_to_use = black_frame
            else:
                frame_to_use = latest_processed_frame.copy()

        # Overlay info only if it's not the black fallback frame
        if latest_processed_frame is not None:
            with frame_data_lock:
                for label, count in frame_data['object_count'].items():
                    text = f'{class_labels[label]} Count: {count}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    text_position = (frame_to_use.shape[1] - text_size[0] - 10, 30 * (label + 1))
                    cv2.putText(frame_to_use, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 255, 255), 2)

                for box, label, score in frame_data['bounding_boxes']:
                    cv2.rectangle(frame_to_use, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame_to_use, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame_to_use)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            time.sleep(0.1)

        count += 1
        time.sleep(0.03)  # Adjust frame rate if necessary

# Web application #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Initialize Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'iatfadteam@gmail.com'
app.config['MAIL_DEFAULT_SENDER'] = ('Admin', 'iatfadteam@gmail.com')
app.config['MAIL_PASSWORD'] = 'pmtu cilz uewx xqqi'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# Dictionaries #
#j = datetime.datetime.now()
# print(j)
Time_Record_dict = {}
Line_Chart_Data_dict = {}
Email_dict = {}

# User model
class User(UserMixin):
    def __init__(self, username, email, password, role, status="Active"):
        self.id = username  # Use username as ID for simplicity
        self.username = username
        self.email = email
        self.password = password
        self.role = role
        self.status = status

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'role' not in session or session['role'] != role:
                username = session.get('username')
                if username:  # Ensure the user is logged in
                    with shelve.open('users.db', writeback=True) as db:
                        user = db.get(username)
                        if user and user["status"] == "Active":
                            user["status"] = "Breached"  # Change status to Breached
                            db[username] = user  # Save the updated user back to the database
                return redirect(url_for("breached"))  # Redirect to Breached page
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/Breached", methods=['GET'])
def breached():
    session.clear()
    return render_template("Breached.html")

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    try:
        username = session['username']
        with shelve.open('users.db') as db:
            user_data = db.get(user_id)
            user = db[username]['username']
            if user_data:
                return User(user_data['username'], user_data['email'], user_data['password'], user_data['role'], user_data.get('status', 'Active'))
    except Exception as e:
        app.logger.error(f"Error loading user {user_id}: {e}")
    return None

# Default route to redirect to login page
@app.route('/')
@login_required
def index():
    return redirect(url_for('logout'))

# Function to open shelve safely
def open_shelve(filename, mode='c'):
    try:
        shelf = shelve.open(filename, mode)
        return shelf
    except Exception as e:
        print(f"Error opening shelve: {e}")
        return None

# Routes for Registration and Login using shelve
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()  # Create an instance of RegisterForm

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        confirm_password = form.confirm_password.data
        role = form.role.data

        # Check if the username or email already exists in the database
        with shelve.open('users.db', 'c') as db:
            username_exists = username in db
            email_exists = any(user_data['email'] == email for user_data in db.values())

            if username_exists:
                flash('Username or email already in use', 'danger')
            elif email_exists:
                flash('Username or email already in use', 'danger')
            else:
                # If neither username nor email is already registered, proceed with registration
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username, email, hashed_password, role, status="Active")
                db[username] = {'username': new_user.username, 'email': new_user.email, 'password': new_user.password, 'role': new_user.role, 'status':new_user.status}
                flash('You are now registered and can log in', 'success')
                return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/register2', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def register2():
    form = RegisterForm()  # Create an instance of RegisterForm

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        confirm_password = form.confirm_password.data
        role = form.role.data

        # Check if the username or email already exists in the database
        with shelve.open('users.db', 'c') as db:
            username_exists = username in db
            email_exists = any(user_data['email'] == email for user_data in db.values())

            if username_exists:
                flash('Username or email already in use', 'danger')
            elif email_exists:
                flash('Username or email already in use', 'danger')
            else:
                # If neither username nor email is already registered, proceed with registration
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username, email, hashed_password, role, status="Active")
                db[username] = {'username': new_user.username, 'email': new_user.email, 'password': new_user.password, 'role': new_user.role, 'status':new_user.status}
                return redirect(url_for('retrieve_users'))

    return render_template('register2.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        try:
            with shelve.open('users.db', 'c') as db:
                if username in db:
                    user = db[username]  # Get the user data as a dictionary
                    role = user['role']
                    stored_password_hash = user['password']

                    if check_password_hash(stored_password_hash, password):
                        if user['status'] in ["Suspended", "Breached"]:  # Check user status
                            flash(f"Your account is {user['status']}. Access denied.", "danger")
                            return redirect(url_for("login"))

                        # Passwords match, proceed with MFA
                        user_email = user['email']

                        # Generate a 6-digit MFA code
                        mfa_code = str(secrets.randbelow(899999) + 100000)
                        session['email'] = user_email  # Save email in session
                        session['mfa_code'] = mfa_code  # Store in session
                        session['username'] = username  # Save username for next steps
                        session['role'] = role

                        # Save current logged in user email to shelve
                        with shelve.open('settings.db', writeback=True) as settings_db:
                            settings_db['CurrentUserEmail'] = user_email

                        # Send the code via email
                        msg =  flask_mail.Message(subject = 'MFA Code',
                                      recipients=[user_email])
                        msg.body = f'Your 6-digit MFA code is {mfa_code}'

                        try:
                            print(mfa_code)
                            mail.send(msg)
                            flash('An authentication code has been sent to your email.', 'info')
                            print(session)
                            return redirect(url_for('mfa_verify'))  # Redirect to MFA verification page
                        except Exception as e:
                            print(f"Email send error: {e}")
                            flash("Error sending MFA", 'danger')
                    else:
                        flash('Invalid login credentials', 'danger')
                else:
                    flash('Invalid login credentials', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')

    return render_template('login.html', form=form)

@app.route('/forgetpassword', methods=['GET', 'POST'])
def forget_password():
    form = forgetpassword()  # Instantiate the form
    if form.validate_on_submit():
        foremail = form.email.data

        try:
            with shelve.open('users.db', 'r') as db:
                for username, user_data in db.items():
                    if user_data['email'] == foremail:
                        user_email = foremail
                        mfa_code = str(secrets.randbelow(899999) + 100000)

                        # Store details in session
                        session['mfa_code'] = mfa_code
                        session['email'] = user_email
                        session['username'] = username

                        msg = flask_mail.Message(subject='MFA Code',
                                      sender='iatfadteam@gmail.com',
                                      recipients=[user_email])
                        msg.body = f'Your 6-digit MFA code is {mfa_code}'
                        mail.send(msg)
                        flash('An authentication code has been sent to your email.', 'info')
                        return redirect(url_for('mfa_verify2'))

                        # Email not found
                flash('Email not found', 'danger')
        except Exception as e:
                    # Log the error and inform the user
            app.logger.error(f"Error during password recovery: {str(e)}")
            flash('An internal error occurred. Please try again later.', 'danger')

    return render_template('forget_password.html', form=form)

@app.route('/mfa-verify2', methods=['GET', 'POST'])
def mfa_verify2():
    form = MFAForm()

    if form.validate_on_submit():
        entered_code = form.code.data

        if entered_code == session.get('mfa_code'):
            # MFA passed, redirect to reset password
            flash('MFA verification successful. Please reset your password.', 'info')
            return redirect(url_for('reset_password'))  # Redirect to reset password page
        else:
            flash('Invalid authentication code', 'danger')

    return render_template('mfa_verify2.html', form=form)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    form = updatepasswordForm()  # Form for password reset

    # Ensure the user has completed MFA
    if 'username' not in session or 'email' not in session:
        flash('You must verify your identity before resetting your password.', 'danger')
        return redirect(url_for('forget_password'))

    username = session['username']

    if form.validate_on_submit():
        new_password = form.password.data
        confirm_password = form.confirm_password.data

        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
        else:
            try:
                with shelve.open('users.db', 'w') as db:
                    # Update the user's password
                    if username in db:
                        user_data = db[username]
                        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
                        user_data['password'] = hashed_password
                        db[username] = user_data  # Save updated password

                        flash('Your password has been updated. Please log in with the new password.', 'success')
                        # Clear session data after password reset
                        session.clear()
                        return redirect(url_for('login'))  # Redirect to login page
                    else:
                        flash('User not found. Please start the process again.', 'danger')
                        return redirect(url_for('forget_password'))
            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'danger')

    return render_template('reset_password.html', form=form)

@app.route('/mfa-verify', methods=['GET', 'POST'])
def mfa_verify():
    form = MFAForm()

    if form.validate_on_submit():
        entered_code = form.code.data

        if entered_code == session.get('mfa_code'):
            # MFA passed, log the user in
            username = session.get('username')

            # Use context manager to ensure the shelf is properly opened and closed
            with shelve.open('users.db', 'r') as db:
                user_email = db[username]['email']
                hashed_password = db[username]['password']
                role = db[username]['role']

            user = User(username, user_email, hashed_password, role)
            login_user(user)
            flash('You are now logged in', 'success')
            session.pop('mfa_code')  # Clear MFA code after success
            return redirect(url_for('set_ip'))
        else:
            flash('Invalid authentication code', 'danger')

    return render_template('mfa_verify.html', form=form)

@app.route('/logout')
def logout():
    session.clear()

    # Clear the stored user email from shelve
    try:
        with shelve.open('settings.db', writeback=True) as db:
            if 'CurrentUserEmail' in db:
                del db['CurrentUserEmail']
    except Exception as e:
        print(f"Error clearing CurrentUserEmail: {e}")

    flash('Please log in to access.', 'info')
    return redirect(url_for('login'))


from flask import Flask, flash, render_template, request, redirect, session, url_for
import shelve, re

# Helper function to get settings from the database
def get_settings():
    with shelve.open('settings.db', 'c') as db:
        settings = db.get('settings', {'interval': 2000, 'threshold': 5})  # Default interval is 2000ms and default threshold is 5
        app.logger.debug(f"Settings read from database: {settings}")
    return settings

def update_settings(new_settings):
    with shelve.open('settings.db', 'c') as db:
        db['settings'] = new_settings
        app.logger.debug(f"Settings updated in database: {new_settings}")

def encode_time(timer_id, start_hour, start_minute, end_hour, end_minute):
    # Create a base packet with the given timer ID
    packet = bytearray([0x4C, 0x8A, 0x01, 0x01, timer_id, 0x00, 0x00,0xEE]) #timer_id was in the wrong byte position

    # Encode start and end times in BCD format
    start_hour_bcd = (start_hour // 10 << 4) | (start_hour % 10)
    start_minute_bcd = (start_minute // 10 << 4) | (start_minute % 10)
    end_hour_bcd = (end_hour // 10 << 4) | (end_hour % 10)
    end_minute_bcd = (end_minute // 10 << 4) | (end_minute % 10)

    # Append times to the packet
    packet.append(start_hour_bcd)
    packet.append(start_minute_bcd)
    packet.append(end_hour_bcd)
    packet.append(end_minute_bcd)

    # Add additional static values (e.g., 0x01)
    packet.extend([0x01])

    # Add padding (0xFF and 0x00)
    packet.extend([0xFF] * 5)
    packet.extend([0x00] * 2)

    # Return the packet as a hexadecimal string
    return packet.hex()

from scapy.all import *

def get_next_ip_id():
    with shelve.open('settings.db', 'c') as db:
        last_id = db.get('last_ip_id', 7500)
        db['last_ip_id'] = last_id + 1
        return last_id
# 
def get_interface_by_ip(target_ip):
    for iface_name, iface_addrs in psutil.net_if_addrs().items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and addr.address == target_ip:
                print(f"Found interface {iface_name} with IP {target_ip}")
                return iface_name
    raise RuntimeError(f"No interface found with IP {target_ip}")

def send_tcp_packet(encoded_byte, source_port, server_isn, server_ack, source_ip, dest_ip):
    destination_ip = dest_ip
    source_ip = source_ip
    destination_port = 50000

    payload = bytes.fromhex(encoded_byte)
    flags = "PA"  # PSH + ACK flags
    ip_id = get_next_ip_id()

    # Detect the correct network interface
    try:
        interface_name = get_interface_by_ip(source_ip)
        print(f"Using interface: {interface_name}")
    except RuntimeError as e:
        print(f"[Error] {e}")
        return

    # Create IP and TCP headers without manually setting checksum
    ip_packet = IP(src=source_ip, dst=destination_ip, id=ip_id, flags="DF", ttl=128)

    # Let Scapy calculate the checksum automatically
    tcp_packet = TCP(
        sport=source_port,
        dport=destination_port,
        seq=server_isn,
        ack=server_ack,
        flags=flags,
        window=64233
    )

    packet = ip_packet / tcp_packet / payload
    print(f"Sent TCP packet with payload: {encoded_byte}")
    send(packet)

    packet_counter = 0

    while True:
        packet_counter += 1
        print(f"Sniff attempt {packet_counter}")  # <-- Add this
        sniffpacket = sniff(iface=interface_name,
                            filter=f"tcp and src host {destination_ip} and port {destination_port}",
                            count=1, timeout=0.5) # Adjust the Ethernet port name to correspond with the connected interface.

        if not sniffpacket:
            print(f"No packet received. Attempt: {packet_counter}")  # <-- Add this
            if packet_counter == 2:
                print("Timeout reached, no packet captured.")
                return False
            continue

        pkt = sniffpacket[0]
        tcp_flags = pkt[TCP].flags
        tcp_payload = bytes(pkt[TCP].payload).rstrip(b'\x00')
        payload_len = len(tcp_payload)
        print(f"Packet captured: {payload_len} bytes | Flags: {tcp_flags}")

        if tcp_flags in ["PA", "A", "FPA", "FA"]:
            stored_ack = sniffpacket[0][TCP].ack
            stored_seq = sniffpacket[0][TCP].seq

            # Send ACK with proper sequence and acknowledgment numbers
            ip_id = get_next_ip_id()
            ack_packet = IP(src=source_ip, dst=destination_ip, id=ip_id) / TCP(
                sport=source_port,
                dport=destination_port,
                seq=stored_ack,
                ack=stored_seq + payload_len,
                flags="A",
                window=64233
            )
            send(ack_packet)
            with shelve.open('settings.db', 'c') as db:
                db['syn_ack_seq'] = stored_ack
                db['syn_ack_ack'] = stored_seq + payload_len

            print("Found correct packet and sent ACK")
            return True
        else:
            continue

# Function to perform the 3-way TCP handshake
def send_syn_packet(client_ip, server_ip, source_port, destination_port):
    try:
        try:
            interface_name = get_interface_by_ip(client_ip)
            print(f"Using interface: {interface_name}")
        except RuntimeError as e:
            print(f"[Error] {e}")
            return
        # Step 1: Generate a random ISN
        isn = random.randint(0, 2**32 - 1)
        ip_id = get_next_ip_id()
        # Step 2: Create and send the SYN packet
        ip = IP(src=client_ip, dst=server_ip,id=ip_id)
        tcp_options = [
            ('MSS', 1460),          # Maximum Segment Size
            ('NOP', None),          # No-Operation (NOP)
            ('WScale', 8),          # Window Scale
            ('NOP', None),          # No-Operation (NOP)
            ('NOP', None),          # No-Operation (NOP)
            ('SAckOK', b'')         # SACK Permitted
        ]
        syn = TCP(
            sport=source_port,
            dport=destination_port,
            flags="S",
            seq=isn,  # Set the ISN
            window=64240,
            options=tcp_options,
            chksum = None
        )

        syn_ack = sr1(ip / syn, timeout=1)

        if syn_ack is None or syn_ack[TCP].flags != "SA":
            print("Did not receive SYN-ACK")
            return None, None
        ip_id = get_next_ip_id()
        ip = IP(src=client_ip, dst=server_ip,id=ip_id)
        # Step 3: Send the ACK to complete the handshake
        ack = TCP(
            sport=source_port,
            dport=destination_port,
            flags="A",
            seq=syn_ack.ack,  # Increment by 1 as per TCP handshake
            ack=syn_ack.seq + 1,  # Acknowledge the server's SYN-ACK
            window=64240,
            chksum = None
        )
        send(ip / ack)
        print("Handshake completed")
        time.sleep(0.2)

        while True:

            psh_ack = sniff(iface=interface_name, filter=f"tcp and src host {server_ip} and port {destination_port}", count=1, timeout=1) # Adjust the Ethernet port name to correspond with the connected interface.

            if not psh_ack:
                # print("no packet")
                continue

            psh_ack = psh_ack[0][TCP]

            if  psh_ack.flags == "PA" : # PSH-ACK received

                payload_data = bytes(psh_ack.payload).rstrip(b'\x00')

                payload_length = len(payload_data)

                print("Keep-Alive PSH-ACK received")

                # Send Keep-Alive ACK
                ip_id = get_next_ip_id()
                ip = IP(src=client_ip, dst=server_ip, id=ip_id)
                keep_alive_ack = TCP(

                    sport=source_port,

                    dport=destination_port,

                    flags="A",

                    seq=psh_ack.ack,  # Use the ACK number from the PSH-ACK

                    ack=psh_ack.seq + payload_length,  # Acknowledge the Keep-Alive payload

                    window=64236,
                    chksum = None
                )
                print(f"actual seq = {psh_ack.ack}")

                send(ip / keep_alive_ack)
                with shelve.open('settings.db', 'c') as db:
                    db['syn_ack_seq'] = psh_ack.ack
                    db['syn_ack_ack'] = psh_ack.seq + payload_length
                print("Acknowledged server's PSH, ACK")
                print(f"ISN{syn_ack.seq}")

                print("Keep-Alive ACK sent")
                time.sleep(1)
            else:
                print("Unexpected packet received")
            # Save the SYN-ACK sequence number for further use

    except Exception as e:
        print(f"Error: {e}")

def send_RST_packet(source_port, server_isn, server_ack, source_ip, destination_ip):
    destination_port = 50000
    ip = IP(src=source_ip, dst=destination_ip)

    # Construct the TCP header
    tcp = TCP(
        sport=source_port,  # Source port
        dport=destination_port,  # Destination port
        flags="RA",  # RST and ACK flags
        seq=server_ack,  # Sequence number
        ack=server_isn+4,  # Acknowledgment number
        window=0,  # Window size
    )

    # Combine the headers into a single packet
    packet = ip / tcp

    # Send the packet
    send(packet, verbose=0)

def start_send_manual_feed(source_port, server_isn, server_ack, source_ip, destination_ip):
    print(f"[DEBUG] --> start_send_manual_feed() triggered at {datetime.now().strftime('%H:%M:%S')}")
    encoded_data = "ccdda10100010001a448"
    success = send_tcp_packet(
        encoded_byte=encoded_data,
        source_port=source_port,
        server_isn=server_isn,
        server_ack=server_ack,
        source_ip=source_ip,
        dest_ip=destination_ip
    )
    if not success:
        print("Start Manual feed packet not found")

def stop_send_manual_feed(source_port, server_isn, server_ack, source_ip, destination_ip):
    print(f"[DEBUG] --> stop_send_manual_feed() triggered at {datetime.now().strftime('%H:%M:%S')}")
    encoded_data = "ccdda10100000001a346"
    success = send_tcp_packet(
        encoded_byte=encoded_data,
        source_port=source_port,
        server_isn=server_isn,
        server_ack=server_ack,
        source_ip=source_ip,
        dest_ip=destination_ip
    )
    if not success:
        print("Stop manual feed packet not found")

# Route to update the interval setting
@app.route('/update_interval', methods=['POST'])
def update_interval():
    try:
        minutes = request.form.get('interval_minutes', 0)  # Retrieve from form
        seconds = request.form.get('interval_seconds', 0)

        app.logger.debug(f"Received minutes: {minutes}, seconds: {seconds}")

        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0

        if minutes < 0 or seconds < 0:
            raise ValueError("Minutes and seconds should be non-negative integers.")

        total_seconds = (minutes * 60) + seconds

        # Update settings
        settings = get_settings()
        settings['interval_seconds'] = total_seconds
        update_settings(settings)

        return redirect(url_for('dashboard'))  # Redirect after updating

    except ValueError as e:
        flash('Invalid input! Minutes and seconds must be non-negative integers.', 'danger')
        return redirect(url_for('settings'))  # Redirect to settings page

    except Exception as e:
        app.logger.error(f"An error occurred while updating interval: {str(e)}")
        flash('An unexpected error occurred.', 'danger')
        return redirect(url_for('settings'))

# Route to retrieve the current interval setting
@app.route('/get_interval', methods=['GET'])
def get_interval():
    try:
        settings = get_settings()
        app.logger.debug(f"Current interval retrieved: {settings['interval']}")
        return jsonify({'interval': settings['interval']}), 200
    except Exception as e:
        app.logger.error(f"An error occurred while retrieving interval: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving interval.'}), 500

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    try:
        threshold = request.json.get('threshold')
        app.logger.debug(f"Received threshold: {threshold}")  # Debug log

        # Validate threshold (should be a positive integer)
        if threshold is not None:
            threshold = int(threshold)
            if threshold <= 0:
                raise ValueError("Threshold should be a positive integer.")

            # Update threshold in settings
            settings = get_settings()
            settings['threshold'] = threshold
            update_settings(settings)
            app.logger.debug(f"Threshold updated successfully to {threshold}")
            return jsonify({'message': 'Threshold updated successfully'}), 200
        else:
            return jsonify({'error': 'Threshold value not provided'}), 400

    except ValueError as e:
        app.logger.error(f"Invalid threshold value: {e}")
        return jsonify({'error': 'Invalid threshold value. Must be a positive integer.'}), 400

    except Exception as e:
        app.logger.error(f"An error occurred while updating threshold: {str(e)}")
        return jsonify({'error': 'An error occurred while updating threshold.'}), 500

@app.route('/get_threshold', methods=['GET'])
def get_threshold():
    try:
        settings = get_settings()
        app.logger.debug(f"Current threshold retrieved: {settings['threshold']}")
        return jsonify({'threshold': settings['threshold']}), 200
    except Exception as e:
        app.logger.error(f"An error occurred while retrieving threshold: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving threshold.'}), 500

@app.route('/pellet_data')
def get_pellet_data():
    # Define test data
    pellet_data = {
        '03 Jul 2025': {'Morning': 250, 'Evening': 200, 'Total': 450},
        '04 Jul 2025': {'Morning': 200, 'Evening': 200, 'Total': 400},
        '05 Jul 2025': {'Morning': 100, 'Evening': 200, 'Total': 300},
        '06 Jul 2025': {'Morning': 120, 'Evening': 130, 'Total': 250},
        '07 Jul 2025': {'Morning': 220, 'Evening': 180, 'Total': 400},
        '08 Jul 2025': {'Morning': 140, 'Evening': 160, 'Total': 300},
        '09 Jul 2025': {'Morning': 125, 'Evening': 150, 'Total': 275},
    }
    # Check if the database exists, and if not, create and populate it
    db_path = 'mock_chart_data.db'
    if not os.path.exists(db_path + '.db'):  # Shelve creates additional files with ".db"
        with shelve.open(db_path, 'c') as db:
            for date, feeds in pellet_data.items():
                db[datetime.strptime(date, "%d %b %Y").strftime("%Y-%m-%d")] = feeds
            db.sync()
        print("Database created and initialized.")

    # Generate the last 7 days (including today)
    current_day = datetime.today().date()
    last_7_days = [(current_day - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]

    # Initialize lists to hold pellet counts
    first_feed_counts = []
    second_feed_counts = []
    total_feed_counts = []

    # Open the shelve database and retrieve data
    with shelve.open(db_path, 'r') as db:
        for day in last_7_days:
            if day in db:
                day_data = db[day]
                morning_count = day_data.get('Morning', 0)
                evening_count = day_data.get('Evening', 0)
                total = day_data.get('Total', 0)
            else:
                morning_count = evening_count = total = 0

            first_feed_counts.append(morning_count)
            second_feed_counts.append(evening_count)
            total_feed_counts.append(total)

    response_data = {
        'labels': [datetime.strptime(day, "%Y-%m-%d").strftime("%d %b") for day in last_7_days],
        'first_feed_left': first_feed_counts,
        'second_feed_left': second_feed_counts,
        'total_feed_count': total_feed_counts
    }

    return jsonify(response_data)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    edit_form = configurationForm(request.form)
    time.sleep(0.5)
    if request.method == 'POST':
        source_ip = request.form.get('source_ip')
        destination_ip = request.form.get('destination_ip')

        print(f"Source IP: {source_ip}, Destination IP: {destination_ip}")
        start_syn_packet_thread("192.168.1.65", "192.168.1.18", port, 50000)

    db = shelve.open('settings.db', 'w')
    Time_Record_dict = db.get('Time_Record',{})
    db.close()
    setting = Time_Record_dict.get('Time_Record_Info')
    checking_interval = setting.get_interval_seconds()
    id_array = []
    for key in Time_Record_dict:
        product = Time_Record_dict.get(key)
        if key == "Time_Record_Info":
            id_array.append(product)
    # Fetch Pellet Data (You can directly use `get_pellet_data` or emulate its behavior)
    print("Fetching pallet data")
    response = get_pellet_data()
    pellet_data = response.json # Convert the Flask Response to JSON

    with shelve_lock:
        with shelve.open('currentcount.db', 'c') as db2:
            object_count = db2.get('object_count', 0)  # Get count dictionary, default to empty
    latest_count = object_count  # Get latest count for label 1, default to 0 if not found

    return render_template('dashboard.html', count=len(id_array), id_array=id_array, edit=0, form=edit_form, latest_count=latest_count,
                           pellet_labels=pellet_data['labels'], first_feed_left=pellet_data['first_feed_left'], second_feed_left=pellet_data['second_feed_left'],total_feed_count = pellet_data['total_feed_count'],checking_interval = checking_interval)

@app.route('/camera_view',methods=['GET','POST'])
@login_required
def camera_view():
    session['access_video_feed'] = True
    return render_template('camera_view.html')

@app.route('/export_data', methods=['POST'])
def export_data():
    # Get data from the request
    data = request.get_json()
    labels = data.get('labels', [])
    first = data.get('first', [])
    second = data.get('second', [])
    total = data.get('total', [])

    # Create an Excel workbook and worksheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = ' Leftover Pellets Over The Past Seven Days'

    # Set up the header
    sheet["A1"] = "Date"
    sheet["B1"] = "first feed number of pellets left"
    sheet["C1"] = "second feed number of pellets left"
    sheet["D1"] = "total feed pellets fed"
    sheet["A1"].font = Font(bold=True)
    sheet["B1"].font = Font(bold=True)
    sheet["C1"].font = Font(bold=True)
    sheet["D1"].font = Font(bold=True)

    # Populate the worksheet with data
    for index, (label, first,second,total) in enumerate(zip(labels, first,second,total), start=2):
        sheet[f"A{index}"] = label
        sheet[f"B{index}"] = first
        sheet[f"C{index}"] = second
        sheet[f"D{index}"] = total

    # Save the workbook to a file
    file_path = 'consumption_data.xlsx'
    workbook.save(file_path)

    return send_file(file_path, as_attachment=True, download_name='leftover_feed_data.xlsx')

import re
latest_valid_settings = None
latest_set_ip_settings = None

@app.route('/update', methods=['GET', 'POST'])
@login_required
def update_setting():
    setting = configurationForm(request.form)
    mode = request.args.get("mode", "auto")
    try:
        if request.method == 'POST':
            if request.is_json:
                data = request.get_json()

                # MANUAL FEED BLOCK
                if mode == "manual":
                    manual_form = data.get("manual_form")
                    manual_feed_action = data.get("manual_feed_action")
                    if manual_form and manual_feed_action in ["start", "stop"]:
                        try:
                            with shelve.open("settings.db", 'c') as db, shelve.open("IP.db", 'r') as ip_db:
                                config = {
                                    'Port': db.get('Port'),
                                    'syn_ack_seq': db.get('syn_ack_seq'),
                                    'syn_ack_ack': db.get('syn_ack_ack'),
                                    'source': ip_db.get("IP", {}).get("source"),
                                    'destination': ip_db.get("IP", {}).get("destination")
                                }

                                Time_Record_dict = db.get('Time_Record', {})
                                db['Time_Record'] = Time_Record_dict

                            for key, value in config.items():
                                if not value:
                                    location = "IP.db" if key in ['source', 'destination'] else "settings"
                                    return jsonify({"status": "error", "message": f"Missing '{key}' in {location}."}), 400

                            port = config['Port']
                            server_isn = config['syn_ack_seq']
                            server_ack = config['syn_ack_ack']
                            source_ip = config['source']
                            destination_ip = config['destination']

                            if manual_feed_action == "start":
                                start_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)
                                return jsonify({"status": "success", "message": "Manual feeding started."})
                            else:
                                stop_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)
                                return jsonify({"status": "success", "message": "Manual feeding stopped."})

                        except Exception as e:
                            return jsonify({"status": "error", "message": f"Manual feed error: {e}"}), 500

                    return jsonify({"status": "error", "message": "Invalid manual feed request."}), 400

                # AUTO FEED BLOCK
                elif mode == "auto":
                    required_fields = ['first_timer', 'second_timer', 'minutes', 'interval_minutes', 'interval_seconds', 'pellets']
                    if not all(k in data for k in required_fields):
                        return jsonify({"status": "error", "message": "Missing required fields."}), 400

                    first_timer = data['first_timer']
                    second_timer = data['second_timer']
                    pattern = r'^(?:[01]\d[0-5]\d|2[0-3][0-5]\d)$'

                    if not re.match(pattern, first_timer) or not re.match(pattern, second_timer):
                        return jsonify({"status": "error", "message": "Time format must be HHMM (e.g. 0830)."}), 400

                    try:
                        first_hour = int(first_timer[:2])
                        second_hour = int(second_timer[:2])
                    except ValueError:
                        return jsonify({"status": "error", "message": "Invalid numeric values in time."}), 400

                    if not (6 <= first_hour <= 12) or not (12 <= second_hour <= 24):
                        return jsonify({"status": "error", "message": "Morning feed must be between 06:00–12:00, and evening feed must be between 12:00–24:00."}), 400

                    try:
                        with shelve.open('settings.db', 'c') as db:
                            Time_Record_dict = db.get('Time_Record', {})
                            j = Time_Record_dict.get('Time_Record_Info')

                            j.set_first_timer(data['first_timer'])
                            j.set_second_timer(data['second_timer'])
                            interval_minutes_str = str(data.get('interval_minutes', '')).strip()
                            interval_minutes = int(interval_minutes_str) if interval_minutes_str.isdigit() else 0
                            interval_seconds_str = str(data.get('interval_seconds', '')).strip()
                            interval_seconds = int(interval_seconds_str) if interval_seconds_str.isdigit() else 0
                            interval_sec = interval_minutes * 60 + interval_seconds
                            j.set_interval_seconds(interval_sec)
                            j.set_pellets(data['pellets'])
                            minutes_str = str(data.get('minutes', '')).strip()
                            minutes = int(minutes_str) if minutes_str.isdigit() else 0
                            j.set_seconds(minutes * 60)

                            db['Time_Record'] = Time_Record_dict

                            config = {'Port': db.get('Port'), 'syn_ack_seq': db.get('syn_ack_seq'), 'syn_ack_ack': db.get('syn_ack_ack')}

                        with shelve.open("IP.db", 'r') as ip_db:
                            ip_data = ip_db.get("IP", {})
                            config['source'] = ip_data.get("source")
                            config['destination'] = ip_data.get("destination")

                        for key, value in config.items():
                            if not value:
                                location = 'IP.db' if key in ['source', 'destination'] else 'settings'
                                return jsonify({"status": "error", "message": f"Missing '{key}' in {location}."}), 400

                        try:
                            global latest_valid_settings
                            latest_valid_settings = j
                            schedule_feeding_alerts(data['first_timer'], data['second_timer'], data['minutes'], session.get("email"))
                            return jsonify({"status": "success", "message": "Feeding schedule updated."})

                        except Exception as e:
                            return jsonify({"status": "error", "message": f"Failed to update feeding schedule: {str(e)}"}), 500

                    except Exception as e:
                        return jsonify({"status": "error", "message": f"Auto feed setup failed: {e}"}), 500

                return jsonify({"status": "error", "message": "Invalid mode."}), 400

            # Fallback for form-based submission
            return render_template('settings.html', form=setting, mode=mode)

        # GET method – load current settings
        with shelve_lock:
            with shelve.open('settings.db', 'r') as db:
                Time_Record_dict = db.get('Time_Record', {})
                j = Time_Record_dict.get('Time_Record_Info')

        setting.first_timer.data = j.get_first_timer()
        setting.second_timer.data = j.get_second_timer()
        setting.pellets.data = j.get_pellets()
        setting.minutes.data = j.get_seconds() // 60

        interval_seconds = j.get_interval_seconds()
        setting.interval_minutes.data = interval_seconds // 60 if interval_seconds else None
        setting.interval_seconds.data = interval_seconds % 60 if interval_seconds else None

        return render_template('settings.html', form=setting, mode=mode)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Unexpected server error: {e}"}), 500

def send_feeding_complete_email(user_email, feed_time):
    with app.app_context():
        try:
            msg = flask_mail.Message(subject="Feeding Complete",
                          recipients=[user_email],
                          body= f"The {feed_time} has been completed",
                          )
            mail.send(msg)
            print(f"Email sent to {user_email} for {feed_time}.")
        except Exception as e:
            print(f"Error sending email: {e}")

def reschedule_feeding_alerts():
    db = shelve.open('settings.db', 'r')
    Time_Record_dict = db['Time_Record']
    j = Time_Record_dict.get('Time_Record_Info')

    # Get updated times and durations
    first_timer = j.get_first_timer()
    second_timer = j.get_second_timer()
    feeding_duration = j.get_seconds()*60

    print(f"[RESCHEDULE] First Timer: {first_timer}, Second Timer: {second_timer}, Feeding Duration: {feeding_duration} seconds")

    user_email = db.get("CurrentUserEmail")
    if not user_email:
        email_db = db.get("Email_Data", {})
        email_instance = email_db.get("Email_Info")
        if email_instance and hasattr(email_instance, "get_recipient_email"):
            user_email = email_instance.get_recipient_email()
        else:
            user_email = "iatfadteam@gmail.com"

    print(f"[RESCHEDULE] Preparing to schedule feeding alert emails for: {user_email}")

    # Calculate new run_date for the first alert (next day)
    now = datetime.now()
    timezone = pytz.timezone("Asia/Singapore")
    first_timer_dt = now.replace(hour=int(first_timer[:2]), minute=int(first_timer[2:]), second=0, microsecond=0)
    first_end_time = timezone.localize(first_timer_dt + timedelta(seconds=feeding_duration))
    # Reschedule the feeding alerts for the next day

    # Modify the existing job with the new run_date
    job = scheduler.get_job('first_feeding_alert')
    if job:
        job.modify(run_date=first_end_time)
        print(f"[RESCHEDULE] First feeding alert rescheduled to: {first_end_time}")
    else:
        scheduler.add_job(
            func=send_feeding_complete_email,
            trigger='date',
            run_date=first_end_time,
            args=[user_email, "first feeding complete"],
            id='first_feeding_alert',
            misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
        )
        print(f"[RESCHEDULE] First feeding alert job created for: {first_end_time}")

    # Repeat the process for the second timer
    second_timer_dt = now.replace(hour=int(second_timer[:2]), minute=int(second_timer[2:]), second=0, microsecond=0)
    second_end_time = timezone.localize(second_timer_dt + timedelta(seconds=feeding_duration))

    # Modify the second feeding alert job
    job = scheduler.get_job('second_feeding_alert')
    if job:
        job.modify(run_date=second_end_time)
        print(f"[RESCHEDULE] Second feeding alert rescheduled to: {second_end_time}")
    else:
        scheduler.add_job(
            func=send_feeding_complete_email,
            trigger='date',
            run_date=second_end_time,
            args=[user_email, "second feeding complete"],
            id='second_feeding_alert',
            misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
        )
        print(f"[RESCHEDULE] Second feeding alert job created for: {second_end_time}")

def schedule_daily_task():
    while True:
        print("Updating schedule")
        reschedule_feeding_alerts()
        print("[SCHEDULER] Next reschedule will happen in 24 hours.")
        time.sleep(86400)

def schedule_feeding_alerts(first_timer, second_timer, feeding_duration, user_email):
    try:
        now = datetime.now()  # Use current date for scheduling
        first_timer_dt = now.replace(hour=int(first_timer[:2]), minute=int(first_timer[2:]), second=0, microsecond=0)
        second_timer_dt = now.replace(hour=int(second_timer[:2]), minute=int(second_timer[2:]), second=0, microsecond=0)

        feeding_duration_in_seconds = int(feeding_duration) * 60  # Convert minutes to seconds

        # Localize the datetime objects
        timezone = pytz.timezone("Asia/Singapore")
        first_end_time = timezone.localize(first_timer_dt + timedelta(seconds=feeding_duration_in_seconds))
        second_end_time = timezone.localize(second_timer_dt + timedelta(seconds=feeding_duration_in_seconds))

        # Ensure feeding times are in the future
        if first_end_time < timezone.localize(now):
            print("First feeding time is in the past. Skipping scheduling.")
        else:
            print("Scheduling first feeding alert at:", first_end_time)
            existing_job = scheduler.get_job('first_feeding_alert')

            if existing_job:
                # Reschedule the existing job
                scheduler.remove_job('first_feeding_alert')

            # Add the job if it doesn't exist
            scheduler.add_job(
                    func=send_feeding_complete_email,
                    trigger='date',
                    run_date=first_end_time,
                    args=[user_email, "first feeding complete"],
                    id='first_feeding_alert',
                    misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
                )
            print("Job 'first_feeding_alert' added.")

        if second_end_time < timezone.localize(now):
            print("Second feeding time is in the past. Skipping scheduling.")
        else:
            print("Scheduling second feeding alert at:", second_end_time)
            existing_job = scheduler.get_job('second_feeding_alert')

            if existing_job:
                # Update the existing job
                scheduler.remove_job('second_feeding_alert')

            # Add a new job if it doesn't exist
            scheduler.add_job(
                    func=send_feeding_complete_email,
                    trigger='date',
                    run_date=second_end_time,
                    args=[user_email, "second feeding complete"],
                    id='second_feeding_alert',
                    misfire_grace_time=3600
                )
            print("Job 'second_feeding_alert added.")

    except ValueError as e:
        print(f"Error parsing time: {e}")
    except Exception as e:
        print(f"Scheduling error: {e}")

@app.route('/update/email', methods=['GET', 'POST'])
@login_required
@role_required("Admin")
def update_email_settings():
    setting = emailForm(request.form)

    if request.method == 'POST' and setting.validate():
        db = shelve.open('settings.db', 'w')
        Email_dict = db['Email_Data']

        j = Email_dict.get('Email_Info')
        j.set_sender_email(setting.sender_email.data)
        j.set_recipient_email(setting.recipient_email.data)
        j.set_APPPassword(setting.App_password.data)
        j.set_days(setting.days.data)

        db['Email_Data'] =Email_dict
        Time_Record_dict = db['Time_Record']

        p = Time_Record_dict.get('Time_Record_Info')
        p.set_confidence(setting.confidence.data)

        db['Time_Record'] = Time_Record_dict

        db.close()

        return redirect(url_for('dashboard'))
    else:
        Email_dict = {}
        db = shelve.open('settings.db', 'r')
        Email_dict = db['Email_Data']
        Time_Record_dict = db['Time_Record']
        db.close()

        j = Email_dict.get('Email_Info')
        setting.sender_email.data = j.get_sender_email()
        setting.recipient_email.data = j.get_recipient_email()
        setting.App_password.data = j.get_APPPassword()
        setting.days.data = j.get_days()

        p = Time_Record_dict.get('Time_Record_Info')
        setting.confidence.data = p.get_confidence()
        print("Confidence rate set to",setting.confidence.data)
        return render_template('email_settings.html', form=setting)

@app.route('/clear_video_feed_access', methods=['POST'])
def clear_video_feed_access():
    db = shelve.open('settings.db', 'w')
    db['Generate_Status'] = False
    db.close()
    return jsonify({'message': 'Video feed access cleared'}), 200  # Returning JSON response

@app.route('/video_feed')

def video_feed():
    db = shelve.open('settings.db', 'w')
    db['Generate_Status'] = True
    db.close()
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error: {e}")
        return "Error generating video feed"

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    form = FeedbackForm()
    user_email = session.get('email')  # Retrieve the email from the session
    user_name = session.get('username')

    if not user_email:
        flash('Please log in to access the feedback form.', 'danger')
        return redirect(url_for('login'))
    
    try:
        db = shelve.open('settings.db', 'r')
        Email_dict = db.get('Email_Data', {})
        recipient_email = "iatfadteam@gmail.com"  # Default fallback

        if 'Email_Info' in Email_dict:
            email_info = Email_dict['Email_Info']
            if hasattr(email_info, 'get_recipient_email'):
                configured_email = email_info.get_recipient_email()
                if configured_email:
                    recipient_email = configured_email
        db.close()
    except Exception as e:
        recipient_email = "iatfadteam@gmail.com"
        app.logger.warning(f"Could not retrieve recipient email, using fallback: {e}")


    if form.validate_on_submit():
        try:
            # Attempt to compose and send the email
            msg = flask_mail.Message(
                subject="New Feedback",
                sender=user_email,
                recipients=[recipient_email],
                body=f"Name: {user_name}\nEmail: {user_email}\nMessage:\n{form.message.data}"
            )
            print(f"message sent to {recipient_email}")
            mail.send(msg)

            # Flash success message and redirect to dashboard
            flash('Your feedback has been sent successfully!', 'success')
            return redirect(url_for('feedback'))

        except Exception as e:
            # Flash error message in case of failure
            flash('An error occurred while sending your feedback. Please try again.', 'danger')
            # Log the error for debugging purposes (optional)
            app.logger.error(f'Feedback form error: {e}')

    return render_template('feedback.html', form=form)

@app.route('/changed_password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = updatepasswordForm()  # Form instance for password change

    # Ensure the user is logged in
    if 'username' not in session:
        flash('You must be logged in to change your password.', 'danger')
        return redirect(url_for('login'))

    username = session['username']  # Retrieve the logged-in user's username

    if form.validate_on_submit():
        new_password = form.password.data
        confirm_password = form.confirm_password.data

        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
        else:
            try:
                with shelve.open('users.db', 'w') as db:
                    # Check if the user exists in the database
                    if username in db:
                        user_data = db[username]
                        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
                        user_data['password'] = hashed_password
                        db[username] = user_data  # Save updated password
                        flash('Your password has been updated successfully.', 'success')
                        print(user_data['role'])
                        return redirect(url_for('dashboard'))  # Redirect to a post-update page
                    else:
                        flash('User not found. Please log in.', 'danger')
                        return redirect(url_for('login'))
            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'danger')

    return render_template('changed_password.html', form=form)

@app.route('/retrieve', methods=['GET'])
@login_required
@role_required('Admin')
def retrieve_users():
    page = request.args.get('page', default=1, type=int)  # Get the current page, default to 1
    per_page = 3  # Number of users per page
    search_query = request.args.get('search', default='', type=str).lower()  # Get search input
    with shelve.open('users.db', 'r') as db:
        user_dict = dict(db)  # Convert shelve object to a dictionary
        total_users = len(user_dict)  # Total number of users

        if search_query:
            filtered_users = {
                username: data
                for username, data in user_dict.items()
                if (
                    search_query in username.lower()
                    or search_query in data.get('email', '').lower()
                    or search_query in data.get('role', '').lower()
                    or search_query in data.get('status', '').lower()
                )
            }
        else:
            filtered_users = user_dict
        total_users = len(filtered_users)
        users = list(filtered_users.items())

    # Pagination logic
    start = (page - 1) * per_page
    end = start + per_page
    paginated_users = users[start:end]  # Slice the users for the current page

    total_pages = (total_users + per_page - 1) // per_page  # Calculate total pages
    prev_page = page - 1 if page > 1 else None
    next_page = page + 1 if page < total_pages else None

    # Pass data to the template
    return render_template(
        'retrieve.html',
        users=paginated_users,  # Pass only paginated users
        page=page,
        total_pages=total_pages,
        prev_page=prev_page,
        next_page=next_page,
        total_users=total_users,
        per_page=per_page,
        search_query=search_query,
    )

@app.route('/update/<username>', methods=['GET', 'POST'])
@login_required
@role_required('Admin')
def update_user(username):
    form = updateemailrole()  # Form instance for updating email, role, and status

    try:
        with shelve.open('users.db', 'w') as db:  # 'w' mode for read/write
            if username not in db:
                
                return redirect(url_for('retrieve_users'))

            user_data = db[username]  # Retrieve the current user data

            if form.validate_on_submit():  # Handle form submission (POST request)
                # Sanitize and validate inputs for roles and statuses
                valid_roles = {'Guest', 'Admin'}
                valid_statuses = {'Active', 'Suspended'}

                # Update user details based on the submitted form
                user_data['email'] = form.email.data
                user_data['role'] = form.role.data if form.role.data in valid_roles else user_data['role']
                user_data['status'] = form.status.data if form.status.data in valid_statuses else user_data['status']

                db[username] = user_data  # Save the updated user data
                flash('User details updated successfully.', 'success')
                return redirect(url_for('retrieve_users'))  # Redirect back to the users' page

            # Pre-fill the form with the user's current details on GET request
            form.email.data = user_data.get('email', '')
            form.role.data = user_data.get('role', '')
            form.status.data = user_data.get('status', '')

    except Exception as e:
        app.logger.error(f"Error updating user '{username}': {str(e)}")
        flash('An error occurred while updating the user. Please try again later.', 'danger')
        return redirect(url_for('retrieve_users'))

    return render_template('update_user.html', form=form, username=username, user_data=user_data)

# Delete (Remove User)
@app.route('/delete/<username>', methods=['POST'])
@login_required
@role_required('Admin')
def delete_user(username):
    with shelve.open('users.db', 'w') as db:
        if username in db:
            del db[username]
            flash(f"User {username} has been deleted successfully.", "success")
        else:
            return "User not found.", 404
    return redirect(url_for('retrieve_users'))

@app.route('/set_ip', methods=['GET','POST'])
@login_required
@role_required('Admin')
def set_ip():
    global latest_set_ip_settings
    setting = ipForm(request.form)
    if request.method == 'POST' and setting.validate():
        source_ip = setting.source_ip.data
        destination_ip = setting.destination_ip.data
        camera_ip = setting.camera_ip.data
        amcrest_username = setting.amcrest_username.data
        amcrest_password = setting.amcrest_password.data

        db = shelve.open('settings.db', 'w')
        port = db['Port']
        db.close()

        start_syn_packet_thread(source_ip, destination_ip, port, 50000)
        send_udp_packet(source_ip, "255.255.255.255", 60000, b"\x00\x00\x00\x00\x00")

        with shelve_lock:
            with shelve.open('IP.db', 'n') as db:
                db["IP"] = {
                    "source": source_ip,
                    "destination": destination_ip,
                    "camera_ip": camera_ip,
                    "amcrest_username": amcrest_username,
                    "amcrest_password": amcrest_password
                }
        latest_set_ip_settings = {
            "source": source_ip,
            "destination": destination_ip,
            "camera_ip": camera_ip,
            "amcrest_username": amcrest_username,
            "amcrest_password": amcrest_password
        }

        return redirect(url_for('update_setting', mode="auto"))

    return render_template('setIP.html', form=setting)

def get_valid_local_ip(preferred=None):
    interfaces = psutil.net_if_addrs()

    def is_valid(ip):
        return ip and not ip.startswith("127.")

    def is_apipa(ip):
        return ip.startswith("169.254.")

    # 1. Use preferred IP if it exists
    if preferred:
        for iface_addrs in interfaces.values():
            for addr in iface_addrs:
                if addr.family == socket.AF_INET and addr.address == preferred:
                    print(f"Using preferred IP: {preferred}")
                    return preferred

    # 2. Try Wi-Fi or LAN IPs in local devices excluding APIPA
    for iface_name, iface_addrs in interfaces.items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and is_valid(addr.address) and not is_apipa(addr.address):
                if "Wi-Fi" in iface_name or "Local Area Connection" in iface_name:
                    print(f"Using fallback interface {iface_name} with IP: {addr.address}")
                    return addr.address

    # 3. Any other valid non-loopback assigned by OS, non-APIPA IP
    for iface_addrs in interfaces.values():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and is_valid(addr.address) and not is_apipa(addr.address):
                print(f"Using generic fallback IP: {addr.address}")
                return addr.address

    # 4. As last resort, return an APIPA IP if any
    for iface_name, iface_addrs in interfaces.items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and is_apipa(addr.address):
                print(f"Using APIPA fallback interface {iface_name} with IP: {addr.address}")
                return addr.address

    raise RuntimeError("No IP address found on any interface")

for iface_name, iface_addrs in psutil.net_if_addrs().items():
    for addr in iface_addrs:
        if addr.family == socket.AF_INET:
            print(f"{iface_name}: {addr.address}")

def send_udp_packet(source, destination, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # Try to bind to the user-provided source IP if available
    try:
        sock.bind((source, 0))
        print(f"Socket successfully bound to user-provided source IP: {source}")
    except OSError as e:
        # If binding fails, try to get a valid local IP address in local device
        print(f"Failed to bind to user source IP {source}. Error: {e}")
        try:
            fallback_ip = get_valid_local_ip()
            sock.bind((fallback_ip, 0))
            print(f"Fallback: Socket bound to device IP: {fallback_ip}")
        except Exception as e2:
            # If fallback also fails, bind to any interface which will handle by the OS
            print(f"Fallback failed. Binding to any interface. Error: {e2}")
            sock.bind(("", 0))

    sock.sendto(message, (destination, port))
    print(f"Broadcast sent: {message.hex()} to {destination}:{port}")
    sock.close()

# Define the syn packet function here
def send_syn_packet_in_background(client_ip, server_ip, source_port, destination_port):
    try:
        # Call the send_syn_packet function in a new thread
        send_syn_packet(client_ip, server_ip, source_port, destination_port)
    except Exception as e:
        print(f"Error in background thread: {e}")

def start_syn_packet_thread(client_ip, server_ip, port, destination_port):
    """Run the send_syn_packet in the background"""
    syn_thread = threading.Thread(target=send_syn_packet_in_background, args=(client_ip, server_ip, port, destination_port))
    syn_thread.daemon = True  # Allow the thread to exit when the main program exits
    syn_thread.start()

def initialize_databases():
    print("Creating new databases and initializing default values.")
    # settings.db
    with shelve.open('settings.db', 'c') as db:
        setting = Settings('0830', '1800', 1, 100, 1, 60)
        db['Time_Record'] = {'Time_Record_Info': setting}
        db['Generate_Status'] = False

        email_setup = Email('iatfadteam@gmail.com', 'iatfadteam@gmail.com', 'pmtu cilz uewx xqqi', 3)
        db['Email_Data'] = {'Email_Info': email_setup}

        db['Port'] = random.randint(50001, 65535)

    # line_chart_data.db
    with shelve.open('line_chart_data.db', 'c') as db:
        today = datetime.today()
        chart_data = {}
        for i in range(7):
            day = today - timedelta(days=i)
            chart_data[day.strftime("%Y-%m-%d")] = Line_Chart_Data(day, 0)
        db['Line_Chart_Data'] = chart_data

def update_existing_databases():
    print("Updating existing database values if needed.")

    with shelve.open('users.db', 'c') as db:
        for key in db:
            user_data = db[key]
            if 'status' not in user_data:
                user_data['status'] = 'Active'
                db[key] = user_data

    with shelve.open('settings.db', 'w') as db:
        db['Port'] = random.randint(53100, 53199)
        db['last_ip_id'] = 7500

    with shelve.open('line_chart_data.db', 'w') as db:
        Line_Chart_Data_dict = db.get('Line_Chart_Data', {})
        today = datetime.today()
        current_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if current_date not in Line_Chart_Data_dict:
            Line_Chart_Data_dict[current_date] = Line_Chart_Data(current_date, 0)
        db['Line_Chart_Data'] = Line_Chart_Data_dict


def setup_mail():
    print("Configuring mail...")
    app.config['MAIL_USERNAME'] = 'iatfadteam@gmail.com'
    app.config['MAIL_PASSWORD'] = 'pmtu cilz uewx xqqi'
    app.config['MAIL_DEFAULT_SENDER'] = ('Admin', 'iatfadteam@gmail.com')
    return Mail(app)


def start_threads():
    print("Starting all system threads...")

    capture_thread = threading.Thread(target=capture_frames) # Capture frames from the camera
    video_thread = threading.Thread(target=video_processing_loop) # Process the captured frames for video feed
    feeding_thread = threading.Thread(target=feeding_scheduler_loop) # Schedule feeding tasks based on the configured times
    schedule_thread = threading.Thread(target=schedule_daily_task) # Reschedule feeding alerts daily
    validate_thread = threading.Thread(target=validate_config_thread) # Validate the configuration and settings periodically

    capture_thread.start()
    time.sleep(5)  # ensure camera is ready
    video_thread.start()
    feeding_thread.start()
    schedule_thread.start()
    validate_thread.start()

    return [capture_thread, video_thread, feeding_thread, schedule_thread]


def cleanup_on_exit():
    print("Cleaning up and closing ports...")
    stop_event.set()

    try:
        with shelve.open('settings.db', 'r') as db:
            port = db.get('Port')
            server_isn = db.get('syn_ack_seq')
            server_ack = db.get('syn_ack_ack')

        with shelve.open("IP.db") as db:
            ip_data = db.get("IP", {})
            source_ip = ip_data.get("source", "N/A")
            destination_ip = ip_data.get("destination", "N/A")

        send_RST_packet(port, server_isn, server_ack, source_ip, destination_ip)
        print("Port closed successfully.")
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to send RST packet or close ports: {e}")


if __name__ == '__main__':
    try:
        # Check if DB is readable
        with shelve.open('settings.db', 'r') as _:
            pass
        update_existing_databases()

    except Exception as e:
        print(f"[INIT ERROR] Could not open DB, initializing new one: {e}")
        initialize_databases()

    # Setup mail
    mail = setup_mail()

    # Start all system threads
    threads = start_threads()

    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        cleanup_on_exit()
        for t in threads:
            t.join()