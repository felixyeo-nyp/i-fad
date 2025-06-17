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
    #2: 'Fecal Matters'
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
feeding_lock = threading.Lock()
frame_data_lock = threading.Lock()
object_count_lock = threading.Lock()
freshest_frame_lock = threading.Lock()
total_time_lock = threading.Lock()

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
    cap = cv2.VideoCapture('rtsp://admin:fyp2024Fish34535@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')
    global  freshest_frame
    with freshest_frame_lock:
        freshest_frame = FreshestFrame(cap)
    global latest_processed_frame
    while not stop_event.is_set():
        # Wait for the newest frame
        sequence_num, frame = freshest_frame.read(wait=True)
        if frame is not None:
            # Process the frame
            with latest_processed_frame_lock:
                latest_processed_frame = frame
        time.sleep(0.03)  # Adjust to control frame rate (~30 FPS)

def process_frames():
    # define the dictionary to store the number of pellets
    # Assuming 1 class for 'Pellet'
    global freshest_frame
    global frame_data
    global object_count
    object_count = {1: 0}
    bounding_boxes = []
    global latest_processed_frame
    global feeding
    global feeding_timer
    global total_time
    global total_count
    showing_timer = None
    line_chart_timer, email_TF = (None, False)
    desired_time = None
    total_time = None
    total_count = 0
    formatted_desired_time = None

    frame_counter = 0  # Counter to track frames
    while True:
        db = shelve.open('settings.db', 'w')
        Time_Record_dict = db['Time_Record']
        db.close()

        setting = Time_Record_dict.get('Time_Record_Info')

        checking_interval = setting.get_interval_seconds()

        hours = setting.get_first_timer()[:2]
        minutes = setting.get_first_timer()[2:]
        hours1 = setting.get_second_timer()[:2]
        minutes1 = setting.get_second_timer()[2:]

        first_feeding_time = int(hours)
        first_feeding_time_min = int(minutes)
        second_feeding_time = int(hours1)
        second_feeding_time_min = int(minutes1)

        # change confidence from here.
        confidence = float(setting.get_confidence()) / 100
        current_datetime = datetime.now()
        bounding_boxes = []
        # Process the predictions and update object count
        temp_object_count = {1: 0}  # Initialize count for the current frame

        frame_counter += 1

        # Process only every 5 seconds

        db = shelve.open('settings.db', 'r')
        # Pause for 1 second on each iteration
        if not db.get('Generate_Status', True):
            db.close()
            print("processing at background")
            time.sleep(20)

        else:
            print("Processing a frame...")
            db.close()
            time.sleep(1)
        current_time = datetime.now().time()

        if ((current_time.hour == first_feeding_time and current_time.minute == first_feeding_time_min) or (
                current_time.hour == second_feeding_time and current_time.minute == second_feeding_time_min)) and (
                total_time is None or total_time > int(setting.get_seconds() * 60)) and not feeding:
            with feeding_lock:
                feeding = True
                print("feeding set")
            feeding_timer = None
            showing_timer = None
            starting_timer = None
            line_chart_timer = time.time()
            total_time = None
        with freshest_frame_lock:
            if freshest_frame is not None:
                cnt, frame = freshest_frame.read(sequence_number=object_count[1] + 1)
                if frame is None:
                    break

            # Preprocess the frame
            img_tensor = torchvision.transforms.ToTensor()(frame).to(device)
            img_tensor = img_tensor.unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                predictions = model(img_tensor)

            for i in range(len(predictions[0]['labels'])):
                label = predictions[0]['labels'][i].item()

                if label in class_labels:
                    box = predictions[0]['boxes'][i].cpu().numpy().astype(int) # used to define the size of the object
                    score = predictions[0]['scores'][i].item() #the probability of the object

                    if (label == 1 and score > confidence):
                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) #(0,255,0) is the color (blue, green, yellow)
                        cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        temp_object_count[label] += 1
                        bounding_boxes.append((box, label, score))

                        # Start feeding timer if pellets are detected
                        if label == 1 and feeding_timer is None and feeding:
                            feeding_timer = time.time()
                            starting_timer = time.time()

        # store the pellets number to the object count which is permanently
        for label, count in temp_object_count.items():
            if label == 1:  # Assuming label 1 represents 'Pellets'
                with object_count_lock:
                    object_count[label] = count
                try:
                    with shelve.open('currentcount.db', 'c') as db2:
                        db2['object_count'] = count  # Save the dictionary
                        db2.close()
                except:
                    print("Somethinh wrong")

        # Check feeding timer and switch to stop feeding if required
        if feeding_timer is not None and feeding and starting_timer is not None:
            elapsed_time = time.time() - feeding_timer
            total_time = time.time() - starting_timer

            print(f'Elapsed time since last check: {elapsed_time:.3f} seconds')
            print(f'Total feeding duration so far: {total_time:.3f} seconds')
            print(f"Checking interval: {checking_interval} seconds")

            with object_count_lock, total_time_lock:
                # Ensure we check at correct intervals
                if elapsed_time > checking_interval and total_time <= int(setting.get_seconds() * 60):
                    if int(temp_object_count[1]) < int(setting.get_pellets() // 10):
                        total_count += int(setting.get_pellets())
                        feeding_timer = time.time()  # Reset timer
                        print(f"Feeding again. Total pellets count: {total_count}")

                        # Send TCP packet for continued feeding using the existing function
                        try:
                            db = shelve.open('settings.db', 'r')
                            Time_Record_dict = db['Time_Record']
                            port = db.get('Port')
                            server_isn = db.get('syn_ack_seq')
                            server_ack = db.get('syn_ack_ack')
                            db.close()

                            setting = Time_Record_dict.get('Time_Record_Info')

                            with shelve.open('IP.db', 'r') as ip_db:
                                ip_data = ip_db.get('IP', {})
                                source_ip = ip_data.get('source')
                                destination_ip = ip_data.get('destination')

                            # Create feeding command bytes
                            current_time = datetime.now()

                            start_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)
                            time.sleep(5)
                            with shelve.open("settings.db", 'c') as db, shelve.open("IP.db", 'r') as ip_db:
                                port = db.get('Port')
                                server_isn = db.get('syn_ack_seq')
                                server_ack = db.get('syn_ack_ack')
                                ip_data = ip_db.get("IP", {})
                                source_ip = ip_data.get("source")
                                destination_ip = ip_data.get("destination")
                                Time_Record_dict = db.get('Time_Record', {})
                                db['Time_Record'] = Time_Record_dict
                            stop_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)

                            print("Feed Complete")

                        except Exception as e:
                            print(f"Error sending TCP feeding command: {e}")
                    else:
                        with shelve.open("settings.db", 'c') as db, shelve.open("IP.db", 'r') as ip_db:
                            port = db.get('Port')
                            server_isn = db.get('syn_ack_seq')
                            server_ack = db.get('syn_ack_ack')
                            ip_data = ip_db.get("IP", {})
                            source_ip = ip_data.get("source")
                            destination_ip = ip_data.get("destination")
                            Time_Record_dict = db.get('Time_Record', {})
                            db['Time_Record'] = Time_Record_dict
                        print("Skipping feeding as pellet count is sufficient.")
                        feeding_timer = time.time()  # Still reset, but skip feeding
                        stop_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)

                if total_time > int(setting.get_seconds() * 60) and sum(object_count.values()) != 0:
                    with feeding_lock:
                        feeding = False  # Stop feeding
                    feeding_timer = None
                    showing_timer = time.time()
                    print("Feeding session ended.")
        with object_count_lock:
            # Display the frame with detections and object count
            for label, count in object_count.items():
                text = f'{class_labels[label]} Count: {count}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_position = (frame.shape[1] - text_size[0] - 10, 30 * (label + 1))
                cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 255, 255), 2)
        with object_count_lock:
            # Display feeding or stop feeding text just below the object counter
            text_position_feed = (frame.shape[1] - text_size[0] - 10, 30 * (max(object_count.keys()) + 1))

        if feeding:
            cv2.putText(frame, "Feeding...", text_position_feed,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            if showing_timer is not None:
                i = time.time() - showing_timer

                if i > 3:
                    showing_timer = None
                    j = time.time() - line_chart_timer

                    line_chart_timer = None

                    db = shelve.open('mock_chart_data.db', 'w')
                    current_date = datetime.today().strftime("%Y-%m-%d")
                    if current_date not in db:
                        db[current_date] = {}  # Initialize an empty dictionary for this date

                    # Retrieve the nested dictionary for current_date
                    day_data = db[current_date]

                    # Ensure the nested key ('8:05 AM') exists in the nested dictionary
                    if '8:05 AM' not in day_data:
                        day_data['8:05 AM'] = object_count[1]  # Initialize the count for this time

                    else:
                        day_data['6:05 PM'] = object_count[1]

                    if 'Total' not in day_data:
                        print(f"Total count in db: {total_count}")
                        day_data['Total'] = total_count
                        total_count = 0
                    else:
                        print(f"Total count in db: {total_count}")
                        day_data['Total'] += total_count
                        total_count = 0
                    # Save the updated nested dictionary back to shelve
                    db[current_date] = day_data

                    db.close()

                    if (current_time.hour >= first_feeding_time) and (
                            current_time.hour >= second_feeding_time and current_time.minute > second_feeding_time_min):
                        print('sending email feature')

                    for today_date in Line_Chart_Data_dict:
                        Line_chart_objects = Line_Chart_Data_dict.get(today_date)
                        print(Line_chart_objects.get_date(), ': ', Line_chart_objects.get_timeRecord())

                    print('running in website')
                else:
                    cv2.putText(frame, "Stop Feeding", text_position_feed,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            else:
                if (current_time.hour <= first_feeding_time and current_time.minute <= first_feeding_time_min) or current_time.hour < first_feeding_time:
                    desired_time = current_datetime.replace(hour=first_feeding_time, minute=first_feeding_time_min,
                                                            second=0, microsecond=0)
                    formatted_desired_time = 'Next Round: ' + desired_time.strftime("%I:%M %p")

                    text_size = cv2.getTextSize(formatted_desired_time, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    text_position = (frame.shape[1] - text_size[0] - 10, 30 * 4)
                    cv2.putText(frame, formatted_desired_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 2)

                elif (
                (current_time.hour <= second_feeding_time and current_time.minute <= second_feeding_time_min)) or (current_time.hour < second_feeding_time):
                    desired_time = current_datetime.replace(hour=second_feeding_time, minute=second_feeding_time_min,
                                                            second=0,
                                                            microsecond=0)
                    formatted_desired_time = 'next round: ' + desired_time.strftime("%I:%M %p")

                    text_size = cv2.getTextSize(formatted_desired_time, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    text_position = (frame.shape[1] - text_size[0] - 10, 30 * 4)
                    cv2.putText(frame, formatted_desired_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                else:
                    # Add one day to the current date and time
                    next_day = current_datetime + timedelta(days=1)
                    # Set desired_time to 8 AM of the next day
                    desired_time = next_day.replace(hour=first_feeding_time, minute=first_feeding_time_min, second=0, microsecond=0)

                    formatted_desired_time = 'Tomorrow at: ' + desired_time.strftime("%I:%M %p")

                    text_size = cv2.getTextSize(formatted_desired_time, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    text_position = (frame.shape[1] - text_size[0] - 10, 30 * 4)
                    cv2.putText(frame, formatted_desired_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 0, 255), 2)
        with frame_data_lock:
            frame_data['object_count'] = temp_object_count
            frame_data['bounding_boxes'] = bounding_boxes
        with latest_processed_frame_lock:
            latest_processed_frame = frame

@login_required
def generate_frames():
    global latest_processed_frame, frame_data
    count = 1
    while not stop_event.is_set():
        if count //60 == 0:
            db = shelve.open('settings.db', 'c')
            if not db.get('Generate_Status',True):
                print("stopped generating")
                break
            db.close()
        with latest_processed_frame_lock:
            frame = latest_processed_frame.copy()  # Create a copy of the frame to avoid modification of original
        if frame is not None:
            # Display object count and bounding boxes from frame_data
            with frame_data_lock:
                for label, count in frame_data['object_count'].items():
                    text = f'{class_labels[label]} Count: {count}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    text_position = (frame.shape[1] - text_size[0] - 10, 30 * (label + 1))
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 255, 255), 2)

                # Draw bounding boxes from frame_data
                for box, label, score in frame_data['bounding_boxes']:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert frame to jpeg and yield it
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        time.sleep(0.03)  # Adjust the frame rate if necessary

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
            db.close()
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
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid authentication code', 'danger')

    return render_template('mfa_verify.html', form=form)

@app.route('/logout')
def logout():
    session.clear()
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
    # Load the current ID from a file (or create it if it doesn't exist)
    with shelve.open('settings.db', 'c') as db:
        last_id = db.get('last_ip_id', 7500)  # Default to 1 if no ID is stored
        db['last_ip_id'] = last_id + 1    # Increment the ID for the next packet
        db.close()
    return last_id

def send_tcp_packet(encoded_byte, source_port, server_isn, server_ack, source_ip, dest_ip):
    destination_ip = dest_ip
    source_ip = source_ip
    destination_port = 50000

    payload = bytes.fromhex(encoded_byte)
    flags = "PA"  # PSH + ACK flags
    ip_id = get_next_ip_id()

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
        sniffpacket = sniff(iface="Ethernet 2",
                            filter=f"tcp and src host {destination_ip} and port {destination_port}",
                            count=1, timeout=0.1) # Adjust the Ethernet port name to correspond with the connected interface.

        if not sniffpacket:
            print(f"No packet received. Attempt: {packet_counter}")  # <-- Add this
            if packet_counter == 10:
                print("Timeout reached, no packet captured.")
                break
            continue

        tcp_payload = bytes(sniffpacket[0][TCP].payload).rstrip(b'\x00')
        payload_len = len(tcp_payload)
        print(f"Packet captured: {payload_len} bytes")

        if sniffpacket[0][TCP].flags == "PA" and payload_len > 0:
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
                db.close()

            print("Found correct packet and sent ACK")
            break
        else:
            continue

# Function to perform the 3-way TCP handshake
def send_syn_packet(client_ip, server_ip, source_port, destination_port):
    try:
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

            psh_ack = sniff(iface="Ethernet 2",filter=f"tcp and src host {server_ip} and port {destination_port}", count=1, timeout=1) # Adjust the Ethernet port name to correspond with the connected interface.

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
                    db.close()
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
    destination_port = 50000
    encoded_data = "ccdda10100010001a448"
    send_tcp_packet(
        encoded_byte=encoded_data,
        source_port=source_port,
        server_isn=server_isn,
        server_ack=server_ack,
        source_ip=source_ip,
        dest_ip=destination_ip
    )
    packet_counter = 0
    while True:
        packet_counter += 1
        sniffpacket = sniff(iface="Ethernet 2",
                            filter=f"tcp and src host {destination_ip} and port {destination_port}",
                            count=1, timeout=0.1) # Adjust the Ethernet port name to correspond with the connected interface.

        if not sniffpacket:
            if packet_counter == 10:
                print("not found")
                break
            continue

        tcp_payload = bytes(sniffpacket[0][TCP].payload).rstrip(b'\x00')
        payload_len = len(tcp_payload)
        print(f"Packet captured: {payload_len} bytes")

        if sniffpacket[0][TCP].flags == "PA" and payload_len > 0:
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
                db.close()

            print("Found correct packet and sent ACK")
            break
        else:
            continue

def stop_send_manual_feed(source_port, server_isn, server_ack, source_ip, destination_ip):
    destination_port = 50000
    encoded_data = "ccdda10100000001a346"
    send_tcp_packet(
        encoded_byte=encoded_data,
        source_port=source_port,
        server_isn=server_isn,
        server_ack=server_ack,
        source_ip=source_ip,
        dest_ip=destination_ip
    )
    packet_counter = 0
    while True:
        packet_counter += 1
        sniffpacket = sniff(iface="Ethernet 2",
                            filter=f"tcp and src host {destination_ip} and port {destination_port}",
                            count=1, timeout=0.1) # Adjust the Ethernet port name to correspond with the connected interface.

        if not sniffpacket:
            if packet_counter == 10:
                print("not found")
                break
            continue

        tcp_payload = bytes(sniffpacket[0][TCP].payload).rstrip(b'\x00')
        payload_len = len(tcp_payload)
        print(f"Packet captured: {payload_len} bytes")

        if sniffpacket[0][TCP].flags == "PA" and payload_len > 0:
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
                db.close()

            print("Found correct packet and sent ACK")
            break
        else:
            continue

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
        '19 May 2025': {'8:05 AM': 35, '6:05 PM': 20, 'Total': 450},
        '20 May 2025': {'8:05 AM': 25, '6:05 PM': 30, 'Total': 400},
        '21 May 2025': {'8:05 AM': 15, '6:05 PM': 40, 'Total': 300},
        '22 May 2025': {'8:05 AM': 25, '6:05 PM': 30, 'Total': 250},
        '23 May 2025': {'8:05 AM': 25, '6:05 PM': 30, 'Total': 400},
        '24 May 2025': {'8:05 AM': 15, '6:05 PM': 40, 'Total': 300},
        '25 May 2025': {'8:05 AM': 45, '6:05 PM': 35, 'Total': 275},
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
                first_feed_counts.append(db[day].get('8:05 AM', 0))
                second_feed_counts.append(db[day].get('6:05 PM', 0))
                total_feed_counts.append(db[day].get('Total', 0))
            else:
                first_feed_counts.append(0)
                second_feed_counts.append(0)
                total_feed_counts.append(0)

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

    with shelve.open('currentcount.db', 'r') as db2:
        object_count = db2.get('object_count', 0)  # Get count dictionary, default to empty
        db2.close()
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

@app.route('/update', methods=['GET', 'POST'])
@login_required
def update_setting():

    setting = configurationForm(request.form)
    print("check2")
    if request.method == 'POST' and setting.validate():
        manual_feed_action = request.form.get('manual_feed_action')
        pattern = r'^(?:[01]\d[0-5]\d|2[0-3][0-5]\d)$'  # Matches HHMM format
        print("check3")

        if manual_feed_action in ["start", "stop"]:
            with shelve.open("settings.db", 'c') as db, shelve.open("IP.db", 'r') as ip_db:
                port = db.get('Port')
                server_isn = db.get('syn_ack_seq')
                if server_isn is None:
                    flash("⚠️ Missing 'syn_ack_seq' in settings. Please initialize it first.", "danger")
                    return redirect(url_for('update_setting'))
                server_ack = db.get('syn_ack_ack')
                ip_data = ip_db.get("IP", {})
                source_ip = ip_data.get("source")
                destination_ip = ip_data.get("destination")
                Time_Record_dict = db.get('Time_Record', {})
                db['Time_Record'] = Time_Record_dict

            if manual_feed_action == "start":
                start_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)
                flash("Manual feeding started.", "success")

            elif manual_feed_action == "stop":
                stop_send_manual_feed(port, server_isn, server_ack, source_ip, destination_ip)
                flash("Manual feeding stopped.", "info")

            return redirect(url_for('update_setting'))

        elif re.match(pattern, setting.first_timer.data) and re.match(pattern, setting.second_timer.data):
            try:
                first_hour = int(setting.first_timer.data[:2])
                first_minute = int(setting.first_timer.data[2:])

                second_hour = int(setting.second_timer.data[:2])
                second_minute = int(setting.second_timer.data[2:])

                print(first_hour)
                print(first_minute)
                print(second_hour)
                print(second_minute)


                # Validate ranges
                if (6 <= first_hour <= 12) and (12 <= second_hour <= 24):
                    print("check4")
                    time.sleep(0.5)

                    db = shelve.open('settings.db', 'c')
                    Time_Record_dict = db.get('Time_Record', {})

                    j = Time_Record_dict.get('Time_Record_Info')

                    j.set_first_timer(setting.first_timer.data)
                    j.set_second_timer(setting.second_timer.data)

                    try:
                        if setting.interval_minutes.data is not None or setting.interval_seconds.data is not None:
                            total_intervalinsec = 0
                            if setting.interval_minutes.data:
                                total_intervalinsec += int(
                                    setting.interval_minutes.data) * 60
                            if setting.interval_seconds.data:
                                total_intervalinsec += int(setting.interval_seconds.data)

                            j.set_interval_seconds(total_intervalinsec)
                            print(f"Updated interval time: {total_intervalinsec} seconds ({total_intervalinsec // 60} min {total_intervalinsec % 60} sec)")

                        else:
                            j.set_interval_seconds(0)
                    except ValueError:
                        print("Invalid interval input detected.")  # Debugging message

                    j.set_pellets(setting.pellets.data)

                    if setting.minutes.data is not None:
                        total_seconds = int(setting.minutes.data)
                        j.set_seconds(total_seconds)
                    else:
                        j.set_seconds(0)

                    port = db['Port']
                    server_isn = db['syn_ack_seq']
                    server_ack = db['syn_ack_ack']

                    db['Time_Record'] = Time_Record_dict
                    db.close()

                    feeding_duration = setting.minutes.data

                    # Calculate the end time for the first feeding session
                    end_hour = first_hour + (first_minute + feeding_duration) // 60
                    end_minute = (first_minute + feeding_duration) % 60

                    # Ensure it does not exceed 24:00
                    if end_hour >= 24:
                        end_hour = 24
                        end_minute = 0

                    encoded_byte = encode_time(0x01, first_hour, first_minute, end_hour, end_minute)

                    # Calculate the end time for the second feeding session
                    end_hour2 = second_hour + (second_minute + feeding_duration) // 60
                    end_minute2 = (second_minute + feeding_duration) % 60

                    # Ensure it does not exceed 24:00
                    if end_hour2 >= 24:
                        end_hour2 = 24
                        end_minute2 = 0

                    encoded_byte2 = encode_time(0x02, second_hour, second_minute, end_hour2, end_minute2)

                    print(encoded_byte2)

                    print(setting.first_timer.data)
                    with shelve.open("IP.db") as db:
                        # Check if "IP" key exists
                        if "IP" in db:
                            ip_data = db["IP"]  # Retrieve the dictionary
                            source_ip = ip_data.get("source", "Source IP not found")
                            destination_ip = ip_data.get("destination", "Destination IP not found")
                            print("Source IP:", source_ip)
                            print("Destination IP:", destination_ip)
                        else:
                            print("No IP data found.")
                    send_tcp_packet(encoded_byte, port, server_isn, server_ack,source_ip,destination_ip)
                    print(f"Encoded packet 1: {encoded_byte}")
                    time.sleep(2)
                    db = shelve.open('settings.db', 'c')
                    port = db['Port']
                    server_isn = db['syn_ack_seq']
                    server_ack = db['syn_ack_ack']

                    db['Time_Record'] = Time_Record_dict
                    db.close()

                    print(setting.second_timer.data)
                    send_tcp_packet(encoded_byte2, port, server_isn, server_ack,source_ip,destination_ip)
                    print(f"Encoded packet 2: {encoded_byte}")

                    user_email = session.get("user_email")
                    first_timer = setting.first_timer.data
                    second_timer = setting.second_timer.data
                    total_minute = setting.minutes.data // 60

                    # Pass total_seconds to the scheduling function
                    schedule_feeding_alerts(first_timer, second_timer, total_minute, user_email)
                    return redirect(url_for('dashboard'))
                else:
                    if not (6 <= first_hour <= 12):
                        setting.first_timer.errors.append('First timer should be between 0600 and 1200.')
                    if not (12 <= second_hour <= 24):
                        setting.second_timer.errors.append('Second timer should be between 1200 and 2400.')
            except ValueError:
                setting.first_timer.errors.append('Invalid time format. Please use HHMM format.')
                setting.second_timer.errors.append('Invalid time format. Please use HHMM format.')
        else:
            if not re.match(pattern, setting.first_timer.data):
                setting.first_timer.errors.append('Invalid time format. Please use HHMM format.')
            if not re.match(pattern, setting.second_timer.data):
                setting.second_timer.errors.append('Invalid time format. Please use HHMM format.')

        return render_template('settings.html', form=setting)
    else:
        time.sleep(0.5)
        Time_Record_dict = {}
        db = shelve.open('settings.db', 'r')
        Time_Record_dict = db['Time_Record']
        db.close()

        j = Time_Record_dict.get('Time_Record_Info')
        setting.first_timer.data = j.get_first_timer()
        setting.second_timer.data = j.get_second_timer()
        total_intervalinsec = j.get_interval_seconds()

        if total_intervalinsec:
            setting.interval_minutes.data = total_intervalinsec // 60
            setting.interval_seconds.data = total_intervalinsec % 60
        else:
            setting.interval_minutes.data = None
            setting.interval_seconds.data = None

        setting.pellets.data = j.get_pellets()

        setting.minutes.data = j.get_seconds() // 60

        return render_template('settings.html', form=setting)

def send_feeding_complete_email(user_email, feed_time):
    with app.app_context():
        try:
            msg = flask_mail.Message(subject="Feeding Complete",
                          recipients=["testproject064@gmail.com"],
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
    try:
        user_email = session.get("email")
    except:
        db = shelve.open('settings.db', 'r')
        email_db = db.get("Email_Data", {"Email_Info":Email("iatfadteam@gmail.com","testproject064@gmail.com",'pmtu cilz uewx xqqi',3)})
        email_instance = email_db.get("Email_Info")
        user_email = email_instance.get_recipient_email()
        print("reschedule"+ user_email)

    # Calculate new run_date for the first alert (next day)
    now = datetime.now()
    timezone = pytz.timezone("Asia/Singapore")
    first_timer_dt = now.replace(hour=int(first_timer[:2]), minute=int(first_timer[3:]), second=0, microsecond=0)
    first_end_time = timezone.localize(first_timer_dt + timedelta(seconds=feeding_duration))
    # Reschedule the feeding alerts for the next day

    # Modify the existing job with the new run_date
    job = scheduler.get_job('first_feeding_alert')  # Retrieve the existing job by its ID
    if job:
        job.modify(run_date=first_end_time)  # Reschedule the job for the new time
    else:
        scheduler.add_job(
            func=send_feeding_complete_email,
            trigger='date',
            run_date=first_end_time,
            args=[user_email, "first feeding complete"],
            id='first_feeding_alert',
            misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
        )
        print("No job found with this ID!")

    # Repeat the process for the second timer
    second_timer_dt = now.replace(hour=int(second_timer[:2]), minute=int(second_timer[3:]), second=0, microsecond=0)
    second_end_time = timezone.localize(second_timer_dt + timedelta(seconds=feeding_duration))

    # Modify the second feeding alert job
    job = scheduler.get_job('second_feeding_alert')  # Retrieve the existing job by its ID
    if job:
        job.modify(run_date=second_end_time)  # Reschedule the job for the new time
    else:
        scheduler.add_job(
            func=send_feeding_complete_email,
            trigger='date',
            run_date=second_end_time,
            args=[user_email, "second feeding complete"],
            id='second_feeding_alert',
            misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
        )
        print("No job found with this ID2!")

def schedule_daily_task():
    while True:
        print("Updating schedule")
        reschedule_feeding_alerts()  # Execute the function
        time.sleep(86400)  # Wait for 24 hours (86400 seconds)

def schedule_feeding_alerts(first_timer, second_timer, feeding_duration, user_email):
    try:
        now = datetime.now()  # Use current date for scheduling
        first_timer_dt = now.replace(hour=int(first_timer[:2]), minute=int(first_timer[2:]), second=0, microsecond=0)
        second_timer_dt = now.replace(hour=int(second_timer[:2]), minute=int(second_timer[2:]), second=0, microsecond=0)

        feeding_duration = int(feeding_duration)

        # Localize the datetime objects
        timezone = pytz.timezone("Asia/Singapore")
        first_end_time = timezone.localize(first_timer_dt + timedelta(seconds=feeding_duration))
        second_end_time = timezone.localize(second_timer_dt + timedelta(seconds=feeding_duration))

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
            print("Job added.")

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

    if form.validate_on_submit():
        try:
            administration = ["iatfadteam@gmail.com"] # change to admin email
            # Attempt to compose and send the email
            msg = flask_mail.Message(
                subject="New Feedback",
                sender=user_email,
                recipients=administration,
                body=f"Name: {user_name}\nEmail: {user_email}\nMessage:\n{form.message.data}"
            )
            print(f"message sent to {administration}")
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
                        db.close()
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
                if search_query in username.lower() or search_query in data.get('email', '').lower()
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
                flash('User not found.', 'danger')
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
    setting = ipForm(request.form)
    if request.method == 'POST' and setting.validate():
        source_ip = setting.source_ip.data
        destination_ip = setting.destination_ip.data
        print(f"Source IP: {source_ip}, Destination IP: {destination_ip}")

        db = shelve.open('settings.db', 'w')
        port = db['Port']
        db.close()

        start_syn_packet_thread(source_ip, destination_ip, port, 50000)

        send_udp_packet(source_ip, "255.255.255.255", 60000, b"\x00\x00\x00\x00\x00")
        db = shelve.open('IP.db', 'n')
        db["IP"] = {"source":source_ip , "destination":destination_ip}
        db.close()
        return redirect(url_for('update_setting'))

    return render_template('setIP.html', form=setting)

def send_udp_packet(source, destination, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    sock.bind((source, 0))

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

# Main code
if __name__ == '__main__':
    try:
        # Attempt to open the shelve database file for reading
        print("Attempting to open the database file for reading.")
        print("Database file opened for reading.")
        print("Attempting to open the database file for updating...")

        with shelve.open('users.db', 'c') as db:
            for key in db:
                user_data = db[key]
                print(user_data)

                if 'status' not in user_data:  # Check if 'role' field exists
                    user_data['status'] = 'Active'  # Set default role (e.g., 'Admin' or 'Guest')
                    db[key] = user_data  # Update the record
                    print(f"Added 'status' field to user: {key}")

        print("Database file updated successfully.")

        db = shelve.open('settings.db', 'w')
        # Attempt to get 'Time_Record' from db, if not found, initialize with empty dictionary
        Time_Record_dict = db.get('Time_Record',{})
        Email_dict = db.get('Email_Data', {})
        Generate_Status = db.get('Generate_Status', False)
        email_setup = Email_dict['Email_Info']

        # app.config['MAIL_USERNAME'] = email_setup.get_sender_email()
        # app.config['MAIL_PASSWORD'] = email_setup.get_APPPassword()
        # app.config['MAIL_DEFAULT_SENDER'] = ('admin', email_setup.get_sender_email())

        app.config['MAIL_USERNAME'] = 'iatfadteam@gmail.com'
        app.config['MAIL_PASSWORD'] = 'pmtu cilz uewx xqqi'
        app.config['MAIL_DEFAULT_SENDER'] = ('Admin', 'iatfadteam@gmail.com')

        mail = Mail(app)
        # newly added read config email from database
        port = random.randint(53100, 53199)
        db['Port'] = port
        db['last_ip_id'] = 7500
        # Start the syn packet in a separate thread
        db.close()

        db = shelve.open('line_chart_data.db', 'w')
        Line_Chart_Data_dict = db.get('Line_Chart_Data',{})  # Attempt to get 'Time_Record' from db, if not found, initialize with empty dictionary
        current_date = (datetime.today()+timedelta(days=1)).strftime("%Y-%m-%d")

        if current_date not in Line_Chart_Data_dict:
            linechart = Line_Chart_Data(current_date, 0)
            Line_Chart_Data_dict[current_date] = linechart
            db['Line_Chart_Data'] = Line_Chart_Data_dict

        ###### test code ####
        today = datetime.today()
        current_date = today - timedelta(days=3)
        current_date1 = today - timedelta(days=2)
        current_date2 = today - timedelta(days=1)
        current_date3 = today

        print(current_date3,'current')

        if current_date3.strftime("%Y-%m-%d") == '2024-05-03':
            oject = Line_Chart_Data_dict.get(current_date3.strftime("%Y-%m-%d"))
            oject.set_timeRecord(0)

            oject1 = Line_Chart_Data_dict.get(current_date2.strftime("%Y-%m-%d"))
            oject1.set_timeRecord(80)

            oject2 = Line_Chart_Data_dict.get(current_date1.strftime("%Y-%m-%d"))
            oject2.set_timeRecord(300)

            oject3 = Line_Chart_Data_dict.get(current_date.strftime("%Y-%m-%d"))
            oject3.set_timeRecord(500)

        db['Line_Chart_Data'] = Line_Chart_Data_dict
        db.close()

        print('the Date you have:\n-------------------------------------------------------')
        for i in Line_Chart_Data_dict:
            print(i,': ', (Line_Chart_Data_dict.get(i).get_timeRecord()))
        print('-------------------------------------------------------')

        # Start the threads for capturing frames and processing frames
        capture_thread = threading.Thread(target=capture_frames)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Set thread >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        inference_thread = threading.Thread(target=process_frames)
        update_schedule_thread = threading.Thread(target=schedule_daily_task)

        # Start the threads
        capture_thread.start()
        time.sleep(5)
        inference_thread.start()
        update_schedule_thread.start()
    except:
        mail = Mail(app)
        # If the file doesn't exist, create a new one
        print("Database file does not exist. Creating a new one.")
        db = shelve.open('settings.db', 'c')

        # create the basic setting for new user
        setting =Settings('0830', '1800' ,1, 100,10,60)
        Time_Record_dict['Time_Record_Info'] = setting
        db['Time_Record'] = Time_Record_dict
        db['Generate_Status'] = False
        db['Check_Interval'] = 10

        # create the basic email setup for user
        email_sender = 'iatfadteam@gmail.com'
        email_password = 'pmtu cilz uewx xqqi'
        email_receiver = 'testproject064@gmail.com'
        email_setup = Email(email_sender, email_receiver, email_password, 3)
        Email_dict['Email_Info'] = email_setup
        db['Email_Data'] = Email_dict
        port = random.randint(50001, 65535)
        db['Port'] = port

        # close the db
        db.close()

        #  create the line chart database
        db = shelve.open('line_chart_data.db', 'c')
        # Get today's date
        today = datetime.today()
        for i in range(7):
            # Calculate the date for the current iteration
            current_date = today - timedelta(days=i)

            # Generate data for the current date
            linechart = Line_Chart_Data(current_date, 0)

            # Store the data in the dictionary
            Line_Chart_Data_dict[current_date.strftime("%Y-%m-%d")] = linechart
        db['Line_Chart_Data'] = Line_Chart_Data_dict
        db.close()
        capture_thread = threading.Thread(target=capture_frames)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Set thread >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        inference_thread = threading.Thread(target=process_frames)
        update_schedule_thread = threading.Thread(target=schedule_daily_task)

        # Start the threads
        capture_thread.start()
        time.sleep(5)
        inference_thread.start()
        update_schedule_thread.start()
    try:
        mail = Mail(app)
        app.run(host='0.0.0.0', port=8080, debug=True)
    finally:
        # Stop threads on exit
        stop_event.set()
        capture_thread.join()
        inference_thread.join()
        print("Closing the port after application finishes...")
        db = shelve.open('settings.db', 'w')
        port = db['Port']
        server_isn = db['syn_ack_seq']
        server_ack = db['syn_ack_ack']
        db.close()
        with shelve.open("IP.db") as db:
            if "IP" in db:
                ip_data = db["IP"]  # Retrieve the dictionary
                source_ip = ip_data.get("source", "Source IP not found")
                destination_ip = ip_data.get("destination", "Destination IP not found")
                print("Source IP:", source_ip)
                print("Destination IP:", destination_ip)
            else:
                print("No IP data found.")
        send_RST_packet(port, server_isn, server_ack,source_ip, destination_ip)  # Send a FIN packet to close the connection
        print("Port closed.")