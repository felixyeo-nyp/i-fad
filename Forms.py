import wtforms
from wtforms import Form, StringField, RadioField, SelectField, TextAreaField,FloatField, validators, ValidationError, IntegerField, FileField,DecimalField
from wtforms.fields import EmailField, DateField, PasswordField
from werkzeug.utils import secure_filename
from wtforms.validators import Email, Length, DataRequired, Regexp, EqualTo, NumberRange, Optional, IPAddress
from wtforms import StringField, validators, RadioField
from flask_wtf import FlaskForm
import re

class configurationForm(FlaskForm):
    morning_feed_1 = StringField("", validators=[
        DataRequired(),
        Regexp(r'^(0[0-9]|1[0-1])[0-5]\d$', message="Morning Feed 1 must be between 0000 and 1159.")
    ])
    morning_feed_2 = StringField("", validators=[
        Optional(),
        Regexp(r'^(0[0-9]|1[0-1])[0-5]\d$', message="Morning Feed 2 must be between 0000 and 1159.")
    ])

    evening_feed_1 = StringField("", validators=[
        DataRequired(),
        Regexp(r'^(1[2-9]|2[0-3])[0-5]\d$', message="Evening Feed 1 must be between 1200 and 2359.")
    ])
    evening_feed_2 = StringField("", validators=[
        Optional(),
        Regexp(r'^(1[2-9]|2[0-3])[0-5]\d$', message="Evening Feed 2 must be between 1200 and 2359.")
    ])

    interval_seconds = IntegerField("", validators=[
        DataRequired(),
        NumberRange(min=1, max=60, message="Interval seconds must be between 1 and 60 seconds.")
    ])

    pellets = IntegerField("", validators=[
        DataRequired(),
        NumberRange(min=1, max=1000, message="Feeding amount must be between 1 and 1000 grams.")
    ])

    minutes = IntegerField("", validators=[
        Optional(),
        NumberRange(min=1, max=60, message="Feeding duration must be between 1 and 60 minutes")
    ])

    feeding_threshold = IntegerField("", validators=[
        DataRequired(),
        NumberRange(min=1, max=1000, message="Feeding threshold must be between 1 and 1000 pellets.")
    ])

    pellet_size = IntegerField("", validators=[
        DataRequired(),
        NumberRange(min=1, max=8, message="Pellet size must be between 1 to 8 mm.")
    ])

    pellets_per_second = IntegerField("", validators=[
        DataRequired(),
        NumberRange(min=1, max=1000, message="Pellets dispense out per second must be between 1 to 1000 gram(s)/s.")
    ])

    def validate(self):
        is_valid = super().validate()
        if not is_valid:
            return False

        total_feed_grams = self.pellets.data
        dispense_rate = self.pellets_per_second.data
        threshold = self.feeding_threshold.data
        check_interval = self.interval_seconds.data
        duration = self.duration_seconds.data

        if dispense_rate <= 0:
            self.pellets_per_second.errors.append("Pellets per second must be greater than 0.")
            return False

        # Total time needed for requested pellets
        required_feed_time = total_feed_grams / dispense_rate

        total_required_time = required_feed_time + check_interval

        if total_required_time > duration:
            self.duration_seconds.errors.append(
                f"Feeding session too short. Requires at least {int(total_required_time)} seconds for {total_feed_grams}g feed amount."
            )
            return False
        
        if total_feed_grams < threshold:
            self.feeding_threshold.errors.append(
                "Feeding threshold must be less than the total feed amount."
            )
            return False

        return is_valid

class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=25)])
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=150)])
    password = PasswordField("Password", validators=[
        DataRequired(),
        Length(min=8, max=80),
        Regexp(r'(?=.*[A-Z])', message='Password must contain at least one uppercase letter.'),
        Regexp(r'(?=.*[a-z])', message='Password must contain at least one lowercase letter.'),
        Regexp(r'(?=.*\d)', message='Password must contain at least one number.'),
        Regexp(r'(?=.*[!@#$%^&*(),.?":{}|<>])', message='Password must contain at least one special character.')
    ])
    confirm_password = PasswordField("Confirm Password", validators=[
        DataRequired(), Length(min=8, max=80),
        EqualTo('password', message='Passwords must match')
    ])
    role = RadioField("Role", choices=[('Admin', 'Admin'), ('Guest', 'Guest')], validators=[DataRequired()])

class updatepasswordForm(FlaskForm):
    password = PasswordField("Password", validators=[
        DataRequired(),
        Length(min=8, max=80),
        Regexp(r'(?=.*[A-Z])', message='Password must contain at least one uppercase letter.'),
        Regexp(r'(?=.*[a-z])', message='Password must contain at least one lowercase letter.'),
        Regexp(r'(?=.*\d)', message='Password must contain at least one number.'),
        Regexp(r'(?=.*[!@#$%^&*(),.?":{}|<>])', message='Password must contain at least one special character.')
    ])
    confirm_password = PasswordField("Confirm Password", validators=[
        DataRequired(), Length(min=8, max=80),
        EqualTo('password', message='Passwords must match')
    ])

class updateemailrole(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=150)])
    role = RadioField("Role", choices=[('Admin', 'Admin'), ('Guest', 'Guest')], validators=[DataRequired()])
    status = RadioField("Status", choices=[('Active', 'Active'), ('Suspended', 'Suspended')], validators=[DataRequired()])

class forgetpassword(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=150)])

class LoginForm(FlaskForm):
    identifier = StringField("Username Or Email", validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=80)])

class emailForm(FlaskForm):
    sender_email = StringField("", validators=[DataRequired(), Email(), Length(max=150)])
    recipient_email = StringField("", validators=[DataRequired(), Email(), Length(max=150)])
    App_password = StringField("", validators=[DataRequired(), Length(max=40)])
    days = IntegerField("", validators=[DataRequired(), validators.NumberRange(min=3, max=6)])
    confidence = DecimalField("", validators=[DataRequired(), validators.NumberRange(min=1,max = 100)])
class MFAForm(FlaskForm):
    code = StringField('MFA Code', validators=[DataRequired(), Length(min=6, max=6)])

class FeedbackForm(FlaskForm):
    message = TextAreaField("Message", validators=[DataRequired(), Length(max=500)])

class ipForm(FlaskForm):
    source_ip = StringField("Source IP", validators=[DataRequired(), IPAddress()])
    destination_ip = StringField("Destination IP", validators=[DataRequired(), IPAddress()])
    camera_ip = StringField('Camera IP', validators=[DataRequired(), IPAddress()])
    amcrest_username = StringField('Amcrest Username', validators=[DataRequired()])
    amcrest_password = PasswordField('Amcrest Password', validators=[DataRequired()])

class DeleteForm(FlaskForm):
    """Empty form, just for CSRF protection on deletes."""
    pass


