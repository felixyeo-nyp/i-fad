import wtforms
from wtforms import Form, StringField, RadioField, SelectField, TextAreaField,FloatField, validators, ValidationError, IntegerField, FileField,DecimalField
from wtforms.fields import EmailField, DateField, PasswordField
from werkzeug.utils import secure_filename
from wtforms.validators import Email, Length, DataRequired, Regexp, EqualTo, NumberRange, Optional, IPAddress
from wtforms import StringField, validators, RadioField
from flask_wtf import FlaskForm
import re

class configurationForm(FlaskForm):
    first_timer = StringField("", validators=[
        DataRequired(),
        Regexp(r'^(0[6-9]|1[0-2])[0-5]\d$', message="First timer must be between 0600 and 1200 (24Hr format).")
    ])
    second_timer = StringField("", validators=[
        DataRequired(),
        Regexp(r'^(1[2-9]|2[0-3])[0-5]\d$', message="Second timer must be between 1200 and 2400 (24Hr format).")
    ])

    interval_minutes = IntegerField("", validators=[
        Optional(),
        NumberRange(min=0, max=60, message="Minutes must be non-negative.")
    ])
    interval_seconds = IntegerField("", validators=[
        Optional(),
        NumberRange(min=0, max=59, message="Seconds must be between 0 and 59.")
    ])

    pellets = IntegerField("", validators=[
        DataRequired(),
        NumberRange(min=100, max=1000, message="Pellets must be between 100 and 1000 grams.")
    ])
    minutes = IntegerField("", validators=[
        Optional(),
        NumberRange(min=0, max=60, message="Minutes must be non-negative.")
    ])

    def validate(self):
        # Call the parent class validate method first
        is_valid = super().validate()

        # Custom validation logic: Ensure at least one of minutes or seconds is entered
        if not self.minutes.data:
            self.minutes.errors.append("At least one of 'minutes' or 'seconds' must be entered.")
            return False

        return is_valid
    def intervalvalidate(self):
        # Call the parent class validate method first
        is_valid = super().validate()

        # Custom validation logic: Ensure at least one of minutes or seconds is entered
        if not self.interval_minutes.data:
            self.interval_minutes.errors.append("Please enter the data.")

            return False


    def totalintervalvalidcheck(self):
        is_valid = super().validate()
        interval_total = (self.interval_minutes.data or 0) * 60
        feeding_total = (self.minutes.data or 0) * 60

        if interval_total > feeding_total:
            self.interval_minutes.errors.append("Interval check duration cannot exceed total feeding duration.")
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
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=25)])
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

class DeleteForm(FlaskForm):
    """Empty form, just for CSRF protection on deletes."""
    pass


