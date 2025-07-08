import shelve
from datetime import datetime
from zoneinfo import ZoneInfo

class Settings:
    def __init__(self, first_timer='', second_timer='', interval_seconds=0, pellets=0, seconds=0, confidence=0.0):
        self.__first_timer = first_timer
        self.__second_timer = second_timer
        self.__interval_seconds = interval_seconds
        self.__pellets = pellets
        self.__seconds = seconds
        self.__confidence = confidence

    # Getter methods
    def get_first_timer(self):
        return self.__first_timer

    def get_second_timer(self):
        return self.__second_timer

    def get_interval_seconds(self):
        return self.__interval_seconds

    def get_pellets(self):
        return self.__pellets

    def get_seconds(self):
        return self.__seconds

    def get_confidence(self):
        return self.__confidence

    # Setter methods
    def set_first_timer(self, first_timer):
        self.__first_timer = first_timer

    def set_second_timer(self, second_timer):
        self.__second_timer = second_timer

    def set_interval_seconds(self, interval_seconds):
        self.__interval_seconds = interval_seconds

    def set_pellets(self, pellets):
        self.__pellets = pellets

    def set_seconds(self, seconds):
        self.__seconds = seconds

    def set_confidence(self, confidence):
        self.__confidence = confidence

class Line_Chart_Data():
    def __init__(self, date, TimeRecord):
        self.__date = date
        self.__TimeRecord = TimeRecord

    def set_date(self, date):
        self.__date = date

    def set_timeRecord(self, timeRecord):
        self.__TimeRecord = timeRecord

    def get_date(self):
        if isinstance(self.__date, str):
            date_obj = datetime.strptime(self.__date, "%Y-%m-%d")
        else:
                date_obj = self.__date

        # Format the datetime object as a string
        return date_obj.strftime("%Y-%m-%d")

    def get_timeRecord(self):
        return self.__TimeRecord


class Email():
    def __init__(self, sender_email, recipient_email, APPPassword, days):
        self.__sender_email = sender_email
        self.__recipient_email = recipient_email
        self.__APPPassword = APPPassword
        self.__days = days

    # Accessor method for sender_email
    def get_sender_email(self):
        return self.__sender_email

    # Mutator method for sender_email
    def set_sender_email(self, new_sender_email):
        self.__sender_email = new_sender_email

    # Accessor method for recipient_email
    def get_recipient_email(self):
        return self.__recipient_email

    # Mutator method for recipient_email
    def set_recipient_email(self, new_recipient_email):
        self.__recipient_email = new_recipient_email

    # Accessor method for APPPassword
    def get_APPPassword(self):
        return self.__APPPassword

    # Mutator method for APPPassword
    def set_APPPassword(self, new_APPPassword):
        self.__APPPassword = new_APPPassword

    # Accessor method for days
    def get_days(self):
        return self.__days

    # Mutator method for days
    def set_days(self, new_days):
        self.__days = new_days


class Line_Chart_Data_Pellets():
    def __init__(self, date, pellets):
        self.__date = date
        self.__pellets = pellets

    def set_date(self, date):
        self.__date = date

    def set_pellets(self, pellets):
        self.__pellets = pellets

    def get_date(self):
        return self.__date.strftime("%Y-%m-%d")

    def get_pellets(self):
        return self.__pellets

SHELVE_PATH    = 'feedback.db'
_COUNTER_KEY   = 'counter'


class Feedback:
    def __init__(
        self,
        fb_id: int = 0,
        user_name: str = '',
        user_email: str = '',
        message: str = '',
        submitted_at: datetime = None
    ):
        self.__id = fb_id
        self.__user_name = user_name
        self.__user_email = user_email
        self.__message = message
        # default to Singapore time (UTC+8)
        sg_tz = ZoneInfo('Asia/Singapore')
        self.__submitted_at = submitted_at or datetime.now(sg_tz)

    # Getter methods
    def get_id(self) -> int:
        return self.__id

    def get_user_name(self) -> str:
        return self.__user_name

    def get_user_email(self) -> str:
        return self.__user_email

    def get_message(self) -> str:
        return self.__message

    def get_submitted_at(self) -> datetime:
        return self.__submitted_at

    # Setter methods
    def set_user_name(self, name: str):
        self.__user_name = name

    def set_user_email(self, email: str):
        self.__user_email = email

    def set_message(self, msg: str):
        self.__message = msg

    def set_submitted_at(self, dt: datetime):
        self.__submitted_at = dt


class FeedbackStore:
    def __init__(self, path: str = SHELVE_PATH):
        self.path = path
        # Ensure database exists and counter is initialized
        with shelve.open(self.path, writeback=True) as db:
            if _COUNTER_KEY not in db:
                db[_COUNTER_KEY] = 0

    def _open(self, writeback: bool = False):
        return shelve.open(self.path, writeback=writeback)

    def add(self, user_name: str, user_email: str, message: str) -> Feedback:
        with self._open(writeback=True) as db:
            next_id = db.get(_COUNTER_KEY, 0) + 1
            db[_COUNTER_KEY] = next_id

            fb = Feedback(
                fb_id=next_id,
                user_name=user_name,
                user_email=user_email,
                message=message
            )
            db[str(next_id)] = fb
        return fb

    def list_all(self) -> list[Feedback]:
        with self._open() as db:
            entries = [db[k] for k in db if k != _COUNTER_KEY]
        entries.sort(
            key=lambda e: (e.get_user_email(), e.get_submitted_at()),
            reverse=True
        )
        return entries

    def get(self, fb_id: int) -> Feedback | None:
        with self._open() as db:
            return db.get(str(fb_id))

    def delete(self, fb_id: int) -> bool:
        with self._open(writeback=True) as db:
            key = str(fb_id)
            if key in db:
                del db[key]
                return True
        return False
