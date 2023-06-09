import hashlib
import os
import re


class Util:
    def generate_salt():
        return os.urandom(16)

    def generate_hash(password, salt):
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000,
            dklen=16
        )
        return key
    
    def is_strong_password(password):
       
        if len(password) < 8 or (not (re.search('[a-zA-Z]', password) and re.search('[0-9]', password) and (re.search('[!@#?]', password)))):
            return False
        
        return True
