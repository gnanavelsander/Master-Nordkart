

import firebase_admin
from firebase_admin import credentials, firestore
import sys 
from pathlib import Path




ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#get the service keys
cred_path = Path(__file__).resolve().parent / "serviceAccountKey.json"

#inilize firebase
def init_firebase():
    # Prevent multiple initializations
    if not firebase_admin._apps:

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Create Firestore client instance
fs = init_firebase()


