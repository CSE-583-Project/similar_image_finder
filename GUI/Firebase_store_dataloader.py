import firebase_admin
from firebase_admin import credentials, storage
cred = credentials.Certificate("final-project-583-d41b1-1863972b1d49.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'final-project-583-d41b1.appspot.com'}) # connecting to firebase

file_path = "images/nyc.jpeg"
bucket = storage.bucket() # storage bucket
blob = bucket.blob(file_path)
blob.upload_from_filename(file_path)
