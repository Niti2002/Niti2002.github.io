import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate('C:/Users/aashi/Dropbox/Danjgo From Utube/imageupload/railworldproject-49942-firebase-adminsdk-fb09s-4a201ba6ef.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'railworldproject-49942.appspot.com'
})


from django.shortcuts import render


def images(request):
    return render(request, "images.html")

