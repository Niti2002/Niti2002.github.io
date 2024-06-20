from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from django.core.files.storage import default_storage
from firebase_admin import storage


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            file_name = default_storage.save(image.name, image)
            file_path = default_storage.path(file_name)
            
            # Upload to Firebase
            bucket = storage.bucket()
            blob = bucket.blob('images/' + image.name)
            blob.upload_from_filename(file_path)

            return redirect('success')
    else:
        form = ImageUploadForm()
    return render(request, 'uplod.html', {'form': form})

def success(request):
    return render(request, 'success.html')


