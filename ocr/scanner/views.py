
import cv2
import pytesseract
from django.shortcuts import render
from .forms import ImageUploadForm
import openai
from openai import OpenAI
client = OpenAI(api_key = "API")
def index(request):
    return render(request, 'base.html')

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded image
            image = form.cleaned_data['image']
            image_path = 'media/' + image.name  # Assuming you want to save it in the 'media' directory
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            # Call Tesseract to extract text from the image
            try:
                text = extract_text_from_image(image_path)
                # Display the uploaded image along with the extracted text
                return render(request, 'result.html', {'image_path': image_path, 'text': text})
            except Exception as e:
                # Handle errors gracefully
                error_message = f"An error occurred: {str(e)}"
                return render(request, 'upload_image.html', {'form': form, 'error_message': error_message})
    else:
        form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form})

def read_image(path):
    im = cv2.imread(path)
    return im

def parse_image(image):
    custom_config = r'--oem 3 --psm 6'
    s = pytesseract.image_to_string(image, config=custom_config)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Take the given text and explain it in simple words"},
        {"role": "user", "content": s}
    ]
    )

    return completion.choices[0].message.content
    # return s

def extract_text_from_image(image_path):
    im = read_image(image_path)
    text = parse_image(im)
    return text
