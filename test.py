from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

# تحميل النموذج 
model = load_model('best_xray_model.keras')

# مسارات الصور
image_paths = [
    'C:/Users/DELL/Desktop/data/new_images/sample_image1.jpg',  
    'C:/Users/DELL/Desktop/data/new_images/sample_image2.jpg'  
]

class_labels = ['infected', 'normal']  

# مسار المجلد لحفظ الصور
output_folder = 'C:/Users/DELL/Desktop/data/images_test'

for i, image_path in enumerate(image_paths):
    img = image.load_img(image_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 

    prediction = model.predict(img_array)
    predicted_class = class_labels[int(prediction[0] > 0.5)]  

    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')

    # حفظ الصورة في المجلد
    output_path = os.path.join(output_folder, f"output_image_{i+1}_{predicted_class}.png")
    plt.savefig(output_path)

    print(f"The predicted class for {image_path} is: {predicted_class}")
    print(f"Image saved at: {output_path}")
    plt.close()  
