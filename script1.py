# import cv2

# pth=r'Pic\Pic\Staff\staff-1.jpg'
# image=cv2.imread(pth)
# print(image.shape)


import face_recognition
import pickle
import numpy as np
from PIL import Image

# Your folder dictionary
people_images = {
    # these are the examples 
    # "Bhagat Singh": [r"Pic\Pic\Staff\staff-1.jpg"],
    # "Prashant Deshmukh": [r"Pic\Pic\Staff\staff-2.jpg"],
    # "Nilanchal Jena": [r"Pic\Pic\Staff\staff-3.jpg"],
    # "Pratamesh Parab": [r"Pic\Pic\Staff\staff-4.jpg"],
    # "Rajkumar Patel" : [r"Pic\Pic\Worker\worker-1.jpg"],
    # "Padanabhao Saho" : [r"Pic\Pic\Worker\worker-2.jpg"],
    # "Anil Palve" : [r"Pic\Pic\Worker\worker-3.jpg"],
    # "Pankaj Kumar" : [r"Pic\Pic\Worker\worker-4.jpg"],
    # "Ramagya Sahani" : [r"Pic\Pic\Worker\worker-5.jpg"],
    # "Gourav " : [r'pic\gourav bhai.jpeg'],
    # "Priyanshu" : [r'pic\priyanshu.jpeg'],
    # "Animesh" : [r'Pic\Pic\Staff\animesh.jpeg'],
    # "Shibansh" : [r'pic\sibansh.jpeg'],
    # "Asutosh" : [r'pic\asutosh1.jpeg',r'pic\asutosh2.jpeg',r'pic\asutosh3.jpeg'],
    # "Prototype Worker" : [r'pic\worker1.jpeg'],
    # "Worker-0":[r'pic\Screenshot (140).png'],
    # "Satish":[r'pic\Screenshot (147).png'],
    # "Sachin":[r'pic\Screenshot (142).png',r'pic\Screenshot (143).png',r'pic\Screenshot (145).png'],
    # "Worker-3":[r'pic\Screenshot (143).png'],
    # "Sarabjit":[r'pic\Screenshot (140).png',r'pic\Screenshot (144).png',r'pic\Screenshot (146).png'],
    # "Worker-5":[r'pic\Screenshot (145).png'],
    # "Worker-6":[r'pic\Screenshot (146).png'],
    # "Mama" : [r'pic\mama.jpeg'],

    "Abhisekh" : [r'pic\Abhisekh1.jpeg',r'pic\Abhisekh2.jpeg',r'pic\Abhisekh3.jpeg'],
    "Anita Devi":[r'pic\Anita1.jpeg',r'pic\Anita2.jpeg',r'pic\Anita3.jpeg'],
    "Dharampal" : [r'pic\Purple Staff1.jpeg',r'pic\Purple Staff2.jpeg',r'pic\Purple Staff3.jpeg',r'pic\Dharmpal1.png'],
    "Sachin Kumar" : [r'pic\Sachin1.jpeg',r'pic\Sachin2.jpeg',r'pic\Sachin3.jpeg',r'pic\Sachin4.png'],
    "Sunil Kumar Meena" : [r'pic\Sunil1.jpeg',r'pic\Sunil2.jpeg',r'pic\Sunil3.jpeg',r'pic\Sunil4.jpeg',r'pic\Sunil5.png',r'pic\Sunil6.png'],
    "Sourav" : [r'pic\Sourav.jpeg'],
}

known_encodings = []
known_names = []

print("Starting encoding process...")

for name, paths in people_images.items():
    for path in paths:
        try:
            # Load and fix the image (8-bit RGB + Resize for massive images)
            pil_img = Image.open(path).convert('RGB')
            pil_img.thumbnail((1200, 1200)) # Resizes 4K images to manageable size
            
            # image_array = np.array(pil_img)
            image_array = np.array(pil_img, dtype='uint8')
            image_array = np.ascontiguousarray(image_array)
            # Find face locations and encodings
            encodings = face_recognition.face_encodings(image_array)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"Success: Encoded {name} from {path}")
            else:
                print(f"Warning: No face found in {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

# Save the data
data = {"encodings": known_encodings, "names": known_names}
with open("encoding.pickle", "wb") as f:
    pickle.dump(data, f)

print("\nFinished! 'encoding.pickle' has been created.")
