from PIL import Image
import matplotlib.pyplot as plt







##########################
#Reframing
###########################
from reframe import get_possible_subjects, refram_to_thirds

#importing image temporarily will later take input from earlier in pipline
#Image path 
img_path = "Test_data\86b80553-5091-48f5-ab74-852ba5ea1caa.webp"
image = Image.open(img_path)

#Get possible subjects
possible_subjects = get_possible_subjects(image)

print("List of possible subjects:")
for i, subject in enumerate(possible_subjects):
    print(f"{i+1}. {subject}")
print("Write the number or name of the subject you want to reframe")
subject = None

#getting subject from user
while subject is None:
    subject_suggestion = input("Subject: ")
    #check if the input is a number
    if subject_suggestion.isdigit():
        subject_suggestion = int(subject_suggestion)
        if subject_suggestion <= len(possible_subjects):
            subject = possible_subjects[subject_suggestion - 1]
        else:
            print("Invalid input. Please try again.")
    else:
        if subject_suggestion in possible_subjects:
            subject = subject_suggestion
        else:
            print("Invalid input. Please try again.")

allow_zoom = None
while allow_zoom is None:
    zoom_suggestion = input("Do you want to allow zooming in the reframed image? (y/n): ")
    if zoom_suggestion.lower() == "y":
        allow_zoom = True
    elif zoom_suggestion.lower() == "n":
        allow_zoom = False
    else:
        print("Invalid input. Please try again.")


image, mask = refram_to_thirds(image, Subject = subject, Return_mask = True, show_focal_points=False, allow_zoom = allow_zoom)

#plot new image
plt.imshow(image)
plt.show()