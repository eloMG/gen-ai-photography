from PIL import Image
import matplotlib.pyplot as plt


##########################
#Reframing
###########################
from reframe import get_possible_subjects, refram_to_thirds, Display_object_detection

#importing image temporarily will later take input from earlier in pipline
#Image path 
img_path = "Test_data/2706.jpg"
image = Image.open(img_path)

#Get possible subjects
possible_subjects = get_possible_subjects(image)

print("List of possible subjects:")
for i, subject in enumerate(possible_subjects):
    print(f"{i+1}. {subject}")
print("Write the number or name of the subject you want to reframe or write show to see the object detection result.")
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
        elif subject_suggestion == "show":
            Display_object_detection(image)
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

#ask user if they wanto to add infill
do_infill = None
while do_infill is None:
    infill_suggestion = input("Do you want to add infill to the reframed image? (y/n): ")
    if infill_suggestion.lower() == "y":
        do_infill = True
    elif infill_suggestion.lower() == "n":
        do_infill = False
    else:
        print("Invalid input. Please try again.")

if do_infill:
    from infill import Infill

    print("Next we add infill to the reframed image.")
    print("please add the following prompts for infill(if not just press enter):")

    prompt = input("Prompt: ")


image = Infill(image, mask, prompt = prompt)


#plot new image
plt.imshow(image)
plt.show()