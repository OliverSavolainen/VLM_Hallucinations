import jsonlines
import json
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO

def scale_bbox(bbox, new_scale,model_bbox_scale):
    corners = bbox.split(",")
    float_corners = [float(corner) / model_bbox_scale for corner in corners]
    scaled_x0, scaled_x1 = (int(float_corners[0] * new_scale[0]),
                            int(float_corners[2] * new_scale[0]))
    scaled_y0, scaled_y1 = (int(float_corners[1] * new_scale[1]),
                            int(float_corners[3] * new_scale[1]))
    return scaled_x0, scaled_y0, scaled_x1, scaled_y1

def draw_boxes(image, bbox):
    draw = ImageDraw.Draw(image)

    x0, y0, x1, y1 = scale_bbox(bbox, image.size, model_bbox_scale=1000.0)

    bbox_corners = [x0, y0, x1, y1]




    draw.rectangle(bbox_corners, outline="blue", width=2)
    #draw.text((0,0), f"{text}",font=ImageFont.truetype("arial.ttf", size=40), fill="blue")

    return image


import tkinter as tk
from PIL import Image, ImageTk
import requests
from io import BytesIO



class ImageLabeler:
    def __init__(self, root, data, image_urls, start_index):
        self.root = root
        self.root.title("Image Labeler")

        self.model_answers = data
        self.image_urls = image_urls
        self.current_answer_index = start_index

        # Load the first image
        self.load_model_pred()

        # Add Yes, No, Previous, and Next buttons
        self.hallucination_button = tk.Button(root, text="Hallucination", command=lambda: self.label_answer("Hallucination"))
        self.hallucination_button.pack(side="left", padx=10, pady=10)

        self.misclassification_button = tk.Button(root, text="Misclassification", command=lambda: self.label_answer("Misclassification"))
        self.misclassification_button.pack(side="left", padx=10, pady=10)

        self.correct_button = tk.Button(root, text="Correct",
                                                  command=lambda: self.label_answer("Correct"))
        self.correct_button.pack(side="left", padx=10, pady=10)


        self.prev_button = tk.Button(root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left", padx=10, pady=10)

        self.next_button = tk.Button(root, text="Next", command=self.next_image)
        self.next_button.pack(side="left", padx=10, pady=10)



    def load_model_pred(self):
        obj = self.model_answers[self.current_answer_index]
        image_id = obj["question_id"]
        image_url = self.image_urls.get(image_id)

        print(image_id)


        if not image_url:
            print(f"Image URL for {image_id} not found.")


        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Failed to download image {image_id}")

        model_answer = ""
        if obj["is_no_answer"]:
            model_answer = "No"
        else:
            model_answer = "Yes"
        object_name = obj["object_name"]
        prompt = obj["prompt"].split("\n")[0]

        if "ground_truth" in obj.keys():
            ground_truth = obj["ground_truth"]
        else:
            ground_truth = ""

        text = f"Question: {prompt} \n Object name: {object_name} \n Model answer: {model_answer} \n Label: {ground_truth}"

        print(obj["bounding_box"])
        if obj["bounding_box"] == "":
            if model_answer == "Yes":
                self.next_image()
            self.image = Image.open(BytesIO(response.content))

        else:
            image = Image.open(BytesIO(response.content))
            self.image = draw_boxes(image, obj["bounding_box"])


        self.photo = ImageTk.PhotoImage(self.image)



        # Display the image
        if hasattr(self, 'label'):
            self.label.config(image=self.photo)
        else:
            self.label = tk.Label(self.root, image=self.photo)
            self.label.pack()



        # Display the text
        if hasattr(self, 'text_label'):
            self.text_label.config(text=text)
        else:
            self.text_label = tk.Label(self.root, text=text, wraplength=400)
            self.text_label.pack()

        self.root.title(f"Image Labeler - {image_url}")

    def label_answer(self, label):
        self.model_answers[self.current_answer_index]["ground_truth"] = label
        print(f"Model pred {self.current_answer_index} labeled as: {label}")


        # Automatically move to the next image
        self.next_image()

    def next_image(self):
        if self.current_answer_index < len(self.model_answers) - 1:
            self.current_answer_index += 1
            self.load_model_pred()
        else:
            print("Reached the last image.")

    def prev_image(self):
        if self.current_answer_index > 0:
            self.current_answer_index -= 1
            self.load_model_pred()
        else:
            print("This is the first image.")

    def on_close(self):
        
        self.root.destroy()
        return data


if __name__ == "__main__":
    root = tk.Tk()


    start_index = 0

    saved_labels_file_name = "labeled_grounded_pope_answers_0_999.jsonl"

    # Replace with your list of image URLs
    data = []
    with jsonlines.open(saved_labels_file_name) as reader:
        for obj in reader:
            data.append(obj)
    
    # with jsonlines.open('intermediate_outputs/pope_objects_with_bboxes.jsonl') as reader:
    #     for obj in reader:
    #         data.append(obj)


    with open('data/bbox_pope_images/labels.json') as f:
        labels = json.load(f)

    image_urls = {img['file_name']: img['coco_url'] for img in labels['images']}

    app = ImageLabeler(root, data, image_urls, start_index)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Ensure the file is closed when the window is closed
    root.mainloop()

    labeled_data = app.model_answers



    with open(saved_labels_file_name, 'w') as output_file:
        for obj in labeled_data:
            json_line = json.dumps(obj)
            output_file.write(json_line + '\n')
