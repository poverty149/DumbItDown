import pytesseract
import cv2
import numpy as np
# import openai
# from openai import OpenAI

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# client = OpenAI(api_key = "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")
# openai.api_key = "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"
def read_image(path):
    im=cv2.imread(path)
    # cv2.imshow('Original', im) 
    # cv2.waitKey(5000) 
    
    # Use the cvtColor() function to grayscale the image 
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    
    # cv2.imshow('Grayscale', gray_image) 
    # cv2.waitKey(5000)   
    return im
    # Window shown waits for any key pressing event 
    # cv2.destroyAllWindows()
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 
def parse_image(image):
    custom_config = r'--oem 3 --psm 6'
    s=pytesseract.image_to_string(image, config=custom_config)
    return s
def summarize(prompt):
  summarizer_instruction = f"""
    [INST]<>
      You are a summarizer and your job is to summarize whatever user asks you too in a most concise manner while being safe. Your summaries must have the same meaning and not include and false information. Make sure you dont use any external knowledge other than whats provided to you. Your final output must only be the summarize text.
    <>
    {prompt}
    [/INST]
  """
  sequences = pipeline(
      summarizer_instruction,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
  )

  answer = sequences[0]['generated_text'].split("[/INST]")[1].strip()
  return answer
if __name__=="__main__":
    im=read_image("img.png")
    text=parse_image(im)
    ans=summarize(text)
    print(ans)
#     completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)
    # response = openai.chat.completions.create(
    #     model="gpt-3.5-turbo",
    # messages=[
    #     {
    #         "role": "system",
    #         "content": "Summarize content you are provided with for a second-grade student."
    #     },
    #     {
    #         "role": "user",
    #         "content": text,
    #     },
    # ],
    # temperature=0.7,
    # max_tokens=64,
    # top_p=1
    # )

    # summary = response.choices[0].text.strip()
    # print(summary)


    