from PIL import Image
import glob

image_list = []
resized_images = []

for filename in glob.glob('C:/Users/PRAMILA/.spyder-py3/mini/combined_text/*.jpeg'):
    print(filename)
    img = Image.open(filename)
    image_list.append(img)

for image in image_list:
    image = image.resize((1100,500))
    resized_images.append(image)
i=0
for (i, new) in enumerate(resized_images):
    new.save('{}{}{}'.format('C:/Users/PRAMILA/.spyder-py3/mini/data/streamlit_image_cache1/', i+1, '.jpeg'))
    
import os
import moviepy.video.io.ImageSequenceClip
image_folder='C:/Users/PRAMILA/.spyder-py3/mini/data/streamlit_image_cache1'
fps=1

image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(r"C:/Users/PRAMILA/.spyder-py3/mini/myvideo.mp4")
