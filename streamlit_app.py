

import streamlit as st
from pathlib import Path
import glob
import start_api
from interface import combine_images, encoding_sentences, preprocess_text
import nltk
import base64
from PIL import Image
import os
import torch
                    
import moviepy.video.io.ImageSequenceClip                     
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app_name = "Deep Wrap-UP"

# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_name, initial_sidebar_state="expanded",page_icon="📚"
)

st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt;font-weight: bold;'>DEEP WRAP-UP 📚</h1>", unsafe_allow_html=True)
st.markdown("<h1 style=' color:#BB1D3F; text_align:center;font-weight: bold;font-size:18pt;'>Made by Quad Techies with ❤️</h1>", unsafe_allow_html=True)
st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>SUMMARY GENERATOR</h1>", unsafe_allow_html=True)  
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request  


import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



# Summarized output from above ::::::::::
# the us has over 637,000 confirmed Covid-19 cases and over 30,826 deaths. 
# president Donald Trump predicts some states will reopen the country in april, he said. 
# "we'll be the comeback kids, all of us," the president says.
#fetching the content from the URL

import pdfplumber

from PyPDF2 import PdfFileReader
def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text

def _create_dictionary_table(text_string) -> dict:
   
    #removing stop words
    stop_words = set(stopwords.words("english"))
    #reducing words to their root form
    stem = PorterStemmer()    
    words = word_tokenize(text_string)
    
	    
    #creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table


def _calculate_sentence_scores(sentences, frequency_table) -> dict:   

    #algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]

        sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words

       

    return sentence_weight

def _calculate_average_score(sentence_weight) -> int:
   
    #calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    #getting sentence average value from source text
    average_score = (sum_values / len(sentence_weight))

    return average_score

def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary

def _run_article_summary(article,sentence_length):
    
    #creating a dictionary for the word frequency table
    frequency_table = _create_dictionary_table(article)
    #tokenizing the sentences
    sentences = sent_tokenize(article)
    #algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)
    #getting the threshold
    threshold = _calculate_average_score(sentence_scores)	   
    #producing the summary
    article_summary = _get_article_summary(sentences, sentence_scores, sentence_length*threshold )	
    return article_summary


def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()           
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href



if __name__ == '__main__':
       st.sidebar.markdown("<h1 style='text-align: center; color: black;'>🧭 Navigation Bar 🧭</h1>", unsafe_allow_html=True)
       nav = st.sidebar.radio("",["URL 🏡","TEXT","FILE"])
       if nav == "URL 🏡":
            article_read = st.text_input("Enter any URL:")
            try:
                article_read = urllib.request.urlopen(article_read)
                article_read = article_read.read()
                summary_length=st.slider('Choose the threshold length of the summary [higher the threshold length,lower the summary!]', min_value=1.0, step=0.1, max_value=5.0,value=1.0)
                st.info ('URL is good!')
                if(st.button("summarize")):
                    #parsing the URL content and storing in a variable
                    article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')
                    
                    # #returning <p> tags
                    paragraphs = article_parsed.find_all('p')
                    print("fetched data:",paragraphs)
                    
                    article_content = ''
                    
                    # #looping through the paragraphs and adding them to the variable
                    for p in paragraphs:  
                         article_content += p.text
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>TEXT</h1>", unsafe_allow_html=True)
                    st.write(article_content)
                    st.write("hello1")			
                    summary_results = _run_article_summary(article_content,summary_length)
                    st.write("hello2")	
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>EXTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(summary_results)
                    
                    model = T5ForConditionalGeneration.from_pretrained('t5-small')
                    tokenizer = T5Tokenizer.from_pretrained('t5-small')
                    device = torch.device('cpu')
                    #text=summary_results
                    # text ="""
                    # Sri Ramakrishna Engineering College (SREC) is an autonomous Engineering college in India founded by Sevaratna Dr. R. Venkatesalu. It is affiliated with the Anna University in Chennai, and approved by the All India Council for Technical Education (AICTE) of New Delhi. It is accredited by the NBA (National Board of Accreditation) for most of its courses and by the Government of Tamil Nadu.
                    # The college was founded in the year 1994 by Philanthropist and Industrialist Sevaratna Dr. R. Venkatesalu. It provides various undergraduate and postgraduate courses in engineering and other technical streams. The college attained its autonomous status in 2007-2008 when Anna University was split into six different universities. SREC is one of many institutions managed by SNR Sons Charitable Trust, founded by Sevaratna Dr. R. Venkatesalu. The college covers a total area of 45 acres.
                    # """
                    
                    
                    preprocess_text = summary_results.strip().replace("\n","")
                    t5_prepared_Text = "summarize: "+preprocess_text
                    print ("original text preprocessed: \n", preprocess_text)
                    
                    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                    
                    
                    # summmarize 
                    summary_ids = model.generate(tokenized_text,
                                                        num_beams=2,
                                                        no_repeat_ngram_size=1,
                                                        min_length=100,
                                                        max_length=1000,
                                                        early_stopping=True)
                    
                    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    print ("\n\nSummarized text: \n",output)
                    #text = ("a young tree, vine, shrub, or herb planted or suitable for planting. b : any of a kingdom (Plantae) of multicellular eukaryotic mostly photosynthetic organisms typically lacking locomotive movement or obvious nervous or sensory organs and possessing cellulose cell")
                    
                
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>ABSTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(output)
            
                    lines=nltk.tokenize.sent_tokenize(summary_results)
                    st.write(lines)
                    results = encoding_sentences(lines)
                    
                
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>SUMMARY WITH IMAGES</h1>", unsafe_allow_html=True)
                    image_path = "combined"
                    text_image_path = "combined_text"
                    
                    #os.mkdir(image_path)
                    j=0
                    for k, row in enumerate(results):
                        line = row["text"]
                        st.markdown(f"### *{line}*")
                        grid = combine_images(row["unsplashIDs"])
                        
                        # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
                        st.image(grid, use_column_width=True)
                        from PIL import Image, ImageFont, ImageDraw
                       # st.write(type(grid))
                        title_font = ImageFont.truetype("PlayfairDisplay-VariableFont_wght.ttf", 100)
                        
                        
                        im = Image.fromarray(grid)
                        im.save(f"{image_path}/your_file{j}.jpeg")
                        my_image = Image.open(f"{image_path}/your_file{j}.jpeg")
                        #st.image(my_image)

                        #title_text = "The Beauty of Nature"
                        image_editable = ImageDraw.Draw(my_image)
                        image_editable.text((10,15), line, (237, 230, 211),font=title_font)
                        my_image.save(f"{text_image_path}/your_file{j}.jpeg")

                        
                        j=j+1

                        #image = grid.save(f"{image_path}/")
               
            
                    image_list = []
                    resized_images = []
                    
                    for filename in glob.glob('combined_text/*.jpeg'):
                        print(filename)
                        img = Image.open(filename)
                        image_list.append(img)
                    
                    for image in image_list:
                        image = image.resize((1100,500))
                        resized_images.append(image)
                    i=0
                    for (i, new) in enumerate(resized_images):
                        new.save('{}{}{}'.format('data/streamlit_image_cache1/', i+1, '.jpeg'))
                        

                    image_folder='data/streamlit_image_cache1'
                    fps=1
                    
                    image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
                    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                    clip.write_videofile("myvideo.mp4")
                    st.markdown(get_binary_file_downloader_html('myvideo.mp4', 'video Summary'), unsafe_allow_html=True)  
            except:
                st.info ('Failed to reach the server.')

                
       if nav == "TEXT":
            article_read = st.text_area("Enter your text",max_chars=None)
            summary_length=st.slider('Choose the threshold length of the summary [higher the threshold length,lower the summary!]', min_value=1.0, step=0.1, max_value=5.0,value=1.0)

            try:

                if(st.button("summarize")):
        
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>TEXT</h1>", unsafe_allow_html=True)
                    st.write(article_read)
                    summary_results = _run_article_summary(article_read,summary_length)
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>EXTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(summary_results)
                    
                    model = T5ForConditionalGeneration.from_pretrained('t5-small')
                    tokenizer = T5Tokenizer.from_pretrained('t5-small')
                    device = torch.device('cpu')
                    
                    preprocess_text = summary_results.strip().replace("\n","")
                    t5_prepared_Text = "summarize: "+preprocess_text
                    print ("original text preprocessed: \n", preprocess_text)
                    
                    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                    
                    
                    # summmarize 
                    summary_ids = model.generate(tokenized_text,
                                                        num_beams=4,
                                                        no_repeat_ngram_size=1,
                                                        min_length=300,
                                                        max_length=1000,
                                                        early_stopping=True)
                    
                    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    #print ("\n\nSummarized text: \n",output)
                    #text = ("a young tree, vine, shrub, or herb planted or suitable for planting. b : any of a kingdom (Plantae) of multicellular eukaryotic mostly photosynthetic organisms typically lacking locomotive movement or obvious nervous or sensory organs and possessing cellulose cell")
                    
                
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>ABSTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(output)
            
                    lines=nltk.tokenize.sent_tokenize(summary_results)
                    st.write(lines)
                    results = encoding_sentences(lines)
                    
                
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>SUMMARY WITH IMAGES</h1>", unsafe_allow_html=True)
                
                
                    image_path = "combined"
                    text_image_path = "combined_text"
                    
                    #os.mkdir(image_path)
                    j=0
                    for k, row in enumerate(results):
                        line = row["text"]
                        st.markdown(f"### *{line}*")
                        grid = combine_images(row["unsplashIDs"])
                        
                        # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
                        st.image(grid, use_column_width=True)
                        from PIL import Image, ImageFont, ImageDraw
                       # st.write(type(grid))
                        title_font = ImageFont.truetype("PlayfairDisplay-VariableFont_wght.ttf", 100)
                        
                        
                        im = Image.fromarray(grid)
                        im.save(f"{image_path}/your_file{j}.jpeg")
                        my_image = Image.open(f"{image_path}/your_file{j}.jpeg")
                        #st.image(my_image)

                        #title_text = "The Beauty of Nature"
                        image_editable = ImageDraw.Draw(my_image)
                        image_editable.text((10,15), line, (237, 230, 211),font=title_font)
                        my_image.save(f"{text_image_path}/your_file{j}.jpeg")

                        
                        j=j+1

                        #image = grid.save(f"{image_path}/")
               
            
                    image_list = []
                    resized_images = []
                    
                    for filename in glob.glob('combined_text/*.jpeg'):
                        print(filename)
                        img = Image.open(filename)
                        image_list.append(img)
                    
                    for image in image_list:
                        image = image.resize((1100,500))
                        resized_images.append(image)
                    i=0
                    for (i, new) in enumerate(resized_images):
                        new.save('{}{}{}'.format('data/streamlit_image_cache1/', i+1, '.jpeg'))
                        
                    import os
                    import moviepy.video.io.ImageSequenceClip
                    image_folder='data/streamlit_image_cache1'
                    fps=1
                    
                    image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
                    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                    clip.write_videofile("myvideo.mp4")
                    st.markdown(get_binary_file_downloader_html('myvideo.mp4', 'Video Summary'), unsafe_allow_html=True)  
                                
                         
            
            except:
                st.info ('Failed to reach the server.')   
                
       if nav == "FILE":
            docx_file = st.file_uploader(label="Upload your file",type=['pdf', 'docx'])
            if(docx_file):
                if docx_file.type == "application/pdf":
                    raw_text = read_pdf(docx_file)
                    #st.write(raw_text)
                    try:
                        with pdfplumber.open(docx_file) as pdf:
                            page = pdf.pages[0]
                            #st.write(page.extract_text())
                            article_read=page.extract_text()
                            st.success(article_read)
                    except:
                        st.write("None")	
                summary_length=st.slider('Choose the threshold length of the summary [higher the threshold length,lower the summary!]', min_value=1.0, step=0.1, max_value=5.0,value=1.0)
    
                try:
    
                    if(st.button("summarize")):
            
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>TEXT</h1>", unsafe_allow_html=True)
                        st.write(article_read)
                        summary_results = _run_article_summary(article_read,summary_length)
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>EXTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                        st.write(summary_results)
                        
                        model = T5ForConditionalGeneration.from_pretrained('t5-small')
                        tokenizer = T5Tokenizer.from_pretrained('t5-small')
                        device = torch.device('cpu')
                        
                        preprocess_text = summary_results.strip().replace("\n","")
                        t5_prepared_Text = "summarize: "+preprocess_text
                        print ("original text preprocessed: \n", preprocess_text)
                        
                        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                        
                        
                        # summmarize 
                        summary_ids = model.generate(tokenized_text,
                                                            num_beams=4,
                                                            no_repeat_ngram_size=1,
                                                            min_length=300,
                                                            max_length=1000,
                                                            early_stopping=True)
                        
                        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        
                        #print ("\n\nSummarized text: \n",output)
                        #text = ("a young tree, vine, shrub, or herb planted or suitable for planting. b : any of a kingdom (Plantae) of multicellular eukaryotic mostly photosynthetic organisms typically lacking locomotive movement or obvious nervous or sensory organs and possessing cellulose cell")
                        
                    
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>ABSTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                        st.write(output)
                
                        lines=nltk.tokenize.sent_tokenize(summary_results)
                        st.write(lines)
                        results = encoding_sentences(lines)
                        
                    
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>SUMMARY WITH IMAGES</h1>", unsafe_allow_html=True)

                        image_path = "combined"
                        text_image_path = "combined_text"
                        
                        #os.mkdir(image_path)
                        j=0
                        for k, row in enumerate(results):
                            line = row["text"]
                            st.markdown(f"### *{line}*")
                            grid = combine_images(row["unsplashIDs"])
                            
                            # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
                            st.image(grid, use_column_width=True)
                            from PIL import Image, ImageFont, ImageDraw
                           # st.write(type(grid))
                            title_font = ImageFont.truetype("PlayfairDisplay-VariableFont_wght.ttf", 100)
                            
                            
                            im = Image.fromarray(grid)
                            im.save(f"{image_path}/your_file{j}.jpeg")
                            my_image = Image.open(f"{image_path}/your_file{j}.jpeg")
                            #st.image(my_image)

                            #title_text = "The Beauty of Nature"
                            image_editable = ImageDraw.Draw(my_image)
                            image_editable.text((10,15), line, (237, 230, 211),font=title_font)
                            my_image.save(f"{text_image_path}/your_file{j}.jpeg")

                            
                            j=j+1

                            #image = grid.save(f"{image_path}/")
                   
                
                        image_list = []
                        resized_images = []
                        
                        for filename in glob.glob('combined_text/*.jpeg'):
                            print(filename)
                            img = Image.open(filename)
                            image_list.append(img)
                        
                        for image in image_list:
                            image = image.resize((1100,500))
                            resized_images.append(image)
                        i=0
                        for (i, new) in enumerate(resized_images):
                            new.save('{}{}{}'.format('data/streamlit_image_cache1/', i+1, '.jpeg'))
                            
                        import os
                        import moviepy.video.io.ImageSequenceClip
                        image_folder='data/streamlit_image_cache1'
                        fps=1
                        
                        image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
                        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                        clip.write_videofile("myvideo.mp4")
                        st.markdown(get_binary_file_downloader_html('myvideo.mp4', 'video Summary'), unsafe_allow_html=True)  
                                    
                        
                
                except:
                    st.info ('Failed to reach the server.')   
