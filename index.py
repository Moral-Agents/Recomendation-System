# -*- coding: utf-8 -*-
import numpy as np
from operator import itemgetter
import json

import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup
import requests,io
import PIL.Image
from urllib.request import urlopen

with open('movie_data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open('movie_titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)
    
movies = [title[0] for title in movie_titles]
#print(len(movie_titles))
#for i in range(5):
#  print(movie_titles[i])
#  print(data[i],"\n")
  
  
class KNearestNeighbours:
    def __init__(self, data, target, test_point, k):
        self.data = data
        self.target = target
        self.test_point = test_point
        self.k = k
        self.distances = list()
        self.categories = list()
        self.indices = list()
        self.counts = list()
        self.category_assigned = None

    @staticmethod
    def dist(p1, p2):
        """Method returns the euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def fit(self):
        """Method that performs the KNN classification"""
        # Create a list of (distance, index) tuples from the test point to each point in the data
        self.distances.extend([(self.dist(self.test_point, point), i) for point, i in zip(self.data, [i for i in range(len(self.data))])])
        # Sort the distances in ascending order
        sorted_li = sorted(self.distances, key=itemgetter(0))
        # Fetch the indices of the k nearest point from the data
        self.indices.extend([index for (val, index) in sorted_li[:self.k]])
        # Fetch the categories from the train data target
        for i in self.indices:
            self.categories.append(self.target[i])
        # Fetch the count for each category from the K nearest neighbours
        self.counts.extend([(i, self.categories.count(i)) for i in set(self.categories)])
        # Find the highest repeated category among the K nearest neighbours
        self.category_assigned = sorted(self.counts, key=itemgetter(1), reverse=True)[0][0]

def KNN_Movie_Recommender(test_point, k):
    # Create dummy target variable for the KNN Classifier
    target = [0 for item in movie_titles]
    # Instantiate object for the Classifier
    model = KNearestNeighbours(data, target, test_point, k=k)
    # Run the algorithm
    model.fit()
    # Print list of 10 recommendations < Change value of k for a different number >
    table = []
    for i in model.indices:
        # Returns back movie title and imdb link
        table.append([movie_titles[i][0], movie_titles[i][2],data[i][-1]])
    #print(table)
    return table


#select_movie = input("Input: ")
#genres = data[movies.index(select_movie)]

#result = (KNN_Movie_Recommender(genres, 5))
#for i in result:
#    print(i)


def movie_poster_fetcher(imdb_link):
    ## Display Movie Poster
    url_data = requests.get(imdb_link).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_dp = s_data.find("meta", property="og:image")
    movie_poster_link = imdb_dp.attrs['content']
    u = urlopen(movie_poster_link)
    raw_data = u.read()
    image = PIL.Image.open(io.BytesIO(raw_data))
    image = image.resize((158, 301), )
    #st.image(image, use_column_width=False)
    return image

def get_movie_info(imdb_link):
    url_data = requests.get(imdb_link).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_content = s_data.find("meta", property="og:description")
    movie_descr = imdb_content.attrs['content']
    movie_descr = str(movie_descr).split('.')
    movie_director = movie_descr[0]
    movie_cast = str(movie_descr[1]).replace('With', 'Cast: ').strip()
    movie_story = 'Story: ' + str(movie_descr[2]).strip()+'.'

    return movie_director,movie_cast,movie_story

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid
#----------------------------------------

st.set_page_config(
    page_title = "Movie Recommender App"
)

def run():  
    st.markdown("<h1 style='text-align: center; color: white;'>Movie Recommender App</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    st.markdown("<h4 style=' position: absolute; top: -65px; right: 650px; width: 200px; height: 100px; color: white;'>Base Movie</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style=' position: absolute; top: -100px; right: -70px; width: 200px; color: white;'>N° Recommendations</h5>", unsafe_allow_html=True)
    st.markdown("<h6 style=' position: absolute; top: -40px; right: 300px; width: 400px; height: 100px; color: white;'>*Recommendation will take this movie as a base</h6>", unsafe_allow_html=True)
    with col1:
     select_movie = st.selectbox('', ['--Select--'] + movies)
       
    
    with col2:
     no_of_reco = st.selectbox('', range(6,25))
    
    
    if select_movie != '--Select--':
        mygrid = make_grid(no_of_reco//3+1,3)
        genres = data[movies.index(select_movie)]
        test_points = genres
        table = KNN_Movie_Recommender(test_points, no_of_reco+1)
        table.pop(0)
        for c in range (len(table)):
            movie, link, ratings = table[c]
            i = c//3
            j = c%3
            with mygrid[i][j]:
                 st.image(movie_poster_fetcher(link), use_column_width=True)
                 st.markdown(f"{c+1}. {movie}")
                 director,cast,story = get_movie_info(link)
                 if(":" in director):
                    st.markdown(director[director.index(":")+2:])
                 else:
                    st.markdown(director)
                 st.markdown(f'<div style="text-align: justify;">{cast}</div>', unsafe_allow_html=True)
                 st.markdown(f'<div style="text-align: justify;">{story}</div>', unsafe_allow_html=True)
                 st.markdown('IMDB Rating: ' + str(ratings) + '/10 ⭐')
 
run()                