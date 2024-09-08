from functools import lru_cache
import numpy as np
import pandas as pd


import requests

import zipfile
import warnings
import requests
import zipfile
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

from PIL import Image

global df_movies, df_ratings, df_final, df_links

@lru_cache(maxsize=1000)  # Cachea hasta 1000 llamadas para acelerar las repeticiones
def get_movie_image_url(tmdb_id, api_key='64b35ee705c55e7ccdd7c754a9b8cdc4'):
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return None

df_movies = pd.read_csv('data/ml-latest-small/movies.csv')
df_ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
df_links = pd.read_csv('data/ml-latest-small/links.csv')




@st.cache_data
def preprocesamiento():
    global df_movies, df_ratings, df_final, df_links
    # Eliminar registros con valores nulos en df_ratings y df_movies
    df_ratings = df_ratings.dropna()
    df_movies = df_movies.dropna()
    df_links = df_links.dropna()
    # Eliminar registros duplicados en df_movies usando la columna movieId como llave
    df_movies = df_movies.drop_duplicates(subset='movieId', keep='first').reset_index(drop=True)
    # Eliminar registros duplicados en df_ratings usando las columnas movieId y userId como llaves
    df_ratings = df_ratings.drop_duplicates(subset=['movieId', 'userId'], keep='first').reset_index(drop=True)  
    # Eliminar registros duplicados en df_links usando la columna movieId como llave
    df_links = df_links.drop_duplicates(subset='movieId', keep='first').reset_index(drop=True)
    # Crear la columna 'content' reemplazando '|' por ' ' en la columna 'genres' de df_movies
    df_movies['content'] = df_movies['genres'].str.replace('|', ' ')
    # Crear la columna 'genre_set' como un set de géneros separados por '|'
    df_movies['genre_set'] = df_movies['genres'].apply(lambda x: set(x.split('|')))
    # Convertir la columna 'timestamp' a tipo datetime en df_ratings
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')
    # Fusionar los DataFrames df_movies, df_ratings, y df_links usando la columna 'movieId'
    df_final = pd.merge(df_ratings, df_movies, on='movieId', how='inner')
    df_final = pd.merge(df_final, df_links, on='movieId', how='inner')
    # Crear una columna 'year' basada en el año de 'timestamp' en df_ratings
    df_final['year'] = df_final['timestamp'].dt.year
    # Eliminar la columna 'timestamp' si no es necesaria
    df_final.drop(columns=['timestamp'], inplace=True)
    return df_final

# Llamar a la función de preprocesamiento para crear df_final
df_final = preprocesamiento()
preprocesamiento()





# Crear la matriz de calificaciones con usuarios como filas y películas como columnas
ratings_matrix = df_final.pivot(index='userId', columns='movieId', values='rating')
# Calcular el promedio de calificaciones por usuario
avg_ratings = ratings_matrix.mean(axis=1)
# Normalizar la matriz de calificaciones restando el promedio de cada usuario
ratings_matrix_normalized = ratings_matrix.sub(avg_ratings, axis=0).fillna(0)

# Entrenar el modelo KNN usando la similitud del coseno
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(ratings_matrix_normalized.values)

@st.cache_data
def recomendacion_knn(usuario, ratings_matrix_normalized, ratings_matrix, df_movies, _knn_model, n_recommendations=10):
    # Verifica que el DataFrame ratings_matrix_normalized tiene datos
    if ratings_matrix_normalized.empty:
        raise ValueError("ratings_matrix_normalized está vacío")
    if isinstance(usuario, int):
        # Si el input es un ID de usuario existente, encontrar su índice en ratings_matrix
        user_idx = ratings_matrix.index.get_loc(usuario)
        distances, indices = knn_model.kneighbors(ratings_matrix_normalized.iloc[user_idx].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
    else:
        # Si el input no es un ID, tratarlo como un nuevo usuario
        new_user_ratings = pd.Series(usuario, index=ratings_matrix.columns).fillna(0)
        new_user_normalized = new_user_ratings - new_user_ratings.mean()
        distances, indices = knn_model.kneighbors(new_user_normalized.values.reshape(1, -1), n_neighbors=n_recommendations + 1)

    # Ajustar las distancias y los índices para ignorar el propio usuario
    distances = distances.flatten()[1:]
    indices = indices.flatten()[1:]

    # Obtener las calificaciones de los usuarios similares
    similar_users = ratings_matrix_normalized.iloc[indices]

    # Calcular las calificaciones promedio ponderadas para las películas
    mean_ratings = similar_users.T.dot(1 - distances) / (1 - distances).sum()
    mean_ratings_df = pd.DataFrame(mean_ratings, index=ratings_matrix.columns, columns=['mean_rating'])
    mean_ratings_df = mean_ratings_df.dropna()

    # Filtrar películas que el usuario ya ha visto
    user_seen_movies = ratings_matrix.loc[usuario].dropna().index
    recommendations = mean_ratings_df[~mean_ratings_df.index.isin(user_seen_movies)]

    # Ordenar las recomendaciones por la calificación promedio en orden descendente
    recommendations = recommendations.sort_values(by='mean_rating', ascending=False)

    # Seleccionar las mejores recomendaciones
    recommendations = recommendations.head(n_recommendations)

    # Añadir información de las películas a las recomendaciones
    movie_info = df_movies[['movieId', 'title', 'genres']].drop_duplicates().set_index('movieId')
    recommendations = recommendations.join(movie_info).reset_index()
    
    # Añadir la columna con la URL de la imagen usando el movieId de df_links
    recommendations['image_url'] = recommendations['movieId'].map(
        lambda x: get_movie_image_url(df_links[df_links['movieId'] == x]['tmdbId'].values[0] if not df_links[df_links['movieId'] == x].empty else None)
    )

    return recommendations



# Función para calcular la similitud de BAYESIANO
@st.cache_data
def recomendacion_populares(genero=None):
    global df_final, df_movies, df_links, df_ratings

    # Filtrar las películas por el género seleccionado si se especifica
    if genero:
        df_final_filtrado = df_final[df_final['genres'].str.contains(genero, case=False)]
    else:
        df_final_filtrado = df_final

    # Calcular cuántas veces cada película ha sido votada
    votos_por_pelicula = df_final_filtrado['title'].value_counts()

    # Calcular el número de votos por cada película y su rating promedio
    vote_count = df_final_filtrado.groupby('title')['rating'].size()
    mean_rating = df_final_filtrado.groupby('title')['rating'].mean()

    # Calcular el promedio general de calificaciones de todas las películas
    C = df_final_filtrado['rating'].mean()

    # Definir un valor mínimo de votos para que una película sea considerada
    m = votos_por_pelicula.quantile(0.90)  # Usamos el percentil 90 como umbral mínimo

    # Filtrar películas que tienen más de m votos
    df_popular_movies = df_final_filtrado[df_final_filtrado['title'].isin(vote_count[vote_count >= m].index)]

    # Recalcular el número de votos y el rating promedio de las películas filtradas
    vote_count = df_popular_movies.groupby('title')['rating'].size()
    mean_rating = df_popular_movies.groupby('title')['rating'].mean()

    # Calcular el promedio bayesiano para cada película
    weighted_score = (vote_count / (vote_count + m) * mean_rating) + (m / (vote_count + m) * C)

    # Crear un nuevo DataFrame con title, mean_rating, vote_count y weighted_score
    df_movie_stats = pd.DataFrame({
        'title': mean_rating.index,
        'mean_rating': mean_rating.values,
        'vote_count': vote_count.values,
        'weighted_score': weighted_score.values
    }).reset_index(drop=True)

    # Añadir la columna movieId al DataFrame con base en el df_final
    df_movie_stats['movieId'] = df_final_filtrado.groupby('title')['movieId'].first().reindex(mean_rating.index).values

    # Filtrar las 10 películas con el promedio bayesiano más alto basado en weighted_score
    top_10_populares = df_movie_stats.nlargest(10, 'weighted_score')

    # Ordenar las 10 películas por weighted_score
    top_10_ordenadas_por_score = top_10_populares.sort_values(by='weighted_score', ascending=False)

    # Seleccionar las columnas deseadas incluyendo vote_count
    result = top_10_ordenadas_por_score[['movieId', 'title', 'mean_rating', 'vote_count', 'weighted_score']].reset_index(drop=True)

    # Añadir la columna con la URL de la imagen usando el movieId de df_links
    result['image_url'] = result['movieId'].map(lambda x: get_movie_image_url(df_links[df_links['movieId'] == x]['tmdbId'].values[0]))
    # Retornar el DataFrame con los resultados
    return result

# Función para calcular la similitud de Jaccard
@st.cache_data
def similitud_jaccard(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)
# Función de recomendación basada en usuario
@st.cache_data
def recomendacion_jaccard_por_usuario(ratings, df_items, df_links, n_recommendations=5):
    """
    Recomendación de ítems basada en la similitud de Jaccard según las calificaciones del usuario.
    """
    # Filtrar las películas que el usuario ha calificado con 4 o más estrellas
    rated_movie_ids = [movie_id for movie_id, rating in ratings.items() if rating >= 4]
    # Asegurarse de que hay suficientes calificaciones altas para hacer recomendaciones
    if not rated_movie_ids:
        return None
    # Verifica y crea la columna 'genre_set' si no existe
    if 'genre_set' not in df_items.columns:
        if 'genres' in df_items.columns:
            df_items['genre_set'] = df_items['genres'].apply(lambda x: set(x.split('|')))
        else:
            raise KeyError("La columna 'genres' no existe en df_items")    
    # Obtener los géneros de las películas calificadas
    rated_genres = df_items[df_items['movieId'].isin(rated_movie_ids)]['genre_set']
    # Unir los géneros para crear un perfil del usuario
    user_profile = set().union(*rated_genres)
    # Calcular la similitud de Jaccard entre el perfil del usuario y todas las películas
    df_items['similaridad'] = df_items['genre_set'].apply(lambda x: similitud_jaccard(user_profile, set(x)))
    # Excluir las películas ya calificadas por el usuario
    df_items = df_items[~df_items['movieId'].isin(rated_movie_ids)].sort_values('similaridad', ascending=False).reset_index(drop=True)
    # Obtener las mejores recomendaciones
    result = df_items.head(n_recommendations)
    # Añadir la columna con la URL de la imagen usando el movieId de df_links
    result['image_url'] = result['movieId'].map(lambda x: get_movie_image_url(df_links[df_links['movieId'] == x]['tmdbId'].values[0]))    
    return result


# Función de recomendación basada en TF
@st.cache_data
def recomendacion_tf_idf_por_titulo(titulo_pelicula, n_recommendations=5):
    # Lista de palabras comunes en español para excluir del análisis
    spanish_stop_words = [
        'de', 'la', 'que', 'el', 'en', 'un', 'con', 'y', 'a', 'los', 'se', 
        'no', 'me', 'si', 'por', 'su', 'para', 'esta', 'tener', 'hacer', 
        'poder', 'decir', 'querer', 'poder', 'ser', 'tener', 'hacer'
    ]
    # Verificar que la columna 'content' exista en df_movies
    if 'content' not in df_movies.columns:
        print("Error: La columna 'content' no está en el DataFrame.")
        return None
    # Crear la matriz TF-IDF a partir de los contenidos de las películas
    tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['content'])

    # Calcular la matriz de similitud de coseno basada en la matriz TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Obtener el índice de la película dada por su título
    try:
        idx = df_movies[df_movies['title'].str.lower() == titulo_pelicula.lower()].index[0]
    except IndexError:
        print(f"Error: La película con título '{titulo_pelicula}' no se encontró.")
        return None

    # Obtener las puntuaciones de similitud de coseno para todas las películas
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar las películas por las puntuaciones de similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filtrar la película de entrada para asegurarse de que no se recomiende a sí misma
    sim_scores = [(i, score) for i, score in sim_scores if i != idx]

    # Obtener los índices de las n_recommendations películas más similares
    top_n = sim_scores[:n_recommendations]

    # Obtener los índices y distancias de las películas recomendadas
    recommended_movies = [
        (df_movies.iloc[i]['movieId'], df_movies.iloc[i]['title'], df_movies.iloc[i]['genres'], score) 
        for i, score in top_n
    ]

    # Convertir la lista de recomendaciones en un DataFrame
    recommended_movies_df = pd.DataFrame(recommended_movies, columns=['movieId', 'title', 'genres', 'similarity'])

    # Añadir la columna con la URL de la imagen usando el movieId de df_links
    recommended_movies_df['image_url'] = recommended_movies_df['movieId'].map(
        lambda x: get_movie_image_url(df_links[df_links['movieId'] == x]['tmdbId'].values[0] if not df_links[df_links['movieId'] == x].empty else None)
    )

    # Reducir los decimales de la similitud a dos
    recommended_movies_df['similarity'] = recommended_movies_df['similarity'].round(2)

    return recommended_movies_df





# Inicializar los estados de la sesión 
if 'mostrar_jaccard' not in st.session_state:
    st.session_state.mostrar_jaccard = False

if 'mostrar_bayesiana' not in st.session_state:
    st.session_state.mostrar_bayesiana = False

if 'mostrar_detalle' not in st.session_state:
    st.session_state.mostrar_detalle = False  

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

if 'user_rating' not in st.session_state:
    st.session_state.user_rating = 0
    
# Inicializar el estado de la sesión para controlar la interfaz
if 'mostrar_selector' not in st.session_state:
    st.session_state.mostrar_selector = False

if 'mostrar_tf_idf' not in st.session_state:
    st.session_state.mostrar_tf_idf = False
    

if 'mostrar_knn' not in st.session_state:
    st.session_state.mostrar_knn= False 
 
    

# Inicializar el estado de la sesión para las calificaciones del usuario si no existe
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}  # Diccionario para almacenar las calificaciones del usuario

# Inicializar el estado de la sesión para almacenar las calificaciones de los usuarios
if 'user_profiles' not in st.session_state:
    st.session_state.user_profiles = {
        'user1': {},
        'user2': {},
        'user3': {}
    }
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = 'user1'

# Función para resetear los estados
def reset_states():
    st.session_state.mostrar_jaccard = False
    st.session_state.mostrar_bayesiana = False
    st.session_state.mostrar_tf_idf = False 
    st.session_state.mostrar_knn= False

# Botón en la barra lateral para "jaccard"
if st.sidebar.button("jaccard"):
    reset_states()
    st.session_state.mostrar_jaccard = True

# Botón en la barra lateral para "bayesiana"
if st.sidebar.button("bayesiana"):
    reset_states()
    st.session_state.mostrar_bayesiana = True
    
# Botón en la barra lateral para "tf-idf"
#if st.sidebar.button("TF-IDF"):
#    reset_states()
#    st.session_state.mostrar_tf_idf = True    
    
# Botón en la barra lateral para "KNN"
if st.sidebar.button("knn"):
    reset_states()
    st.session_state.mostrar_knn = True 
    


# Mostrar lógica de Jaccard solo si el botón ha sido presionado
if st.session_state.mostrar_jaccard:
    st.title("Recomendador de Películas Basado en Similitud de Jaccard por Usuario")
    
    # Selector de usuario
    st.header("Selecciona tu Perfil")
    selected_user = st.selectbox('Usuario:', options=list(st.session_state.user_profiles.keys()))
    st.session_state.selected_user = selected_user

    # Interfaz para calificar películas
    st.header("Califica Películas")
    titulos_disponibles = df_movies['title'].unique()
    titulo_seleccionado = st.selectbox('Selecciona una Película para Calificar:', options=titulos_disponibles)

    if titulo_seleccionado:
        # Mostrar las estrellas para calificar
        st.write("Califica con Estrellas:")
        cols = st.columns(5)  # Crear 5 columnas para 5 estrellas
        for i in range(1, 6):
            if cols[i-1].button("⭐", key=f"star_{i}", help=f"{i} estrellas"):
                movie_id = df_movies[df_movies['title'] == titulo_seleccionado]['movieId'].values[0]
                st.session_state.user_profiles[st.session_state.selected_user][movie_id] = i
                st.write(f"Calificación guardada: {titulo_seleccionado} - {i} estrellas")

        # Mostrar recomendaciones basadas en las calificaciones del usuario
        if st.session_state.user_profiles[st.session_state.selected_user]:
            recomendaciones = recomendacion_jaccard_por_usuario(
                st.session_state.user_profiles[st.session_state.selected_user], 
                df_movies, 
                df_links, 
                n_recommendations=5
            )

            if recomendaciones is not None and not recomendaciones.empty:
                st.write("Películas Recomendadas Basadas en Tus Calificaciones:")
                cols = st.columns(5) 

                for index, row in recomendaciones.iterrows():
                    col = cols[index % 5]
                    with col:
                        st.image(row['image_url'], caption=row['title'], use_column_width=True)
            else:
                st.write("No hay suficientes calificaciones altas para generar recomendaciones.")
        else:
            st.write("No has calificado ninguna película aún. Por favor, califica algunas películas para obtener recomendaciones.")

# Función para mostrar la lista de películas
def mostrar_lista_peliculas(genero):
    recomendaciones = recomendacion_populares(genero=genero)

    st.title("Top 10 Películas Populares:")
    cols = st.columns(5)

    for index, row in recomendaciones.iterrows():
        col = cols[index % 5]
        with col:
            if st.button("Detalles", key=f"button_{index}", help=row['title']):
                st.session_state.selected_movie = row.to_dict()
                st.session_state.mostrar_detalle = True
            st.image(row['image_url'], use_column_width=True)
# Mostrar lógica de Bayesiana solo si el botón ha sido presionado
if st.session_state.mostrar_bayesiana:
    generos_disponibles = df_final['genres'].str.split('|').explode().unique()
    genero_seleccionado = st.selectbox('Selecciona un Género:', options=generos_disponibles)
    # Llamar a la función para mostrar la lista de películas
    mostrar_lista_peliculas(genero=genero_seleccionado)


# Función para mostrar recomendaciones basadas en KNN en Streamlit
def mostrar_recomendaciones_knn():
    st.title("Recomendador de Películas Basado en KNN")

    # Selector de usuario
    selected_user = st.selectbox('Selecciona un Usuario:', options=ratings_matrix.index)

    # Obtener recomendaciones para el usuario seleccionado
    recomendaciones = recomendacion_knn(selected_user, ratings_matrix_normalized, ratings_matrix, df_movies, knn_model)

    if recomendaciones is not None and not recomendaciones.empty:
        st.write("Películas Recomendadas Basadas en KNN:")
        cols = st.columns(5)  # Ajustar el número de columnas para mostrar las imágenes

        for index, row in recomendaciones.iterrows():
            col = cols[index % 5]  # Usar módulo para iterar sobre las columnas
            with col:
                st.image(row['image_url'], caption=row['title'], use_column_width=True)
    else:
        st.write("No se encontraron suficientes datos para generar recomendaciones para el usuario seleccionado.")

if st.session_state.mostrar_knn:
    # Mostrar la interfaz para recomendaciones basadas en KNN
    mostrar_recomendaciones_knn()    




def detalles():
    # Mostrar detalles de la película seleccionada
    if st.session_state.selected_movie:
        movie = st.session_state.selected_movie
        st.title("Detalles de la Película Seleccionada")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.write(f"**Título:** {movie['title']}")
            st.write(f"**Rating Promedio:** {movie['mean_rating']:.1f}")
            st.write(f"**Conteo de Votos:** {movie['vote_count']}")
            st.write(f"**Score Ponderado:** {movie['weighted_score']:.2f}")

            # Calificación con estrellas usando botones
            st.write("Califica esta película:")
            cols = st.columns(5)  # Crear 5 columnas para 5 estrellas
            for i in range(1, 6):
                if cols[i-1].button("⭐", key=f"star_{i}"):
                    st.session_state.user_rating = i  # Actualizar el rating en el estado de la sesión

            # Mostrar la calificación actual del usuario
            st.write(f"Tu calificación: {'⭐' * st.session_state.user_rating}{'☆' * (5 - st.session_state.user_rating)}")

        with col2:
            st.image(movie['image_url'], caption=movie['title'], use_column_width=True)

        # Botón para volver a la lista de películas
        if st.button("Volver a la Lista de Películas", on_click=lambda: reset_movie_selection()):
            pass

    # Función para restablecer la selección de película
    def reset_movie_selection():
        st.session_state.selected_movie = None  # Regresa a la vista de selección de películas
        st.session_state.mostrar_lista = True  # Asegura que se muestre la lista de imágenes de nuevo
        st.session_state.mostrar_selector = True  # Asegura que se muestre el selector de género de nuevo
        st.session_state.mostrar_detalle = False  # Ocultar el botón de detalles en la barra lateral

    # Mostrar el botón "Detalles" en la barra lateral solo si hay una película seleccionada
    if st.session_state.mostrar_detalle:
        if st.sidebar.button("Detalles"):
            # Acción al hacer clic en el botón de detalles en la barra lateral
            st.session_state.mostrar_selector = False  # Ocultar selector para dejar espacio al detalle
#detalles()    


st.write(df_movies.columns)

# Función para mostrar las recomendaciones basadas en TF-IDF
def mostrar_recomendaciones_tf_idf():
    st.title("Recomendador de Películas Basado en TF-IDF")

    # Selector de películas para elegir el título
    titulos_disponibles = df_movies['title'].unique()
    titulo_seleccionado = st.selectbox('Selecciona una Película para Recomendaciones:', options=titulos_disponibles)

    if titulo_seleccionado:
        # Obtener recomendaciones basadas en el título seleccionado
        recomendaciones = recomendacion_tf_idf_por_titulo(titulo_seleccionado, n_recommendations=5)

        if recomendaciones is not None and not recomendaciones.empty:
            st.write("Películas Recomendadas Basadas en Similitud de Contenido:")
            cols = st.columns(5)  # Ajustar el número de columnas para mostrar las imágenes

            for index, row in recomendaciones.iterrows():
                col = cols[index % 5]  # Usar módulo para iterar sobre las columnas
                with col:
                    st.image(row['image_url'], caption=row['title'], use_column_width=True)
        else:
            st.write("No se encontraron recomendaciones para la película seleccionada.")

if st.session_state.mostrar_tf_idf:
    # Mostrar la interfaz para recomendaciones basadas en TF-IDF
    mostrar_recomendaciones_tf_idf()
