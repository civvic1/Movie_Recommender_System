# Importaciones y carga de datos
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Inicialización de st.session_state
if 'user_profiles' not in st.session_state:
    st.session_state.user_profiles = {}
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = 'Jaccard'  # Opción predeterminada
if 'mostrar_detalle' not in st.session_state:
    st.session_state.mostrar_detalle = False
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# Función para obtener la URL de la imagen de una película
def get_movie_image_url(tmdb_id, api_key='64b35ee705c55e7ccdd7c754a9b8cdc4'):
    if tmdb_id is None or np.isnan(tmdb_id):
        return "https://via.placeholder.com/150"
    url = f'https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return "https://via.placeholder.com/150"

# Función de preprocesamiento
@st.cache_data
def preprocesamiento():
    df_movies = pd.read_csv('data/ml-latest-small/movies.csv')
    df_ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    df_links = pd.read_csv('data/ml-latest-small/links.csv')

    # Eliminar registros con valores nulos
    df_ratings = df_ratings.dropna()
    df_movies = df_movies.dropna()
    df_links = df_links.dropna()

    # Eliminar registros duplicados
    df_movies = df_movies.drop_duplicates(subset='movieId', keep='first').reset_index(drop=True)
    df_ratings = df_ratings.drop_duplicates(subset=['movieId', 'userId'], keep='first').reset_index(drop=True)
    df_links = df_links.drop_duplicates(subset='movieId', keep='first').reset_index(drop=True)

    # Crear la columna 'content' reemplazando '|' por ' '
    df_movies['content'] = df_movies['genres'].str.replace('|', ' ')
    # Crear la columna 'genre_set' como un conjunto de géneros
    df_movies['genre_set'] = df_movies['genres'].apply(lambda x: set(x.split('|')))

    # Convertir la columna 'timestamp' a tipo datetime
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

    # Fusionar los DataFrames
    df_final = pd.merge(df_ratings, df_movies, on='movieId', how='inner')
    df_final = pd.merge(df_final, df_links, on='movieId', how='inner')

    # Crear una columna 'year' basada en el año de 'timestamp'
    df_final['year'] = df_final['timestamp'].dt.year

    # Eliminar la columna 'timestamp' si no es necesaria
    df_final.drop(columns=['timestamp'], inplace=True)

    return df_movies, df_ratings, df_links, df_final

# Llamar a la función de preprocesamiento
df_movies, df_ratings, df_links, df_final = preprocesamiento()

# Función para calcular la similitud de Jaccard
def similitud_jaccard(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0

# Función de recomendación basada en usuario usando Jaccard
def recomendacion_jaccard_por_usuario(ratings, df_items, df_links, n_recommendations=5):
    # Filtrar las películas que el usuario ha calificado con 4 o más estrellas
    rated_movie_ids = [movie_id for movie_id, rating in ratings.items() if rating >= 4]
    # Asegurarse de que hay suficientes calificaciones altas para hacer recomendaciones
    if not rated_movie_ids:
        return None
    # Obtener los géneros de las películas calificadas
    rated_genres = df_items[df_items['movieId'].isin(rated_movie_ids)]['genre_set']
    # Unir los géneros para crear un perfil del usuario
    user_profile = set().union(*rated_genres)
    # Calcular la similitud de Jaccard entre el perfil del usuario y todas las películas
    df_items['similaridad'] = df_items['genre_set'].apply(lambda x: similitud_jaccard(user_profile, x))
    # Excluir las películas ya calificadas por el usuario
    df_recommendations = df_items[~df_items['movieId'].isin(rated_movie_ids)].sort_values('similaridad', ascending=False).reset_index(drop=True)
    # Obtener las mejores recomendaciones
    result = df_recommendations.head(n_recommendations)
    # Añadir la columna con la URL de la imagen usando el movieId de df_links
    result = result.merge(df_links[['movieId', 'tmdbId']], on='movieId', how='left')
    result['image_url'] = result['tmdbId'].apply(lambda x: get_movie_image_url(x))
    return result

# Función de recomendación basada en TF-IDF
def recomendacion_tf_idf_por_titulo(titulo_pelicula, n_recommendations=5):
    # Lista de palabras comunes en español para excluir del análisis
    spanish_stop_words = [
        'de', 'la', 'que', 'el', 'en', 'un', 'con', 'y', 'a', 'los', 'se',
        'no', 'me', 'si', 'por', 'su', 'para', 'esta', 'tener', 'hacer',
        'poder', 'decir', 'querer', 'poder', 'ser', 'tener', 'hacer'
    ]

    # Crear la matriz TF-IDF a partir de los títulos de las películas
    tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['title'])

    # Calcular la matriz de similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Obtener el índice de la película seleccionada
    try:
        idx = df_movies[df_movies['title'].str.lower() == titulo_pelicula.lower()].index[0]
    except IndexError:
        st.write(f"Error: La película con título '{titulo_pelicula}' no se encontró.")
        return None

    # Obtener las puntuaciones de similitud de coseno para todas las películas
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar las películas por similitud en orden descendente, excluyendo la película seleccionada
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, score) for i, score in sim_scores if i != idx]

    # Obtener los índices de las n recomendaciones más similares
    top_n = sim_scores[:n_recommendations]

    # Obtener detalles de las películas recomendadas
    recommended_movies = []
    for i, score in top_n:
        movie = df_movies.iloc[i]
        movie_id = movie['movieId']
        title = movie['title']
        genres = movie['genres']
        similarity = round(score, 2)

        # Obtener tmdbId y URL de la imagen
        tmdb_id = df_links[df_links['movieId'] == movie_id]['tmdbId'].values
        tmdb_id = tmdb_id[0] if len(tmdb_id) > 0 else None
        image_url = get_movie_image_url(tmdb_id)

        recommended_movies.append({
            'movieId': movie_id,
            'title': title,
            'genres': genres,
            'similarity': similarity,
            'image_url': image_url
        })

    # Convertir la lista de recomendaciones en un DataFrame
    recommended_movies_df = pd.DataFrame(recommended_movies)

    return recommended_movies_df

# Preparación para el modelo KNN
# Crear la matriz de calificaciones con usuarios como filas y películas como columnas
ratings_matrix = df_final.pivot(index='userId', columns='movieId', values='rating')
# Calcular el promedio de calificaciones por usuario
avg_ratings = ratings_matrix.mean(axis=1)
# Normalizar la matriz de calificaciones restando el promedio de cada usuario
ratings_matrix_normalized = ratings_matrix.sub(avg_ratings, axis=0).fillna(0)

# Entrenar el modelo KNN usando la similitud del coseno
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(ratings_matrix_normalized.values)

# Función de recomendación basada en KNN
def recomendacion_knn(usuario, ratings_matrix_normalized, ratings_matrix, df_movies, df_links, n_recommendations=5):
    if usuario in ratings_matrix.index:
        # Si el usuario existe en la matriz de calificaciones
        user_idx = ratings_matrix.index.get_loc(usuario)
        distances, indices = knn_model.kneighbors(ratings_matrix_normalized.iloc[user_idx].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
    else:
        st.write("El usuario seleccionado no existe en la base de datos.")
        return None

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
    recommendations = recommendations.join(movie_info, on='movieId')
    recommendations = recommendations.reset_index()

    # Añadir la columna con la URL de la imagen usando el movieId de df_links
    recommendations = recommendations.merge(df_links[['movieId', 'tmdbId']], on='movieId', how='left')
    recommendations['image_url'] = recommendations['tmdbId'].apply(lambda x: get_movie_image_url(x))

    return recommendations

# Sidebar con opciones
st.sidebar.title("Opciones de Recomendación")
opcion_seleccionada = st.sidebar.radio(
    "Selecciona un método de recomendación:",
    ('Jaccard', 'Bayesiana', 'TF-IDF', 'KNN'),
    key='opcion_recomendacion'
)
st.session_state.selected_option = opcion_seleccionada

# Lógica principal de la aplicación
if st.session_state.selected_option == 'Jaccard':
    st.title("Recomendador de Películas Basado en Similitud de Jaccard por Usuario")

    # Sección para agregar nuevos usuarios
    st.header("Crear Nuevo Usuario")
    new_user = st.text_input("Ingresa el nombre del nuevo usuario:")
    if st.button("Agregar Usuario"):
        if new_user:
            if new_user not in st.session_state.user_profiles:
                st.session_state.user_profiles[new_user] = {}
                st.success(f"Usuario '{new_user}' agregado exitosamente.")
            else:
                st.warning(f"El usuario '{new_user}' ya existe.")
        else:
            st.error("Por favor, ingresa un nombre de usuario válido.")

    # Selector de usuario
    st.header("Selecciona tu Perfil")
    usuarios = list(st.session_state.user_profiles.keys())
    if not usuarios:
        st.write("No hay usuarios disponibles. Por favor, crea un nuevo usuario.")
    else:
        selected_user = st.selectbox('Usuario:', options=usuarios, key="user_select_jaccard")
        st.session_state.selected_user = selected_user

        # Interfaz para calificar películas
        st.header("Califica Películas")
        titulos_disponibles = df_movies['title'].unique()
        titulo_seleccionado = st.selectbox('Selecciona una Película para Calificar:', options=titulos_disponibles, key="movie_select_jaccard")

        if titulo_seleccionado:
            # Mostrar las estrellas para calificar
            st.write("Califica con Estrellas:")
            cols = st.columns(5)  # Crear 5 columnas para 5 estrellas
            for i in range(1, 6):
                if cols[i-1].button("⭐", key=f"star_jaccard_{i}_{titulo_seleccionado}", help=f"{i} estrellas"):
                    movie_id = df_movies[df_movies['title'] == titulo_seleccionado]['movieId'].values[0]
                    if selected_user not in st.session_state.user_profiles:
                        st.session_state.user_profiles[selected_user] = {}
                    st.session_state.user_profiles[selected_user][movie_id] = i
                    st.success(f"Calificación guardada: {titulo_seleccionado} - {i} estrellas")

            # Mostrar calificaciones del usuario seleccionado en el centro de la página
            st.header(f"Películas calificadas por {selected_user}:")
            user_ratings = st.session_state.user_profiles[selected_user]
            if user_ratings:
                # Crear una copia del DataFrame filtrado
                user_rated_movies = df_movies[df_movies['movieId'].isin(user_ratings.keys())].copy()
                user_rated_movies['Calificación'] = user_rated_movies['movieId'].map(user_ratings)
                st.table(user_rated_movies[['title', 'Calificación']])
            else:
                st.write(f"{selected_user} aún no ha calificado ninguna película.")

            # Mostrar recomendaciones basadas en las calificaciones del usuario
            user_ratings = st.session_state.user_profiles.get(selected_user, {})
            if user_ratings:
                recomendaciones = recomendacion_jaccard_por_usuario(
                    user_ratings,
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

elif st.session_state.selected_option == 'TF-IDF':
    st.title("Recomendador de Películas Basado en TF-IDF")

    # Selector de películas para elegir el título
    titulos_disponibles = df_movies['title'].unique()
    titulo_seleccionado = st.selectbox('Selecciona una Película para Recomendaciones:', options=titulos_disponibles, key="tfidf_movie_select")

    if titulo_seleccionado:
        # Obtener recomendaciones basadas en el título seleccionado
        recomendaciones = recomendacion_tf_idf_por_titulo(titulo_seleccionado, n_recommendations=5)

        if recomendaciones is not None and not recomendaciones.empty:
            st.write("Películas Recomendadas Basadas en Similitud de Contenido:")
            cols = st.columns(5)

            for index, row in recomendaciones.iterrows():
                col = cols[index % 5]
                with col:
                    st.image(row['image_url'], caption=row['title'], use_column_width=True)
        else:
            st.write("No se encontraron recomendaciones para la película seleccionada.")

elif st.session_state.selected_option == 'KNN':
    st.title("Recomendador de Películas Basado en KNN")

    # Selector de usuario
    usuarios_disponibles = ratings_matrix.index.tolist()
    selected_user = st.selectbox('Selecciona un Usuario:', options=usuarios_disponibles, key="user_select_knn")

    if selected_user:
        # Obtener recomendaciones
        recomendaciones = recomendacion_knn(selected_user, ratings_matrix_normalized, ratings_matrix, df_movies, df_links, n_recommendations=5)

        if recomendaciones is not None and not recomendaciones.empty:
            st.write("Películas Recomendadas Basadas en KNN:")
            cols = st.columns(5)

            for index, row in recomendaciones.iterrows():
                col = cols[index % 5]
                with col:
                    st.image(row['image_url'], caption=row['title'], use_column_width=True)
        else:
            st.write("No se encontraron suficientes datos para generar recomendaciones para el usuario seleccionado.")

elif st.session_state.selected_option == 'Bayesiana':
    st.title("Recomendador de Películas Basado en Método Bayesiano")
    ## Define tu fórmula en LaTeX
    #formula = r'''
    #WR = vv + m \times R + mv + m \times C
    #'''
    ## Muestra la fórmula
    #st.latex(formula)
    # Obtener los géneros disponibles
    generos_disponibles = df_movies['genres'].str.split('|').explode().unique()

    # Selector de géneros
    genero_seleccionado = st.selectbox('Selecciona un Género:', options=generos_disponibles)

    # Obtener las recomendaciones
    recomendaciones = df_movies[df_movies['genres'].str.contains(genero_seleccionado)].copy()
    recomendaciones['rating_mean'] = recomendaciones['movieId'].map(df_ratings.groupby('movieId')['rating'].mean())
    recomendaciones['rating_count'] = recomendaciones['movieId'].map(df_ratings.groupby('movieId')['rating'].count())

    # Calcular la calificación bayesiana
    C = df_ratings['rating'].mean()
    m = 10  # Puedes ajustar este parámetro
    recomendaciones['bayesian_rating'] = (recomendaciones['rating_mean'] * recomendaciones['rating_count'] + C * m) / (recomendaciones['rating_count'] + m)

    # Ordenar las películas por calificación bayesiana
    recomendaciones = recomendaciones.sort_values('bayesian_rating', ascending=False)

    # Mostrar las mejores recomendaciones
    top_recommendations = recomendaciones.head(5)
    top_recommendations = top_recommendations.merge(df_links[['movieId', 'tmdbId']], on='movieId', how='left')
    top_recommendations['image_url'] = top_recommendations['tmdbId'].apply(lambda x: get_movie_image_url(x))

    # Mostrar las películas y los botones de detalles
    st.title("Top 5 Películas Populares:")
    cols = st.columns(5)

    for index, row in top_recommendations.iterrows():
        col = cols[index % 5]
        with col:
            if st.button("Detalles", key=f"button_bayesiana_{index}", help=row['title']):
                st.session_state.selected_movie = row.to_dict()
                st.session_state.mostrar_detalle = True
            st.image(row['image_url'], use_column_width=True)

# Mostrar detalles de la película seleccionada (si aplica)
if st.session_state.mostrar_detalle and st.session_state.selected_movie:
    st.write("Detalles de la película:")
    st.write(f"Título: {st.session_state.selected_movie['title']}")
    st.write(f"Géneros: {st.session_state.selected_movie['genres']}")
    st.image(st.session_state.selected_movie['image_url'], use_column_width=True)
    st.session_state.mostrar_detalle = False  # Restablecer el estado después de mostrar los detalles
