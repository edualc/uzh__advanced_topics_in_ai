from .utils import *
import jsonpickle


class SimpleRecommendationBot:
    def __init__(self, g):
        self.g = g

        WD, WDT, SCHEMA, DDIS, RDFS = load_namespaces()
        self.namespaces = {'WD': WD, 'WDT': WDT, 'SCHEMA': SCHEMA, 'DDIS': DDIS, 'RDFS': RDFS}

        self._setup_movies_by_letter()
        self._setup_genres_by_letter()
        self._setup_year_groups()
        self._setup_movie_vectors()
        self._setup_actors_by_letter()


    def _setup_year_groups(self):
        self.year_groups = []
        for start,end in zip(np.linspace(1900, 2025, 26)[0:-1], np.linspace(1900, 2025, 26)[1:]):
            self.year_groups.append((start, end))


    def _create_genre_vector(self, genre_labels):
        genre_vector = np.zeros(self.N_genres)
        for genre_label in genre_labels:
            genre_vector[self.genre2idx[genre_label.lower()]] = 1
        return genre_vector
    

    def _create_year_vector(self, production_year):
        year_vector = np.zeros(self.N_year_groups)
        for i, (start, end) in enumerate(self.year_groups):
            if start <= production_year - 15 <= end:
                year_vector[i] = 0.2
            if start <= production_year - 10 <= end:
                year_vector[i] = 0.7
            if start <= production_year - 5 <= end:
                year_vector[i] = 0.8
            if start <= production_year <= end:
                year_vector[i] = 1
            if start <= production_year + 5 <= end:
                year_vector[i] = 0.8
            if start <= production_year + 10 <= end:
                year_vector[i] = 0.7
            if start <= production_year + 15 <= end:
                year_vector[i] = 0.2
        return year_vector


    def _setup_movie_vectors(self):
        self.N_genres = len(self.genre2id.keys())
        self.N_year_groups = len(self.year_groups)
        self.N_movies = len(self.movie_genres.keys())
        self.genre2idx = {genre: i for i, genre in enumerate(self.genre2id.keys())}
        self.movie2idx = {movie: i for i, movie in enumerate(self.movie_genres.keys())}
        
        self.movie_vectors = np.zeros((self.N_movies, self.N_genres + self.N_year_groups))

        for i, movie in enumerate(self.movie_genres.keys()):
            movie_data = self.movie_genres[movie]

            genre_labels = [genre_label for _, genre_label in movie_data['genres']]
            genre_vec = self._create_genre_vector(genre_labels)

            production_year = np.min(list(movie_data['prod_year']))
            year_vec = self._create_year_vector(production_year)

            self.movie_vectors[i, :] = np.concatenate([genre_vec, year_vec])

    def _extract_actors_from_graph(self):
        actors_path = 'dataset/processed/actors.json'
        if os.path.exists(actors_path):
            with open(actors_path, 'r') as ifile:
                return jsonpickle.decode(ifile.read())

        actor_query = f"""
        PREFIX wd: <{self.namespaces['WD']}>
        PREFIX wdt: <{self.namespaces['WDT']}>
        SELECT ?actor ?actorLabel ?imdbID
        WHERE {{
            ?actor wdt:P106 wd:Q33999 .
            ?actor rdfs:label ?actorLabel .
            FILTER(LANG(?actorLabel) = 'en') .
            ?actor wdt:P345 ?imdbID
        }}
        """

        result = self.g.query(actor_query)
        tmp = {}
        for row in result:
            label = row['actorLabel'].toPython().lower()
            actor = row['actor'].toPython().split('/')[-1]
            tmp[label] = {
                'full_name': row['actorLabel'].toPython(),
                'entity': actor,
                'imdbID': row['imdbID'].toPython()
            }

        with open(actors_path, 'w') as ofile:
            ofile.write(jsonpickle.encode(tmp))

        return tmp

    def _extract_data_from_graph(self):
        movie_genres_path = 'dataset/processed/movie_genres.json'
        if os.path.exists(movie_genres_path):
            with open(movie_genres_path, 'r') as ifile:
                return jsonpickle.decode(ifile.read())

        query = f"""
        PREFIX wd: <{self.namespaces['WD']}>
        PREFIX wdt: <{self.namespaces['WDT']}>
        SELECT ?movieLabel (YEAR(?prodDate) as ?prodYear) ?movieId ?genreId ?genreLabel
        WHERE {{
            ?movieId wdt:P136 ?genreId.
            ?movieId wdt:P31 wd:Q11424.
            ?movieId wdt:P577 ?prodDate.
            ?movieId rdfs:label ?movieLabel .
            FILTER(LANG(?movieLabel) = 'en').
            ?genreId rdfs:label ?genreLabel.
            FILTER(LANG(?genreLabel) = 'en')
        }}
        """
        result = self.g.query(query)
        tmp = {}
        for row in result:
            label = row['movieLabel'].toPython().lower()
            movie = row['movieId'].toPython().split('/')[-1]
            genre = row['genreId'].toPython().split('/')[-1]
            genre_label =  row['genreLabel'].toPython()
            if label not in tmp:
                tmp[label] = {
                    'full_name': row['movieLabel'].toPython(),
                    'entities': set(),
                    'genres': set(),
                    'prod_year': set()
                }
            tmp[label]['genres'].add((genre, genre_label))
            tmp[label]['entities'].add(movie)
            tmp[label]['prod_year'].add(row['prodYear'].toPython())

        with open(movie_genres_path, 'w') as ofile:
            ofile.write(jsonpickle.encode(tmp))

        return tmp


    def _setup_movies_by_letter(self):
        self.movie_genres = self._extract_data_from_graph()

        # create a dictionary where the key are the first letters of the alphabet
        # where as values are all movies that start with that letter
        self.movies_by_two_letters = {}
        self.movie2id = {}
        for movie in self.movie_genres:
            letter = movie[:2].lower()
            if letter not in self.movies_by_two_letters:
                self.movies_by_two_letters[letter] = []
            self.movies_by_two_letters[letter].append(movie.lower())
            self.movie2id[movie.lower()] = self.movie_genres[movie]['entities']


    def _setup_genres_by_letter(self):
        self.genres_by_two_letters = {}
        self.genre2id = {}
        for movie in self.movie_genres:
            for genre_id, genre_label in self.movie_genres[movie]['genres']:
                letter = genre_label[:2].lower()
                if letter not in self.genres_by_two_letters:
                    self.genres_by_two_letters[letter] = []
                if genre_label.lower() not in self.genres_by_two_letters[letter]:
                    self.genres_by_two_letters[letter].append(genre_label.lower())
                self.genre2id[genre_label.lower()] = genre_id
    

    def _setup_actors_by_letter(self):
        self.actors = self._extract_actors_from_graph()

        self.actors_by_two_letters = {}
        self.actor2id = {}
        self.actor2imdb = {}
        for actor in self.actors:
            letter = actor[:2].lower()
            if letter not in self.actors_by_two_letters:
                self.actors_by_two_letters[letter] = []
            self.actors_by_two_letters[letter].append(actor.lower())
            self.actor2id[actor.lower()] = self.actors[actor]['entity']
            self.actor2imdb[actor.lower()] = self.actors[actor]['imdbID']

    def get_all_token_sequences(self, text, min_length=1, max_length=None):
        if text.endswith('?') or text.endswith(',') or text.endswith('.'):
            text = text[:-1]
        text = text.replace(',','')
        
        tokens = text.split()
        sequences = dict()
        
        # If max_length is not specified, use the full length of tokens
        if max_length is None:
            max_length = len(tokens)
        
        # Generate sequences with length constraints
        for start in range(len(tokens)):
            for length in range(min_length, min(max_length + 1, len(tokens) - start + 1)):
                sequence = ' '.join(tokens[start:start + length]).lower()
                
                if start not in sequences:
                    sequences[start] = []
                sequences[start].append(sequence)

        return sequences


    def get_entities__check_movies(self, test_string, overlap_key, overlap_position_key, overlap_dict):
        if overlap_key not in self.movies_by_two_letters:
            return []
        
        longest_substring = None
        tmp = []
                
        substrings = overlap_dict[overlap_position_key]
        movie_candidates = self.movies_by_two_letters[overlap_key]
        
        # check if any substring is in the movies list
        for substring in substrings:
            if substring in movie_candidates:
                if longest_substring is None or len(substring) > len(longest_substring):
                    longest_substring = substring
                
        if longest_substring is not None:
            # print(f"Substring: {longest_substring}")
            # find index of the substring in the original text in terms of tokens
            start_index = test_string.lower().find(longest_substring)
            end_index = start_index + len(longest_substring)
            # print(f"Start index: {start_index}, End index: {end_index}")
            tmp.append((longest_substring, start_index, end_index))
        
        return tmp
    
    def get_entities__check_genres(self, test_string, overlap_key, overlap_position_key, overlap_dict):
        if overlap_key not in self.genres_by_two_letters:
            return []
        
        longest_substring = None
        tmp = []
                
        substrings = overlap_dict[overlap_position_key]
        genre_candidates = self.genres_by_two_letters[overlap_key]
        
        # check if any substring is in the genres list
        for substring in substrings:
            substring_candidates = set([substring, f"{substring} film", f"{substring} story"])
            found_overlap = substring_candidates & set(genre_candidates)

            for candidate in found_overlap:
                if longest_substring is None or len(candidate) > len(longest_substring):
                    longest_substring = candidate
                
        if longest_substring is not None:
            # print(f"Substring: {longest_substring}")
            # find index of the substring in the original text in terms of tokens
            start_index = test_string.lower().find(longest_substring)
            end_index = start_index + len(longest_substring)
            # print(f"Start index: {start_index}, End index: {end_index}")
            tmp.append((longest_substring, start_index, end_index))

        return tmp
    
    def get_entities__check_actors(self, test_string, overlap_key, overlap_position_key, overlap_dict):
        if overlap_key not in self.actors_by_two_letters:
            return []
        
        longest_substring = None
        tmp = []
                
        substrings = overlap_dict[overlap_position_key]
        actor_candidates = self.actors_by_two_letters[overlap_key]
        
        # check if any substring is in the actors list
        for substring in substrings:
            if substring in actor_candidates:
                if longest_substring is None or len(substring) > len(longest_substring):
                    longest_substring = substring
                
        if longest_substring is not None:
            # print(f"Substring: {longest_substring}")
            # find index of the substring in the original text in terms of tokens
            start_index = test_string.lower().find(longest_substring)
            end_index = start_index + len(longest_substring)
            # print(f"Start index: {start_index}, End index: {end_index}")
            if longest_substring != 'you':
                tmp.append((longest_substring, start_index, end_index))
        
        return tmp

    def get_entities(self, test_string):
        overlap_dict = self.get_all_token_sequences(test_string)

        genre_results = []
        substring_results = []
        actor_results = []
        for overlap_position_key in overlap_dict:
            overlap_key = overlap_dict[overlap_position_key][0][:2]

            substring_results.extend(
                self.get_entities__check_movies(test_string, overlap_key, overlap_position_key, overlap_dict)
            )

            genre_results.extend(
                self.get_entities__check_genres(test_string, overlap_key, overlap_position_key, overlap_dict)
            )

            actor_results.extend(
                self.get_entities__check_actors(test_string, overlap_key, overlap_position_key, overlap_dict)
            )

        found_movies = self._cleanup_results(substring_results)
        found_genres = self._cleanup_results(genre_results)
        found_actors = self._cleanup_results(actor_results)
        return found_movies, found_genres, found_actors
    

    def _cleanup_results(self, substring_results):
        substring_results = sorted(substring_results, key=lambda x: len(x[0]), reverse=True)
        final_results = []
        for i, (substring, start, end) in enumerate(substring_results):
            if substring in final_results:
                continue

            is_substring = False
            for j, (other_substring, other_start, other_end) in enumerate(substring_results):
                if i != j and start >= other_start and end <= other_end:
                    is_substring = True
                    break
                if i != j and substring == other_substring:
                    final_results.append(substring)

            if not is_substring:
                final_results.append(substring)

        return final_results


    def recommend_movies(self, test_string):
        found_movies, found_genres, found_actors = self.get_entities(test_string)

        if len(found_movies) == 0 and len(found_genres) == 0:
            return "Hmm, I couldn't find any movies or genres in your query. Could you rephrase?"

        query_vector = np.zeros(self.N_genres + self.N_year_groups)
        for genre in found_genres:
            if genre in self.genre2idx:
                query_vector[self.genre2idx[genre]] = 1

        for movie in found_movies:
            if movie in self.movie2id:
                query_vector += self.movie_vectors[self.movie2idx[movie]]

        movie_scores = np.dot(self.movie_vectors, query_vector)
        # remove movies that we queried with
        for movie in found_movies:
            if movie in self.movie2idx:
                movie_scores[self.movie2idx[movie]] = 0
        movie_scores = [(movie, score) for movie, score in zip(self.movie2idx.keys(), movie_scores)]
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        recommended_movies = []
        for movie_label, score in movie_scores[:3]:
            recommended_movies.append(self.movie_genres[movie_label]['full_name']) 

        return f"Let me think... Ah, yes. Have you tried watching: '{recommended_movies[0]}', '{recommended_movies[1]}' or maybe even '{recommended_movies[2]}'?"
