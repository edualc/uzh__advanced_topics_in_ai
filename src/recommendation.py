
from rdflib.term import URIRef
from sklearn.metrics import pairwise_distances
import numpy as np


class RecommendationBot:
    def __init__(self, bot_ner, g, nodes, predicates, genres, movies, url2nodes, entity_emb, rel2id, ent2id, id2ent, relation_emb):
        self.bot_ner = bot_ner
        self.g = g
        self.genres = genres
        self.movies = movies
        self.nodes = nodes
        self.predicates = predicates
        self.url2nodes = url2nodes
        self.entity_emb = entity_emb
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.id2ent = id2ent
        self.relation_emb = relation_emb

    def get_recommendation(self, input):
        # try:
        ner_results=self.bot_ner._ner_results(input)
        if len(ner_results)==0:
            genre_search = [(i,i+" film") for i in input.replace(".","").split(" ")]
            matched_genres = []
            for word,word_film in genre_search:
                if word in self.genres.keys():
                    matched_genres.append((word, self.genres[word]))
                if word_film in self.genres.keys():
                    matched_genres.append((word_film, self.genres[word_film]))
            recommendation = self.recommend_genre_embed([url for label, url in matched_genres])
            if recommendation[0] is None:
                return "Sorry, I cannot recommend you a movie based on your query. The reasons might be that I do not know the movies you mentioned or there is a minor problem with the format of your input. You might want to re-check and/or rephrase your sentence. I will be waiting here. "
            genres_matched = ""
            for matched in list(dict.fromkeys([label.replace(" film","") for label,url in matched_genres])):
                genres_matched +=matched + ", "
            genres_matched = genres_matched.strip()[:-1]
            return f"If you like {genres_matched} movies, I would recommend you watch {recommendation[0][1]}."
            
        entities = self.bot_ner.entity_extraction(ner_results, input)
        entities_matched = [self.bot_ner.match_things(self.nodes, ent) for ent in entities]
        urls_entities = [url for ent,url in entities_matched]
        try:
            pub_dates = sorted(list(dict.fromkeys([[str(o)[:3] for s,p,o in self.g.triples((URIRef(url), URIRef(self.predicates["publication date"]), None))][0]
                            for url in urls_entities])))
        except:
            pub_dates=[]
        recommendation = self.recommend_genre_embed(urls_entities)
        if recommendation[0] is None:
            return "Sorry, I cannot recommend you a movie based on your query. The reasons might be that I do not know the movies you mentioned or there is a minor problem with the format of your input. You might want to re-check and/or rephrase your sentence. I will be waiting here "
        genres_matched = ""
        genres_checked = []
        for i in range(len(recommendation[1])):
            if recommendation[1][i][1].split(" ")[0]=="film":
                continue
            genres_checked.append(recommendation[1][i][1].replace(' film',''))
        genres_matched += ", ".join(genres_checked)
        genres_matched = genres_matched
        if len(pub_dates)==2:
            genres_matched += " from around " + pub_dates[0] +"0s or " + pub_dates[1] + "0s"
        elif len(pub_dates)==1:
            genres_matched += " from around " + pub_dates[0] +"0s"
        return f"Based on what you like, I would recommend you watching movie with the genres {genres_matched} such as {recommendation[0][1]}."
        # except Exception as e:
        #     return "Sorry, I cannot recommend you a movie based on your query. The reasons might be that I do not know the movies you mentioned or there is a minor problem with the format of your input. You might want to re-check and/or rephrase your sentence. I will be waiting here. "

    def match_list_items(self, list_of_lists):
        print(list_of_lists)
        num_lists = len(list_of_lists)
        matched_items = []
        for item in list_of_lists[0]:
            matched_num = 0
            for i in range(1,num_lists):
                if item in list_of_lists[i]:
                    matched_num+=1
            if matched_num==num_lists-1:
                matched_items.append(item)
        return matched_items

    def recommend_genre_embed(self, matched_entity_url_list):
        genre = self.relation_emb[self.rel2id[self.predicates["genre"]]]
        dist_genre = np.zeros(self.entity_emb.shape[0])
        dist_movie = np.zeros(self.entity_emb.shape[0])
        idx_movie = []
        objects_movie=[]
        # add vectors according to TransE scoring function.
        for entity_url in matched_entity_url_list:
            for s, p, o in self.g.triples((URIRef(entity_url), None, None)):
                if str(o) in self.genres.values():
                    objects_movie.append(list(dict.fromkeys([str(o)])))
                    
                if str(s) in self.genres.values():
                    objects_movie.append(list(dict.fromkeys([str(s)])))

            head = self.entity_emb[self.ent2id[entity_url]]
            lhs = head + genre
            # compute distance to *any* entity
            dist_movie += pairwise_distances(head.reshape(1, -1), self.entity_emb).reshape(-1)
            idx_movie.append(pairwise_distances(head.reshape(1, -1), self.entity_emb).reshape(-1).argmin())
            dist_genre += pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)

        if len(objects_movie) == 0:
            return None, None
        
        # find most plausible entities
        matched_genres = self.match_list_items(objects_movie)
        if 'http://www.wikidata.org/entity/Q11424' in matched_genres:
            matched_genres.remove('http://www.wikidata.org/entity/Q11424')
        most_likely_movie = dist_movie.argsort()
        most_likely_genre = dist_genre.argsort()
        # compute ranks of entities
        # ranks = dist.argsort().argsort()

        most_plausible_movies = []
        for idx in most_likely_movie[:100]:
            movie_id = self.id2ent[idx]
            if movie_id not in self.url2nodes.keys():
                continue

            movie_url = self.url2nodes[movie_id]
            if movie_url in self.movies.keys():
                if idx not in idx_movie:
                    if movie_url not in [self.url2nodes[url] for url in matched_entity_url_list]:
                        most_plausible_movies.append((str(movie_id), movie_url))

        if len(matched_genres)!=0:
            most_plausible_genre = []
            for url in matched_genres:
                try:
                    most_plausible_genre.append((str(self.id2ent[url]), url))
                except KeyError:
                    continue
            if len(most_plausible_genre)==0:
                return None, None
        else:
            most_plausible_genre = [(str(self.id2ent[idx]), self.url2nodes[self.id2ent[idx]])
                for idx in most_likely_genre[:3]]
        
        return most_plausible_movies[0], most_plausible_genre
