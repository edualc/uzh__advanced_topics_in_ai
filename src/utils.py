from rdflib.namespace import Namespace, RDFS
from rdflib.term import URIRef
import rdflib
from nltk.corpus import wordnet as wn
import numpy as np
import csv
import jsonpickle
import os

WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


def load_graph():
    g = rdflib.Graph()
    g.parse('dataset/14_graph.nt', format='turtle')
    print(f"Graph loaded.")
    return g


def load_embeddings():
    entity_emb = np.load('dataset/ddis-graph-embeddings/entity_embeds.npy')
    relation_emb = np.load('dataset/ddis-graph-embeddings/relation_embeds.npy')
    print(f"Embeddings loaded.")
    return entity_emb, relation_emb


def load_dictionaries(g):
    with open('dataset/ddis-graph-embeddings/entity_ids.del', 'r') as ifile:
        ent2id = {str(rdflib.term.URIRef(ent)): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
        id2ent = {v: k for k, v in ent2id.items()}
    with open('dataset/ddis-graph-embeddings/relation_ids.del', 'r') as ifile:
        rel2id = {str(rdflib.term.URIRef(rel)): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
        id2rel = {v: k for k, v in rel2id.items()}

    ent2lbl = {str(ent): str(lbl) for ent, lbl in g.subject_objects(RDFS.label)}
    lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}
    print(f"Dictionaries loaded.")

    return ent2id, id2ent, rel2id, id2rel, ent2lbl, lbl2ent


def load_namespaces():
    WD = Namespace('http://www.wikidata.org/entity/')
    WDT = Namespace('http://www.wikidata.org/prop/direct/')
    SCHEMA = Namespace('http://schema.org/')
    DDIS = Namespace('http://ddis.ch/atai/')
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    print(f"Namespaces loaded.")
    return WD, WDT, SCHEMA, DDIS, RDFS


def load_nodes_and_preciates(g):
    def extract_nodes(g):
        nodes = {}
        query ="""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

        SELECT ?lbl WHERE {{
            <{}> rdfs:label ?lbl .
            FILTER(LANG(?lbl) = "en").
        }}
        LIMIT 1
        """

        graph_entities = set(g.subjects(unique=True)) | {s for s in g.objects(unique=True) if isinstance(s, URIRef)}
        for node in graph_entities:
            entity = node.toPython()
            if isinstance(node, URIRef):            
                qres = g.query(query.format(entity))
                for row in qres:
                    answer = row.lbl
                
                nodes[str(answer)] = entity
        return nodes

    def extract_predicates(g):
        query ="""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

        SELECT ?lbl WHERE {{
            <{}> rdfs:label ?lbl .
            FILTER(LANG(?lbl) = "en").
        }}
        LIMIT 1
        """
        predicates = {}

        graph_predicates = set(g.predicates(unique=True))
        for predicate in graph_predicates:
            predicate_ = predicate.toPython()       
            qres = g.query(query.format(predicate_))
            for row in qres:
                answer = row.lbl
            
            predicates[str(answer)] = predicate_

        return predicates

    # make variables for the nodes and predicates path
    nodes_path = 'dataset/processed/nodes.json'
    predicates_path = 'dataset/processed/predicates.json'

    # check indiviudally if the files exist and if so load them
    if os.path.exists(nodes_path):
        with open(nodes_path, 'r') as ifile:
            nodes = jsonpickle.decode(ifile.read())
    else:
        print(f"Extracting nodes from graph...")
        nodes = extract_nodes(g)
        with open(nodes_path, 'w') as ofile:
            ofile.write(jsonpickle.encode(nodes))
    print(f"Nodes loaded.")

    if os.path.exists(predicates_path):
        with open(predicates_path, 'r') as ifile:
            predicates = jsonpickle.decode(ifile.read())
    else:
        print(f"Extracting predicates from graph...")
        predicates = extract_predicates(g)
        with open(predicates_path, 'w') as ofile:
            ofile.write(jsonpickle.encode(predicates))
    print(f"Predicates loaded.")

    return nodes, predicates


def load_url2nodes(nodes, ent2lbl):
    def extract_url2nodes():
        url2nodes = dict(zip(nodes.values(), nodes.keys()))
        return ent2lbl | url2nodes

    url2nodes_path = 'dataset/processed/url2nodes.json'
    if os.path.exists(url2nodes_path):
        with open(url2nodes_path, 'r') as ifile:
            url2nodes = jsonpickle.decode(ifile.read())
    else:
        url2nodes = extract_url2nodes()
        with open(url2nodes_path, 'w') as ofile:
            ofile.write(jsonpickle.encode(url2nodes))

    return url2nodes


def load_movies(g, nodes):
    WD, WDT, _, _, _ = load_namespaces()

    def extract_movies():
        movies = {}
        for ent, url in nodes.items():
            if len([o for s,p,o in g.triples((URIRef(url), WDT.P31, WD.Q11424))])!=0:
                movies[ent] = url
        return movies

    movies_path = 'dataset/processed/movies.json'
    if os.path.exists(movies_path):
        with open(movies_path, 'r') as ifile:
            movies = jsonpickle.decode(ifile.read())
    else:
        movies = extract_movies()
        with open(movies_path, 'w') as ofile:
            ofile.write(jsonpickle.encode(movies))

    return movies


def load_genres(g, predicates, ent2lbl):
    genres = {}
    for genre_url in list(dict.fromkeys([str(o) for s,p,o in g.triples((None, URIRef(predicates["genre"]), None))])):
        if genre_url not in genres.values():
            try:
                label = ent2lbl[genre_url]
                genres[label] = genre_url
            except:
                continue
    return genres


def convert_word(input, from_pos, to_pos):    
    """ Transform words given from/to POS tags """
    words,temp_word_list=[],[]
    for index, word in enumerate(input.split(" ")):
        synsets = wn.synsets(word, pos=from_pos)

        # Word not found
        if not synsets:
            if len(words)==0:
                words.append((word,1.0))
            else:
                words =[(w+" "+word, p) for w,p in words]
        else:
            # Get all lemmas of the word (consider 'a'and 's' equivalent)
            lemmas = []
            for s in synsets:
                for l in s.lemmas():
                    if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                        lemmas += [l]

            # Get related forms
            derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]
            # filter only the desired pos (consider 'a' and 's' equivalent)
            related_noun_lemmas = []

            for drf in derivationally_related_forms:
                if from_pos == "n":
                    related_noun_lemmas += [drf[0]]
                else:
                    for l in drf[1]:
                        if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                            related_noun_lemmas += [l]

            # Extract the words from the lemmas
            temp_word_list=[l.name() for l in related_noun_lemmas]
            temp_word_list = [(w, float(temp_word_list.count(w)) / len(temp_word_list)) for w in set(temp_word_list)]

            # Take all the combinations for synonyms of different words
            # Build the result in the form of a list containing tuples (word, probability)
            if len(words)==0:
                words=temp_word_list
            else:
                words =[(w_b+" "+w_t, p_b*p_t) for w_b,p_b in words for w_t,p_t in temp_word_list]
                words.sort(key=lambda w:-w[1])

    # return all the possibilities sorted by probability
    return words
