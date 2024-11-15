from rdflib.term import Literal
import spacy
from spacy import displacy

from .utils import *


class FactualBotPartOfSpeech:
    def __init__(self, g, nodes, predicates):
        self.g = g
        self.nodes = nodes
        self.predicates = predicates

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("spacy_wordnet", after='tagger')

    def get_factual__pos(self, question: str):
        if 'who' in question.lower().split(' '):
            question_type = 'who'
        elif 'when' in question.lower().split(' '):
            question_type = 'when'
        elif 'what' in question.lower().split(' '):
            question_type = 'what'
        else:
            question_type = 'unknown'

        # get all possible entities from parsing the question
        # ============================================================
        #
        parsed_dict = self._parse_question(question)
        # print(parsed_dict)

        # look up any possible match in the predicates/nodes
        # ============================================================
        #
        def lookup_item(label, nodes, predicates):
            matches = []
            if label in nodes.keys():
                matches.append(nodes[label])
            if label in predicates.keys():
                matches.append(predicates[label])
            return matches

        for entity in parsed_dict.keys():
            # in case of verbs, we want to check for synonyms, e.g.
            # "played" -> "actor"
            #
            if parsed_dict[entity]['type'] == 'VERB':
                # check synonyms based on noun form of the verb
                noun_forms = convert_word(entity, WN_VERB, WN_NOUN)

                if question_type == 'when':
                    noun_forms.extend([(f"{noun[0]} date",0) for noun in noun_forms])
                    # print(noun_forms)

                candidate_synonyms = list(filter(lambda x: x in self.nodes.keys(), [x[0] for x in noun_forms]))
                candidate_synonyms.append(entity)

                tmp = []
                for candidate in candidate_synonyms:
                    if candidate == 'star':
                        tmp.extend(lookup_item('cast member', self.nodes, self.predicates))
                    tmp.extend(lookup_item(candidate, self.nodes, self.predicates))
                    
                parsed_dict[entity]['matches'] = tmp

            else:
                parsed_dict[entity]['matches'].extend(lookup_item(entity, self.nodes, self.predicates))

            parsed_dict[entity]['matches'] = list(set(parsed_dict[entity]['matches']))
        # print(parsed_dict)

        # build query based on question word
        # ============================================================
        #
        possible_predicates = set()
        possible_items = set()

        for phrase in parsed_dict.keys():
            for match in parsed_dict[phrase]['matches']:
                identifier = match.split('/')[-1]

                if identifier.startswith('P'):
                    possible_predicates.add(f"http://www.wikidata.org/prop/direct/{identifier}")
                elif identifier.startswith('Q'):
                    possible_items.add(f"http://www.wikidata.org/entity/{identifier}")
        # print(f"Identified Items: {possible_items}")
        # print(f"Identified Predicates: {possible_predicates}")
        
        # Build possible queries
        # ============================================================
        #
        if question_type == 'who':
            queries = []
            for item in possible_items:
                for predicate in possible_predicates:
                    queries.append(self._who_query(item, predicate))

        elif question_type == 'when':
            queries = []
            for item in possible_items:
                for predicate in possible_predicates:
                    queries.append(self._when_query(item, predicate))

        elif question_type == 'what':
            queries = []
            for item in possible_items:
                for predicate in possible_predicates:
                    queries.append(self._what_query(item, predicate))
                    queries.append(self._what_query__with_label(item, predicate))

        # else:
            # print('UNKNOWN QUESTION TYPE')


        # Execute queries
        # ============================================================
        #
        query_answered = False
        for query in queries:
            # print(query)
            res = self.g.query(query)
            
            if len(res) == 0:
                continue

            for row in res:
                result = row.query
                
                if question_type in ['when', 'what']:
                    if not isinstance(result, Literal):
                        continue

                # print(type(result))
                if result is not None:
                    query_answered = True
                    return f"According to the graph, I think the answer is {result}."

        if not query_answered:
            return "Could not find answer in graph"

    def print_tree(self, question):
        doc = self.nlp(question)
        displacy.render(doc, style='dep', jupyter=True)

    def _label_query(self, item_iri):
        return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?lbl WHERE {{
            <{}> rdfs:label ?lbl .
            FILTER(LANG(?lbl) = "en").
        }}
        LIMIT 1
        """.format(item_iri)

    def _who_query(self, item_iri, predicate_iri):
        return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?query WHERE {{
            <{}> <{}> ?person .
            ?person rdfs:label ?query .
        }}
        LIMIT 1
        """.format(item_iri, predicate_iri)

    def _when_query(self, item_iri, predicate_iri):
        return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?query WHERE {{
            <{}> <{}> ?query .
        }}
        LIMIT 1
        """.format(item_iri, predicate_iri)

    def _what_query(self, item_iri, predicate_iri):
        return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?query WHERE {{
            <{}> <{}> ?query .
        }}
        LIMIT 1
        """.format(item_iri, predicate_iri)

    def _what_query__with_label(self, item_iri, predicate_iri):
        return """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?query WHERE {{
            <{}> <{}> ?item .
            ?item rdfs:label ?query .
            FILTER(LANG(?query) = "en").
        }}
        LIMIT 1
        """.format(item_iri, predicate_iri)

    def _parse_question(self, question: str):
        doc = self.nlp(question)
        sent = list(doc.sents)[0]

        root_type = sent.root.pos_
        # print(f"Root Type: {root_type}")

        entity_nodes = []
        preps_to_split = []


        if root_type == 'AUX':
            for child in sent.root.children:
                if child.dep_ == 'nsubj':
                    entity_nodes.append(child)
                    entity_nodes, preps_to_split = self._check_for_child_prep_pobj(child, entity_nodes, preps_to_split)


        elif root_type == 'VERB':
            for child in sent.root.children:
                if child.dep_ == 'dobj':
                    entity_nodes.append(child)
                    entity_nodes, preps_to_split = self._check_for_child_prep_pobj(child, entity_nodes, preps_to_split)

                elif child.dep_ == 'prep':
                    preps_to_split.append(child)

                    for subchild in child.children:
                        if subchild.dep_ == 'pobj':
                            entity_nodes.append(subchild)
                            entity_nodes, preps_to_split = self._check_for_child_prep_pobj(subchild, entity_nodes, preps_to_split)

                elif child.dep_ == 'nsubjpass':
                    entity_nodes.append(child)
                    entity_nodes, preps_to_split = self._check_for_child_prep_pobj(child, entity_nodes, preps_to_split)
                
                elif child.dep_ == 'nsubj':
                    if child.pos_ != 'PRON':
                        entity_nodes.append(child)
                        entity_nodes, preps_to_split = self._check_for_child_prep_pobj(child, entity_nodes, preps_to_split)


        # print(entity_nodes)
        entities = dict()
        if root_type == 'VERB':
            entities[sent.root.text] = { 'type': 'VERB', 'matches': [] }
        for node in entity_nodes:
            phrase = self._build_entity_phrase(node, preps_to_split)
            entities[phrase] = { 'type': None, 'matches': [] }
                
        return entities

    def _build_entity_phrase__rec(self, entity, preps_to_split, entities=list()):
        if entity not in entities:
            entities.append(entity)
        for child in entity.children:
            if child.dep_ == 'prep' and child in preps_to_split:
                continue
            entities = self._build_entity_phrase__rec(child, preps_to_split, entities)
        return entities

    def _build_entity_phrase(self, entity, preps_to_split):
        entities = self._build_entity_phrase__rec(entity, preps_to_split)

        tmp = []
        # print(list(entity.subtree))
        for token in entity.subtree:
            if token in entities:
                tmp.append(token)
        
        if tmp[0].text == 'the':
            tmp = tmp[1:]

        return ' '.join([token.text for token in tmp]).replace(' :', ':')

    def _check_for_child_prep_pobj(self, child, entity_nodes, preps_to_split):
        for subchild in child.children:
            if subchild.dep_ == 'prep':
                preps_to_split.append(subchild)

                for subsubchild in subchild.children:
                    if subsubchild.dep_ == 'pobj':
                        entity_nodes.append(subsubchild)
                        entity_nodes, preps_to_split = self._check_for_child_prep_pobj(subsubchild, entity_nodes, preps_to_split)
        return entity_nodes, preps_to_split

