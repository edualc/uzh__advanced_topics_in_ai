from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
from thefuzz import fuzz,process
from torch import nn
from nltk.corpus import wordnet as wn
import editdistance
from rdflib.term import URIRef
from sklearn.metrics import pairwise_distances

from .utils import *


class FactualBotNER(nn.Module):
    def __init__(self, g, nodes,url2nodes, predicates, entity_emb, relation_emb, ent2id, rel2id, ent2lbl, lbl2ent, id2ent,crowd_data):
        super().__init__()
        self.setup_ner()

        self.factual_question_patterns = [
            "who is the (.*) of ENTITY",
            "who was the (.*) of ENTITY",
            "who was the (.*) for ENTITY",
            "who was the (.*) in ENTITY",
            "what is the (.*) of ENTITY",
            "who (.*) ENTITY",
            # "who (.*) in ENTITY",
            "who wrote the (.*) of ENTITY",
            "who wrote the (.*) for ENTITY",
            "when was ENTITY (.*)",
            # "when did ENTITY (.*)",
            "where was ENTITY (.*)",
            "where is ENTITY (.*)",
            "can you tell me the (.*) of ENTITY"
        ]
        self.g = g
        self.nodes = nodes
        self.url2nodes = url2nodes
        self.predicates = predicates
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.ent2lbl = ent2lbl
        self.lbl2ent = lbl2ent
        self.id2ent = id2ent
        self.crowd_data = crowd_data
    
    def setup_ner(self):
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.ner = pipeline("ner", model=model, tokenizer=tokenizer)

    def entity_extraction(self,ner_results,example):
        entity = ""
        entity_list = []
        reset=0
        for entity_num in range(len(ner_results)):
            if (ner_results[entity_num]["word"].find("#") ==-1) & (reset!=0):
                entity = entity + " " + ner_results[entity_num]["word"]
                reset +=1
            else:
                entity = entity + ner_results[entity_num]["word"].replace("#","")
                reset +=1
            
            if (entity_num < len(ner_results)-1):
                if (ner_results[entity_num+1]["start"] - ner_results[entity_num]["end"] > 3):
                    entity_list.append(entity)
                    reset=0
                    entity = ""
                    continue
            else:
                entity_list.append(entity)
                reset=0
                continue
        given_entity_flawed = entity_list
        for n,entity in enumerate(entity_list):
            entity = entity.replace("?","").strip()
            if len(entity.split(" "))>1:
                try:
                    first_word = entity.split(" ")[0]
                    last_word = entity.split(" ")[-1]
                    search_str = first_word + "(.+?)" + last_word
                    entity_list[n] = re.search(search_str,example).group(0)
                except:
                    ent=""
                    for w in entity.split(" "):
                        ent+= w + " "
                    entity_list[n] = ent
            else:
                continue
        return entity_list,given_entity_flawed
    
    def preprocessing_before_ner(self,question):
        try:
            question_new=re.sub(re.search("(.*?)of",question).group(0), re.search("(.*?)of",question).group(0).lower() ,question)
        except:
            words_question = question.split(" ")
            words_question[0] =words_question[0].lower()
            question_new = ""
            for word in words_question:
                question_new += word + " "
        return question_new
    
    def preprocessing_before_patterndetection(self,question):
        tmp_words = [" in ", " for "," on ", " of "]
        for tmp in tmp_words:
            if tmp in question:
                question=question.replace(tmp," ")
        if "the movie" in question:
            question=question.replace("the movie ","")
        return question
    
    def preprocessing(self,question):
        tmp_words = [" in ", " for "," on ", " of "]
        tmp_verbs = ["was","were","is","are","did","do","does","have","has"]
        for tmp in tmp_words:
            if tmp in question:
                question=question.replace(tmp," ")
        if "the movie" in question:
            question=question.replace("the movie ","")
        if "the" in question:
            index_the = [idx for idx,word in enumerate(question.split(" ")) if word=="the"]
            words_between = ""
            for i in range(1,index_the[0]):
                words_between += question.split(" ")[i] + " "
            question=question.replace(words_between.strip(),"...")
        
        if "when" in question.lower():
            if question.split(" ")[1] in tmp_verbs:
                question=question.replace(question.split(" ")[1],"...")
        return question.replace("?","").lower()

    # which pattern is used in the given question?
    def pattern_detection(self,ner_results,example):
        entities_extracted,given_entity_flawed = self.entity_extraction(ner_results,example)
        matched_entity,_= self.match_things(self.nodes, entities_extracted[0])
        pattern_and_entity = [[re.sub("ENTITY",matched_entity, pattern),matched_entity] for pattern in self.factual_question_patterns]
        example_updated = re.sub(given_entity_flawed[0].replace("?","").strip(),matched_entity, example)
        pattern_entity_included = [lists[0] for lists in pattern_and_entity]
        entity_from_pattern_and_entity = list(dict.fromkeys([lists[1] for lists in pattern_and_entity]))


        question_pattern = process.extract(self.preprocessing_before_patterndetection(example_updated),pattern_entity_included,scorer=fuzz.ratio)[0][0]
        question_pattern_ = [re.sub(value,"ENTITY",question_pattern) for value in entity_from_pattern_and_entity if question_pattern.find(value)!=-1][0]

        index = [num for num,value in enumerate(self.factual_question_patterns) if value==question_pattern_][0]

        return question_pattern,index,example_updated

    def relation_extraction(self,ner_results,example):
        question_pattern, index,example_updated = self.pattern_detection(ner_results,example)

        relation = re.match(self.preprocessing(question_pattern), self.preprocessing(example_updated)).group(1)
        if len(relation.split(" "))==1 and (wn.synsets(relation)[0].pos() == WN_VERB):
            relations = [synonym for synonym,score in convert_word(relation, WN_VERB, WN_NOUN)]
            if "play" in relation or "star" in relation:
                relations.append("cast member")
            return self.match_relations(relations,relation,ner_results,example)
        else:
            if ("when" in example.lower()) and ("premiere" in relation):
                relation = "publication date"
            return relation # take care of directed, released, etc. cases
    
    def match_things(self,dict, input,entity_predicates=None):
        tmp = 9999
        match_key = ""
        match_value = ""
        for key, value in dict.items():
            if editdistance.eval(key.lower(), input) < tmp:
                tmp = editdistance.eval(key.lower(), input)
                match_key = key
                match_value = value
        
        if entity_predicates is not None:
            tmpp = np.inf
            match_relation_key = ""
            match_relation_value= ""
            for key in entity_predicates:
                if editdistance.eval(key.lower(), input) < tmpp:
                    tmpp = editdistance.eval(key.lower(), input)
                    match_relation_key = key
                    match_relation_value = self.predicates[key]
            if editdistance.eval(match_key, match_relation_key)<=0.1*(len(match_key)+len(match_relation_key)):
                match_key = match_relation_key
                match_value = match_relation_value

        return match_key,match_value

    def match_relations(self, inputs,relation,ner_results,example):
        tmp = 9999
        entities,_ = self.entity_extraction(ner_results,example)
        matched_entity, matched_entity_url= self.match_things(self.nodes, entities[0])
        entity_predicates = list( dict.fromkeys([k for s,p,o in self.g.triples((URIRef(matched_entity_url), None, None)) for k,v in self.predicates.items() if v==str(p)]) )

        match_key = ""
        for input in inputs:
            if input in entity_predicates:
                match_key=input
                break
            if editdistance.eval(relation.lower(), input) < tmp:
                tmp = editdistance.eval(relation.lower(), input)
                match_key = input
        return match_key
    
    def find_entities_with_same_name(self,matched_entity):
        list_matched = []
        for key,value in self.url2nodes.items():
            if value == matched_entity:
                list_matched.append((key,value))
        return list_matched
        
    def final_query(self,matched_entity,matched_entity_url,matched_predicate,matched_predicate_url):
        query_option1 ="""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

        SELECT ?lbl WHERE {{
            <{}> <{}> ?answer.
            ?answer rdfs:label ?lbl .
            FILTER(LANG(?lbl) = "en").
        }}
        LIMIT 1
        """

        query_option2 ="""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

        SELECT ?lbl WHERE {{
            ?answer <{}> <{}>.
            ?answer rdfs:label ?lbl .
            FILTER(LANG(?lbl) = "en").
        }}
        LIMIT 1
        """

        crowd_answer = self.crowd_source(matched_entity_url,matched_predicate_url)
        if crowd_answer != "":
            return crowd_answer

        list_matched = self.find_entities_with_same_name(matched_entity)
        for key,value in list_matched:
            qres1 = self.g.query(query_option1.format(key,matched_predicate_url))
            qres2 = self.g.query(query_option2.format(matched_predicate_url,key))
            answer = ""
            try:
                for row in qres1:
                    answer = row.lbl
            except answer == "":
                for row in qres2:
                    answer = row.lbl 
            if answer!="":
                break

        if answer == "":
            try:
                answer1, answer2, answer3 = self.final_embed(matched_entity_url,matched_predicate_url)    
                return f"According to the embeddings, the {matched_predicate} of {matched_entity} is {answer1}, {answer2}, {answer3}." 
            except:  
                return "Sorry, I could not find the answer. Can you please rephrase the question?"
        else:
            # answer1, answer2, answer3 = self.final_embed(matched_entity_url,matched_predicate_url)    
            return f"""According to the the graph, the {matched_predicate} of {matched_entity} is {answer}."""
    
    def uriref_ent_extractor(self,url):
        entity = url.split("/")[-1]
        if "entity" in url:
            entity = "wd:"+entity
        elif "prop" in url:
            entity = "wdt:"+entity
        elif "ddis" in url:
            entity = "ddis:"+entity
        elif "schema" in url:
            entity = "schema:"+entity
        elif "rdf-schema" in url:
            entity = "rdfs:"+entity
        return entity
    
    def crowd_source(self,matched_entity_url,matched_predicate_url):
        entity = self.uriref_ent_extractor(matched_entity_url)
        predicate = self.uriref_ent_extractor(matched_predicate_url)
        answer = ""
        if entity in self.crowd_data["Input1ID"].values and predicate in self.crowd_data["Input2ID"].values:
            votes = self.crowd_data[(self.crowd_data["Input1ID"]==entity) & (self.crowd_data["Input2ID"]== predicate)]
            try:
                support_votes = votes[votes["AnswerLabel"]=="CORRECT"]["Votes"].values[0]
            except:
                support_votes=0
            try:
                reject_votes = votes[votes["AnswerLabel"]=="INCORRECT"]["Votes"].values[0]
            except:
                reject_votes=0
            kappa = votes["kappa"].values[0]
            result = votes["Input3ID"].values[0]
            if "wd:" in result:
                result = 'http://www.wikidata.org/entity/'+result.split(":")[-1]
                result = self.url2nodes[result]

            entity = self.url2nodes[matched_entity_url]
            predicate = self.url2nodes[matched_predicate_url]

            answer = f"""The {predicate} of {entity} is {result}.\n[Crowd, inter-rater agreement {kappa:.4f}, The answer distribution for this specific was {support_votes} support votes, {reject_votes} reject votes]"""
        
        return answer
    
    def final_embed(self,matched_entity_url,matched_predicate_url):
        head = self.entity_emb[self.ent2id[matched_entity_url]]
        pred = self.relation_emb[self.rel2id[matched_predicate_url]]
        # add vectors according to TransE scoring function.
        lhs = head + pred
        # compute distance to *any* entity
        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        # find most plausible entities
        most_likely = dist.argsort()
        # compute ranks of entities
        ranks = dist.argsort().argsort()

        most_plausible_3_answers = [(str(self.id2ent[idx]), self.ent2lbl[self.id2ent[idx]])
            for rank, idx in enumerate(most_likely[:3])]
        
        answer1, answer2, answer3 = most_plausible_3_answers[0][1],most_plausible_3_answers[1][1],most_plausible_3_answers[2][1]
        return answer1, answer2, answer3

    def _ner_results(self, input):
        return self.ner(self.preprocessing_before_ner(input))

    def forward(self,input):
        ner_results = self._ner_results(input)
        entities,_ = self.entity_extraction(ner_results,input)
        entity = entities[0]
        relation = self.relation_extraction(ner_results,input)
        matched_entity, matched_entity_url= self.match_things(self.nodes, entity)
        entity_predicates = list( dict.fromkeys([k for s,p,o in self.g.triples((URIRef(self.nodes[matched_entity]), None, None)) for k,v in self.predicates.items() if v==str(p)]) )
        matched_predicate, matched_predicate_url= self.match_things(self.predicates,relation,entity_predicates)

        output = self.final_query(matched_entity,matched_entity_url,matched_predicate,matched_predicate_url)
        return output
            
    def get_entities(self, input):
        ner_results = self._ner_results(input)

        entity = ""
        entity_list = []
        reset=0
        for entity_num in range(len(ner_results)):
            if (ner_results[entity_num]["word"].find("#") ==-1) & (reset!=0):
                entity = entity + " " + ner_results[entity_num]["word"]
                reset +=1
            else:
                entity = entity + ner_results[entity_num]["word"].replace("#","")
                reset +=1
            
            if (entity_num < len(ner_results)-1):
                if (ner_results[entity_num+1]["start"] - ner_results[entity_num]["end"] > 3):
                    entity_list.append(entity)
                    reset=0
                    entity = ""
                    continue
            else:
                entity_list.append(entity)
                reset=0
                continue
        for n,entity in enumerate(entity_list):
            entity = entity.replace("?","").strip()
            if len(entity.split(" "))>1:
                try:
                    first_word = entity.split(" ")[0]
                    last_word = entity.split(" ")[-1]
                    search_str = first_word + "(.+?)" + last_word
                    entity_list[n] = re.search(search_str, input).group(0)
                except:
                    ent=""
                    for w in entity.split(" "):
                        ent+= w + " "
                    entity_list[n] = ent
            else:
                continue
        temp = entity_list
        idx = []
        for i,ent in enumerate(entity_list):
            if "," in ent:
                idx.append(i)
                if ",and" in ent or ", and" in ent:
                    temp[i] =ent.replace(",and",",").replace(", and",",")
                temp=temp + temp[i].split(",")
        for i in sorted(idx,reverse=True):
            temp.pop(i)
        entity_list = list(dict.fromkeys([ent.strip() for ent in temp]))
        return entity_list