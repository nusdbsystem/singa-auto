#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import pickle
import base64
import numpy as np
import argparse
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from singa_auto.model import BaseModel, utils
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class
from singa_auto.datasets.image_classification_dataset import ImageDataset4Clf
from io import BytesIO



class MLPFoodRecommendationModel(BaseModel):

    '''
    This class defines a MLP food recommendation model with knowledge graphs.
    '''

    @staticmethod
    def get_knob_config():
        return {}

    def __init__(self, **knobs):

        self._knobs = knobs
        self.__dict__.update(knobs)

        self.KB = dict()
        self.word_list = dict()
        self.depth = 2
        self.width = 4
        self.degree = 64
        self.denominator = (self.degree + 1) * self.width * self.depth 
        self.num_sequence_samples = 32
        self.num_encoding_samples = 1
        self.stride = 2
        self.decode_num_feat = self.num_encoding_samples * self.num_sequence_samples
 

    def add_triple_pair(self, h, p, t):
        if not(h in self.KB):
            self.KB[h] = dict()
        if not(p in self.KB[h]):
            self.KB[h][p] = dict()
        self.KB[h][p][t] = -1
    
        if not(t in self.KB):
            self.KB[t] = dict()
        q = "re-" + p
        if not(q in self.KB[t]):
            self.KB[t][q] = dict()
        self.KB[t][q][h] = -1
        return 0
    
    def compute_triple_weights(self):
        print("Computing triple weights...")
        for h in self.KB:
            for p in self.KB[h]:
                if len(self.KB[h][p]) == 0:
                    continue
                if not("re-") in p:
                    q = "re-" + p
                else:
                    q = p.replace("re-", "")
                for t in self.KB[h][p]:
                    imp = 2 / (len(self.KB[h][p]) + len(self.KB[t][q]))
                    self.KB[h][p][t] = imp
        return 0
    
    def read_knowledge_graph(self, fpath):
        print("Reading the knowledge graph from %s.\n"%fpath)
        f = open(fpath, "r")
        n = 0
    
        for x in f.readlines():
            y = x.replace("\n","")
            y = y.split(" ")
            self.add_triple_pair(y[0], y[1], y[2])
            for w in y + ["re-" + y[1]]:
                if not(w in self.word_list):
                    self.word_list[w] = n
                    n = n + 1
        f.close()
        print("Done.")
        return self.word_list
    
    def sequences_to_fractional_dense_encodings(self, seq_list, word_list, denominator = 1000, stride = 1, num_samples = 2):
        m = len(seq_list)
        n = 0
        for w in word_list:
            if word_list[w] > n:
                n = word_list[w]
        n = n + 1
        encodings = np.zeros((m, num_samples, n), dtype = float)
        for i in range(m):
            seq = seq_list[i]
            word_positions = dict()
            for j in range(len(seq)):
                w = seq[j]
                jt = int(j / stride)
                if not(w in word_positions):
                    word_positions[w] = [jt]
                else:
                    word_positions[w].append(jt)
            for k in range(num_samples):
                for w in word_positions:
                    if not(w in word_list):
                        if not(w in ["Root_Entity", "Root_Predicate"]):
                            print("Warning: %s is not in the word list."%w)
                        continue
                    pos = random.choice(word_positions[w])
                    idx = word_list[w]
                    if pos < denominator:
                        frac = (denominator - pos) * 1.0 / denominator
                    else:
                        frac = 1.0 / denominator
                    encodings[i,k,idx] = frac
        return encodings
    
    
            
    def annotate_encodings_on_sequence(self, seq, enc, word_list):
        res = []
        for w in seq:
            if w in word_list:
                idx = word_list[w]
                res.append([w, enc[idx]])
            else:
                res.append([w, -1])
        return res
    
    
    def knowledge_base_weight_normalisation(self):
        for h in self.KB:
            imp_sum = 0
            for p in self.KB[h]:
                for t in self.KB[h][p]:
                    self.KB[h][p][t] = max(self.KB[h][p][t], 1 / 10) # Add boundary. This is the best.
                    #KB[h][p][t] = 1 # Disable weighting. Seems to be better.
                    #KB[h][p][t] = 1.0 / KB[h][p][t] # Reverse weighting.
                    imp_sum = imp_sum + self.KB[h][p][t]
            if imp_sum == 0:
                continue
            for p in self.KB[h]:
                for t in self.KB[h][p]:
                    self.KB[h][p][t] = self.KB[h][p][t] / imp_sum
        return 0
    
    def sample_triples_from_knowledge_base(self, entity, n):
        result = []
        h = entity
    
        v = np.sort(np.random.rand(n))
    
        i = 0
        v1 = 0
        for p in self.KB[h]:
            for t in self.KB[h][p]:
                if i == n:
                    break
    
                v2 = v1 + self.KB[h][p][t]
                
                while v[i] > v1 and v[i] <= v2:
                    result.append([h, p, t])
                    i = i + 1
                    
                    if i == n:
                        break
                v1 = v2
    
        return result
     
    def sample_sequence_from_knowledge_base(self, entity, depth, width, degree):
    
        if depth == 0:
            return []
    
        seq = [entity]
    
        h = entity
    
        # expl - explanation
        expl = self.sample_triples_from_knowledge_base(h, degree)
        for x in expl:
            seq.append(x[1])
            seq.append(x[2])
    
        triples = self.sample_triples_from_knowledge_base(h, width)
        for hs in triples:
            seq.extend(["[Related_Entity]", hs[1], hs[2]])
            expl = self.sample_triples_from_knowledge_base(hs[2], degree)
            for x in expl:
                seq.append(x[1])
                seq.append(x[2])
    
        next_triple = self.sample_triples_from_knowledge_base(h, 1)[0]
        next_predicate = next_triple[1]
        next_entity = next_triple[2]
        next_sequence = self.sample_sequence_from_knowledge_base(next_entity, depth - 1, width, degree)
        if next_sequence != []:
            seq.extend(["[Next_Entity]", next_predicate])
            seq.extend(next_sequence)

        return seq          
   
    def read_dataset(self, fpath):
        f = open(fpath, "r")
        feat = []
        tgt = []
        
        for x in f.readlines():
            y = x.split(" ")
    
            root_entity = y[0]
            
            seq_list = []
            for k in range(self.num_sequence_samples):
                seq = ["[Root_Entity]", "Root_Predicate"] + self.sample_sequence_from_knowledge_base(root_entity, self.depth, self.width, self.degree)
                seq_conv = []
                for z in seq:
                    if z in ["[Root_Entity]", "[Related_Entity]", "[Next_Entity]"]:
                        continue
                    if z == root_entity:
                        seq_conv.append("Root_Entity")
                    else:
                        seq_conv.append(z)
                seq_list.append(seq_conv)
    
            encodings = self.sequences_to_fractional_dense_encodings(seq_list = seq_list, word_list = self.word_list, denominator = self.denominator, stride = self.stride, num_samples = self.num_encoding_samples)
            for i in range(encodings.shape[0]):
                for j in range(encodings.shape[1]):
                    feat.append(encodings[i][j])
                    tgt.append(int(y[1]))
                    #input(y[0])
                    #input(feat[-1])
                    #input(tgt[-1])
        #feat = np.array(feat)
        #tgt = np.array(tgt) 
        return feat, tgt

   
    def convert_entity_to_features(self, entity):
        
        if not(entity in self.KB):
            return "None"
        else:
            feat = []

            root_entity = entity + "" 
            
            seq_list = []
            for k in range(self.num_sequence_samples):
                seq = ["[Root_Entity]", "Root_Predicate"] + self.sample_sequence_from_knowledge_base(root_entity, self.depth, self.width, self.degree)
                seq_conv = []
                for z in seq:
                    if z in ["[Root_Entity]", "[Related_Entity]", "[Next_Entity]"]:
                        continue
                    if z == root_entity:
                        seq_conv.append("Root_Entity")
                    else:
                        seq_conv.append(z)
                seq_list.append(seq_conv)
    
            encodings = self.sequences_to_fractional_dense_encodings(seq_list = seq_list, word_list = self.word_list, denominator = self.denominator, stride = self.stride, num_samples = self.num_encoding_samples)
            for i in range(encodings.shape[0]):
                for j in range(encodings.shape[1]):
                    feat.append(encodings[i][j])
            return feat
    
    
    def train(self, dataset_path, work_dir,  **kwargs):

        os.system("tar -xvf %s -C %s"%(dataset_path, work_dir))

        print("Initialising classifiers...")
        self.tag_list = []
        f = open("%s/training_data/tag_list.txt"%work_dir, "r")
        for x in f.readlines():
            x = x.replace("\n","")
            if x != "":
                self.tag_list.append(x)

        self.clf = dict()
        for tag in self.tag_list:
            self.clf[tag] = MLPClassifier(random_state = 1, max_iter = 100, solver = "lbfgs", hidden_layer_sizes = (128, 128, 128, 128))
            # can use random forests if MLP is too slow.
            #self.clf[tag] = RandomForestClassifier(n_estimators = 10, random_state=0)


        print("Reading knowledge base...")
        kb_path = "%s/training_data/food_knowledge_base.tri"%work_dir
        self.read_knowledge_graph(kb_path)
        self.compute_triple_weights()
        self.knowledge_base_weight_normalisation()

        acc = 0
        n = 0
        for tag in self.tag_list:
            print("Training a classifier for Tag \"%s\"."%tag)
            tr_fpath = "%s/training_data/%s_training.txt"%(work_dir, tag)
            tr_feat, tr_tgt = self.read_dataset(tr_fpath)
            #tr_feat = tr_feat[0:100]
            #tr_tgt = tr_tgt[0:100]
            self.clf[tag].fit(tr_feat, tr_tgt)
            s = self.clf[tag].score(tr_feat, tr_tgt)
            print("The training accuracy on Tag \"%s\" is %lf."%(tag, s))
            #self.clf[tag].fit(tr_feat, tr_tgt)
            acc = acc + s * len(tr_tgt)
            n = n + len(tr_tgt)
        acc = acc / n
        utils.logger.log('Train accuracy: {}'.format(acc))

     
    def evaluate(self, dataset_path, work_dir,  **kwargs):

        os.system("tar -xvf %s -C %s"%(dataset_path, work_dir))

        acc = 0
        n = 0
        for tag in self.tag_list:
            print("Evaluating classifier for Tag \"%s\"."%tag)
            ev_fpath = "%s/evaluation_data/%s_evaluation.txt"%(work_dir, tag)
            ev_feat, ev_tgt = self.read_dataset(ev_fpath)
            s = self.clf[tag].score(ev_feat, ev_tgt)
            print("The accuracy on Tag \"%s\" is %lf."%(tag, s))
            #self.clf[tag].fit(tr_feat, tr_tgt)
            acc = acc + s * len(ev_tgt)
            n = n + len(ev_tgt)
        acc = acc / n
        return acc


    def predict(self, queries, work_dir = None):

        result = []
        for query in queries:
            x = eval(query)
            entity = x[0]
            tag = x[1]

            feat = self.convert_entity_to_features(entity)
            if feat == "None":
                result.append("Food %s does not exist in the knowledge base."%(entity))
            elif not(tag in self.clf):
                result.append("Tag %s does not exist in the classifier."%(tag))
            else:
                proba = self.clf[tag].predict_proba(feat)
                pred = np.zeros(proba.shape[1])

                for j in range(proba.shape[0]):
                    pred = pred + proba[j]
                pred = pred / proba.shape[0]
                result.append(str(pred.tolist()))
        return result

    def dump_parameters(self):
        params = pickle.dumps(self.__dict__)
        return params

    def load_parameters(self, params):
        self.__dict__ = pickle.loads(params)

    def print_knowledge_graph(self):
        # print the knowledge graph.
        for h in self.KB:
            for p in self.KB[h]:
                for t in self.KB[h][p]:
                    input([h,p,t,self.KB[h][p][t]])
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='data/food_nbnc_training_data.tar',
                        help='Path to training dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/food_nbnc_evaluation_data.tar',
                        help='Path to validation dataset')
    #parser.add_argument('--test_path',
    #                    type=str,
    #                    default='data/food_nbnc_evaluation_data.tar',
    #                    help='Path to test dataset')

    (args, _) = parser.parse_known_args()

    queries = [str(["海菜", "puerpera_tag"]), str(["鱼肉", "pregnant_tag"]), str(["Mars", "pregnant_tag"])]

    test_model_class(model_file_path=__file__,
                     model_class='MLPFoodRecommendationModel',
                     task='GENERAL_TASK',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     #test_dataset_path=args.test_path,
                     queries=queries)

