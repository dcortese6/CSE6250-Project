import os
import pickle
import glob
import xml.etree.ElementTree as et


def get_disease_names(xml, exclude=set()):
    """Get list of diseases from standoff files"""

    disease_names = set()
    tree = et.parse(xml)

    for disease_elem in tree.iter('disease'):
        disease_name = disease_elem.attrib['name']
        if disease_name not in exclude:
            disease_names.add(disease_name)

    return sorted(list(disease_names))


class DatasetProvider():
    def __init__(self,
                corpus_path,
                annot_xml,
                disease,
                judgement,
                # use_pickled_alphabet=False,
                # alphabet_pickle=None,
                min_token_freq=0):
        """Index words by frequency in a file"""

        self.corpus_path = corpus_path
        self.annot_xml = annot_xml
        self.min_token_freq = min_token_freq
        self.disease = disease
        self.judgement = judgement
        self.alpha_pkl = 'data/lookups/alphabet.pkl'
        self.label2int = {'Y':0, 'N':1, 'Q':2, 'U':3}

        self.token2int = {}

        # alpha_pkl = 'data/lookups/alphabet.pkl'

        # when training, make alphabet and pickle it
        # when testing, load it from pickle
        # if use_pickled_alphabet:
        # if os.path.exists(alpha_pkl):
        try:
            print('reading alphabet alphabet pickle file')
            pkl = open(self.alpha_pkl, 'rb')
            self.token2int = pickle.load(pkl)
        except FileNotFoundError as e:
            print(e)
            print("Alphabet Pickle file doesn't exist. Please run preprocess file to prepare datasets.")
    
    def parse_standoff_file(self, xml, disease, task):

        doc2label = {} # key: doc id, value: label
        tree = et.parse(xml)

        for task_elem in tree.iter('diseases'):
            if task_elem.attrib['source'] == task:
                for disease_elem in task_elem:
                    if disease_elem.attrib['name'] == disease:
                        for doc_elem in disease_elem:
                            id = doc_elem.attrib['id']
                            label = doc_elem.attrib['judgment']
                            doc2label[id] = label

        return doc2label
    
    def parse_standoff(self, pattern, disease, task):

        doc2label = {} # key: doc id, value: label

        for xml_file in sorted(glob.glob(pattern)):
            print('loading annotations from', xml_file)
            d2l = self.parse_standoff_file(xml_file, disease, task)
            doc2label.update(d2l)

        return doc2label
    
    def extract_tokens(self, text, ignore_negation=False):
        """Return a file as a list of CUIs"""
    
        tokens = []
        for token in text.lower().split():
            if token.isalpha():
                tokens.append(token)
        return tokens

        # if ignore_negation:
        #     tokens = []
        #     for token in text.split():
        #         if token.startswith('n'):
        #             tokens.append(token[1:])
        #         else:
        #             tokens.append(token)
        #         return tokens

        # else:
            # return text.split()
            
    def load(self, maxlen=float('inf')):

        labels = []    # int labels
        examples = []  # examples as int sequences
        no_labels = [] # docs with no labels

        # document id -> label mapping
        doc2label = self.parse_standoff(
            self.annot_xml,
            self.disease,
            self.judgement
            )
        
        # load examples and labels
        for f in os.listdir(self.corpus_path):

            # doc_id = f.split('.')[0]
            file_path = os.path.join(self.corpus_path, f)
            # file = open(file_path).read()
            
            tree = et.parse(file_path)
            root = tree.getroot()
            for elem in root:
                for doc in elem:
                    doc_id = doc.attrib['id']
                    for txt in doc:
                        file_feat_list = self.extract_tokens(txt.text)

                        example = []
                        for token in set(file_feat_list):
                            if token in self.token2int:
                                example.append(self.token2int[token])
                            else:
                                example.append(self.token2int['oov_word'])
                                
                        if len(example) > maxlen:
                            example = example[0:maxlen]

                    # no labels for some documents for some reason
                    if doc_id in doc2label:
                        string_label = doc2label[doc_id]
                        int_label = self.label2int[string_label]
                        labels.append(int_label)
                        examples.append(example)
                    else:
                        no_labels.append(doc_id)

        print('%d documents with labels for %s/%s in %s' \
        % (len(labels), self.disease,
            self.judgement, self.annot_xml.split('/')[-1]))

        print('%d documents with no labels for %s/%s in %s' \
        % (len(no_labels), self.disease,
            self.judgement, self.annot_xml.split('/')[-1]))
        
        return examples, labels
