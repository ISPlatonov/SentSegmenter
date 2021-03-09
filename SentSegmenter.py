from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
    )

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger( emb)

names_extractor = NamesExtractor(morph_vocab)

text = '''
Wintershall Dea надеется, что "Северный поток — 2" будет введен в эксплуатацию как можно быстрее, он необходим Европе, сказал член правления Wintershall Dea, ответственный за деятельность компании в России, Латинской Америке, а также за газотранспортные проекты Тило Виланд
'''

class SentSegmenter:
        
    def __init__(self, text):
        self.doc = Doc(text)
        self.tokens_list = []
        self.tokenize()
        self.sents = []
        self.sents2jsons()

    def tokenize(self):
        self.doc.segment(segmenter)
        self.doc.tag_morph(morph_tagger)
        self.doc.tag_ner(ner_tagger)
        self.doc.parse_syntax(syntax_parser)
        self.doc2sents()

    # returns the only tokens
    # that have needed rel
    # and are not used before
    def rel_word_tokens(self, sent, rel=('root'), used_ids=[]):
        res = []
        for word in sent:
            #print(  'word =', word)
            if word.rel.split(':')[0] in rel and word.id not in used_ids:
                res.append(word)
        return res

    # list of lists to flat list
    def flat_list(self, ll):
        return [item for list in ll for item in list]

    # returns a text list by the tokens
    def tokens2txt(self, tl):
        wl = [w.text for w in tl]
        res = ' '.join(wl)
        return res

    # this is not exactly what we're needed for
    # it should take in a sent with the subject
    # remember: subj --commits--> obj
    def sentence_division(self, sent, rel=('root'), used_ids=[], return_used_ids=False):

        rel_words = self.rel_word_tokens(sent, rel, used_ids)
        rel_ids = [w.id for w in rel_words]
        #print('rel words:', [w.text for w in rel_words])
        root_words = self.rel_word_tokens(sent, 'root')
        #print(root_ids)

        rel_ids = list(set(rel_ids) - set(used_ids))
        sent_parts = [[w] for w in rel_words]
        sent_parts_ids = [[w.id] for w in rel_words]
        root_ids = [w.id for w in root_words]
        if rel == ('root'):
            root_ids = [w.head_id for w in root_words]
        #root_ids.extend(used_ids)

        #print('  rel ids =', rel_ids)
        #print('  root ids =', root_ids)
        #print('  used ids =', used_ids)
        
        for s in sent:
            branch = []
            branch_ids = []
            if s.id in rel_ids or s.id in used_ids:
                #print('  skip:', s.id)
                continue
            si = s
            #print('si.id =', si.id, si.id not in root_ids)
            while si.id not in rel_ids and si.id not in root_ids and si.id not in used_ids:#(not (True in [si.id in spii for spii in sent_parts_ids])):# and (si.head_id not in root_ids):
                #print(' si.id = ', si.id, ' ', si.id not in root_ids)
                branch.append(si)
                branch_ids.append(si.id)
                # searching new si
                for ns in sent:
                #print('si.head_id =', si.head_id)
                    if si.head_id == ns.id:
                        si = ns
                        break
            
            if si.id in root_ids or si.id in used_ids:
                #print('  skip:', si.id)
                root_ids.extend(branch_ids)
                continue
            #print('    branch =', [w.text for w in branch])
            # adding the branch to its sent_part
            for i in range(len(sent_parts)):
                #print(sent_parts_ids[i])
                if si.id in sent_parts_ids[i]:
                    sent_parts[i].extend(branch)
                    sent_parts_ids[i].extend(branch_ids)

        if return_used_ids == True:
            return [[word for word in sent if word in part] for part in sent_parts], list(set(self.flat_list(sent_parts_ids)))
        return [[word for word in sent if word in part] for part in sent_parts]

    # returns json generator
    # by the token list
    def sents2jsons(self):#, sents):

        for token_sent in self.tokens_list:

            root_parts = self.sentence_division(token_sent)
            word_bags = [[token.text for token in s] for s in root_parts]
            root_texts = [' '.join(part) for part in word_bags]
            #print(root_parts)

            for root_part_text in root_parts:
                #print('text =', tokens2txt(text))
                # obj
                param = ('obj')
                obj_parts, obj_ids = self.sentence_division(root_part_text, param, return_used_ids=True)
                #print(param, 'ids =', obj_ids)
                word_bags = [[token.text for token in s] for s in obj_parts]
                obj_texts = [' '.join(part) for part in word_bags]
                #print(param, '=', obj_texts)

                # nsubj
                param = ('nsubj')
                nsubj_parts, nsubj_ids = self.sentence_division(root_part_text, param, used_ids=obj_ids, return_used_ids=True)
                #print(param, 'ids =', nsubj_ids)
                word_bags = [[token.text for token in s] for s in nsubj_parts]
                nsubj_texts = [' '.join(part) for part in word_bags]
                #print(param, '=', nsubj_texts)

                # commit
                commit_parts = self.sentence_division(root_part_text, used_ids=obj_ids + nsubj_ids)
                word_bags = [[token.text for token in s] for s in commit_parts]
                commit_texts = [' '.join(part) for part in word_bags]
                #print('commit', '=', commit_texts)

                ans = {'obj' : obj_texts,
                        'nsubj' : nsubj_texts,
                        'commit' : commit_texts}
                #yield ans
                self.sents.append(ans)

    def doc2sents(self):
        self.tokens_list = [[word for word in self.doc.sents[n].tokens] for n in range(len(self.doc.sents))]

    def print_sents(self):
        for sent in self.sents:
            print(sent)

    def return_sents(self):
        return self.sents

segmented_text = SentSegmenter(text)
segmented_text.print_sents()
