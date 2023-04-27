import spacy
from pyinflect import getInflection
import pandas as pd

class InteractionParser:
    
    ENTS_TO_REMOVE = ['DATE', 'GPE', 'FAC', 'ORG', 'LOC', 'TIME']
    
    def __init__(self, mask_names=False, require_gpu=False, spacy_model='en_core_web_trf', **kwargs):
        if require_gpu:
            spacy.require_gpu(**kwargs)
        print('Loading Spacy model:', spacy_model)
        self.nlp = spacy.load(spacy_model)
        print('Spacy model loaded')
        self.mask_names = mask_names
    
    def _get_text_with_det(self, token, conj=''):
        token_text = '[NAME]' if self.mask_names and token.ent_type_ == 'PERSON' else token.text
        dets = [c for c in token.children if c.dep_ == 'det' or c.dep_ == 'poss']
        
        out = token_text
        if len(dets) > 0:
            out = dets[0].text + ' ' + out
        if conj != '':
            out = conj + ' ' + out
            
        return out

    def _expand_name_conjunctions(self, token):
        # NOTE: known limitation: can only handle two people (not "Alice, Bob and Cathy")
        if token.ent_type_ != 'PERSON':
            yield '', token
        else:
            conjs = [x for x in token.children if x.dep_ == 'cc']
            conj = '' if len(conjs) == 0 else conjs[0].text
            yield '', token
            if conj != '': # will break in case of 3 or more people as shown above, because first conj is null
                for x in token.children:
                    if x.dep_ == 'conj' and x.ent_type_ == 'PERSON':
                        yield conj, x
    
    def _get_pp_data(self, prep, obj):
        text = f'{prep.text}::{self._get_text_with_det(obj)}'
        dep = obj.dep_
        ent_type = obj.ent_type_
        return (text, dep, ent_type)
    
    def _get_action_data(self, text):
        doc = self.nlp(text)

        for verb in [token for token in doc if token.pos_ == 'VERB']:

            children = [
                (self._get_text_with_det(y, conj=conj), x.dep_, y.ent_type_) 
                ## ^ careful with x and y above
                for x in verb.children
                for conj, y in self._expand_name_conjunctions(x)
            ]
            pp_subchildren = [
                self._get_pp_data(x, y)
                for x in verb.children
                if x.dep_ == 'prep'
                for y in x.children
            ]
            out_df = pd.DataFrame(children + pp_subchildren, columns=['token', 'dep', 'ent_type'])
            out_df = out_df[~out_df.dep.isin(['punct', 'prep'])].copy()
            out_df.token = out_df.token.str.strip()
            out_df = out_df[out_df.token != '']
            out_df = out_df[~out_df.ent_type.isin(self.ENTS_TO_REMOVE)].copy()
            
            yield verb.lemma_, out_df

    def _get_interaction_data(self, text):
        for L, df in self._get_action_data(text):
            if (df.ent_type == 'PERSON').sum() > 1:
                yield L, df
        
    def _idata2descriptions(self, idata):
        for lemma, args_df in idata:
            args_df['isname'] = args_df.ent_type == 'PERSON'

            n_people = (args_df.isname & args_df.dep.isin(['nsubj', 'dobj', 'pobj'])).sum()
            has_person_subj = ((args_df.dep == 'nsubj') & args_df.isname).any()

            if n_people > 1 and has_person_subj:

                subj = ' '.join(args_df[(args_df.dep == 'nsubj') & args_df.isname].token.tolist())

                infl = getInflection(lemma, 'VBG')
                if infl is not None:
                    gerund = infl[0]
                    desc = f'{subj} {gerund}'

                    for arg_row in args_df[args_df.dep == 'dobj'].itertuples():
                        obj = arg_row.token
                        desc += f' {obj}'


                    for arg_row in args_df[args_df.dep == 'pobj'].itertuples():
                        if '::' in arg_row.token: # should always contain this but in rare cases it doesn't
                            preposition, obj = arg_row.token.split('::')
                            desc += f' {preposition} {obj}'

                    yield lemma, desc

    def parse(self, text):
        idata = self._get_interaction_data(text)
        return list(self._idata2descriptions(idata))