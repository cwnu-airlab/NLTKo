import os

# import sys
# sys.path.append("/Users/dowon/nltk_ko/nltk/tag")
# from libs import *

"""
This script will run a POS or SRL tagger on the input data and print the results
to stdout.
"""

import argparse
import logging

from .libs import *
import requests
import zipfile

class EspressoTagger:
    def __init__(self, data_dir=None, lang='ko'):
        self.data_dir = data_dir
        if data_dir == None:
            path=os.path.dirname(__file__)
            path=path + '/data'
            self.data_dir = path
        #     print(path)
        self.lang = lang

    def tag(self, task, text, use_tokenizer=False):
        """
        This function provides an interactive environment for running the system.
        It receives text from the standard input, tokenizes it, and calls the function
        given as a parameter to produce an answer.
        
        :param task: 'pos', ner', 'wsd', 'srl' or 'dependency'
        :param use_tokenizer: whether to use built-in tokenizer
        """
        
        use_tokenizer = not use_tokenizer
        task_lower = task.lower()
        
        if not self._check_model():
            self._download_model()

        set_data_dir(self.data_dir)

        if task_lower == 'pos':
                tagger = taggers.POSTagger(language=self.lang, data_dir=self.data_dir)
        elif task_lower == 'ner':
                tagger = taggers.NERTagger(language=self.lang, data_dir=self.data_dir)
        elif task_lower == 'wsd':
                tagger = taggers.WSDTagger(language=self.lang, data_dir=self.data_dir)
        elif task_lower == 'srl':
                tagger = taggers.SRLTagger(language=self.lang, data_dir=self.data_dir)
        elif task_lower == 'dependency':
                tagger = taggers.DependencyParser(language=self.lang)
        else:
                raise ValueError('Unknown task: %s' % task)
        
        # while True:
        #         try:
        #                 text = input("> ")
        #         except KeyboardInterrupt:
        #                 break
        #         except EOFError:
        #                 break

        #         if not text : break
                
        if use_tokenizer:
                result = tagger.tag(text)
        else:
                tokens = text.split()
                if task_lower != 'dependency':
                        result = [tagger.tag_tokens(tokens, True)]
                else:
                        result = [tagger.tag_tokens(tokens)]						
        return self._result_tagged(result, task_lower)

    def _result_tagged(self, tagged_sents, task):
        """
        Prints the tagged text to stdout.
        
        :param tagged_sents: sentences tagged according to any of espresso taggers.
        :param task: the tagging task (either 'pos', 'ner', 'wsd', 'srl' or 'dependency')
        """

        ##TODO: print부분 return으로 변경
        if task == 'pos':
                return self._return_tagged_pos(tagged_sents)
        elif task == 'ner':
                return self._return_tagged_ner(tagged_sents)
        elif task == 'wsd':
                return self._return_tagged_wsd(tagged_sents)
        elif task == 'srl':
                return self._return_tagged_srl(tagged_sents)
        elif task == 'dependency':
                return self._return_parsed_dependency(tagged_sents)
        else:
                raise ValueError('Unknown task: %s' % task)

            

    def _return_parsed_dependency(self, parsed_sents):
        """Prints one token per line and its head"""
        result = []
        temp_list = []
        temp_list2 = []
        for sent in parsed_sents:
            temp_list = sent.to_conll().split('\t')
            temp_list = temp_list[1:]
            for ele in temp_list:
                if '\n' in ele:
                    ele = ele[:ele.find('\n')]
                temp_list2.append(ele)
            result.append(self._dependency_after(temp_list2)[:])
            temp_list2 = []
            
        return result

    def _return_tagged_pos(self, tagged_sents):
        """Prints one sentence per line as token_tag"""
        result = []
        for sent in tagged_sents:
                # s = ' '.join('_'.join(item) for item in sent)
            for item in sent:
                s = '_'.join(item)
                result.append(s)

        return result

    def _return_tagged_srl(self, tagged_sents):
        for sent in tagged_sents:
                print (' '.join(sent.tokens))
                for predicate, arg_structure in sent.arg_structures:
                        print (predicate)
                        for label in arg_structure:
                                argument = ' '.join(arg_structure[label])
                                line = '\t%s: %s' % (label, argument)
                                print (line)
                print ('\n')

    def _return_tagged_ner(self, tagged_sents):
        """Prints one sentence per line as token_tag"""
        result = []
        for sent in tagged_sents:
            for item in sent:
                s = '_'.join(item)
                result.append(s)
        
        return result

    def _return_tagged_wsd(self, tagged_sents):
        """Prints one sentence per line as token_tag"""
        result = []
        for sent in tagged_sents:
            for item in sent:
                s = '_'.join(item)
                result.append(s)
        
        return result

    def _download_model(self):
        """Downloads the model from the server"""
        temp_path = os.path.dirname(__file__) + '/data.zip'
        url = "https://air.changwon.ac.kr/~airdemo/storage/espresso_data_1/data.zip"
        print("Downloading Espresso5 model...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                    f.write(chunk)

        if os.path.exists(self.data_dir):
            os.rmdir(self.data_dir)

        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(__file__))

    def _check_model(self):
        """Checks if the model is available and downloads it if necessary"""
        if not os.path.exists(self.data_dir):
            return False
        else:
            return True

    def _dependency_after(self, list):
        len_list = len(list)
        temp_list = []
        repeat = len_list//3
        for i in range(repeat):
            index = i*3
            tup1 = (i+1, )
            tup2 = tuple(list[index:index+3])
            tup = tup1 + tup2
            temp_list.append(tup[:])
            
            
        return temp_list

if __name__ == '__main__':
    tagger = EspressoTagger()
    print(tagger.tag('pos', "나는 아름다운 강산에 살고있다."))