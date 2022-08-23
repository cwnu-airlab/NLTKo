# Natural Language Toolkit: NLTK's very own tokenizer.
#
# Copyright (C) 2001-2020 NLTK Project
# Author:
# URL: <http://nltk.sourceforge.net>
# For license information, see LICENSE.TXT


import re
from nltk.tokenize.api import TokenizerI


class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """

    CONTRACTIONS2 = [
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(d)(?#X)('ye)\b",
        r"(?i)\b(gim)(?#X)(me)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(lem)(?#X)(me)\b",
        r"(?i)\b(mor)(?#X)('n)\b",
        r"(?i)\b(wan)(?#X)(na)\s",
    ]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b", r"(?i)\b(wha)(t)(cha)\b"]


class NLTKWordTokenizer(TokenizerI):
    """
    The NLTK tokenizer that has improved upon the TreebankWordTokenizer.

    The tokenizer is "destructive" such that the regexes applied will munge the
    input string to a state beyond re-construction. It is possible to apply
    `TreebankWordDetokenizer.detokenize` to the tokenized outputs of
    `NLTKDestructiveWordTokenizer.tokenize` but there's no guarantees to
    revert to the original string.
    """

    # Starting quotes.
    STARTING_QUOTES = [
        (re.compile(u"([«“‘„]|[`]+)", re.U), r" \1 "),
        (re.compile(r"^\""), r"``"),
        (re.compile(r"(``)"), r" \1 "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
        (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d)(\w)\b", re.U), r"\1 \2"),
    ]

    # Ending quotes.
    ENDING_QUOTES = [
        (re.compile(u"([»”’])", re.U), r" \1 "),
        (re.compile(r'"'), " '' "),
        (re.compile(r"(\S)(\'\')"), r"\1 \2 "),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # For improvements for starting/closing quotes from TreebankWordTokenizer,
    # see discussion on https://github.com/nltk/nltk/pull/1437
    # Adding to TreebankWordTokenizer, nltk.word_tokenize now splits on
    # - chervon quotes u'\xab' and u'\xbb' .
    # - unicode quotes u'\u2018', u'\u2019', u'\u201c' and u'\u201d'
    # See https://github.com/nltk/nltk/issues/1995#issuecomment-376741608
    # Also, behavior of splitting on clitics now follows Stanford CoreNLP
    # - clitics covered (?!re|ve|ll|m|t|s|d)(\w)\b

    # Punctuation.
    PUNCTUATION = [
        (re.compile(r'([^\.])(\.)([\]\)}>"\'' u"»”’ " r"]*)\s*$", re.U), r"\1 \2 \3 "),
        (re.compile(r"([:,])([^\d])"), r" \1 \2"),
        (re.compile(r"([:,])$"), r" \1 "),
        (re.compile(r"\.{2,}", re.U), r" \g<0> "), # See https://github.com/nltk/nltk/pull/2322
        (re.compile(r"[;@#$%&]"), r" \g<0> "),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r"\1 \2\3 ",
        ),  # Handles the final period.
        (re.compile(r"[?!]"), r" \g<0> "),
        (re.compile(r"([^'])' "), r"\1 ' "),
        (re.compile(r"[*]", re.U), r" \g<0> "), # See https://github.com/nltk/nltk/pull/2322
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile(r"\("), "-LRB-"),
        (re.compile(r"\)"), "-RRB-"),
        (re.compile(r"\["), "-LSB-"),
        (re.compile(r"\]"), "-RSB-"),
        (re.compile(r"\{"), "-LCB-"),
        (re.compile(r"\}"), "-RCB-"),
    ]

    DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text, convert_parentheses=False, return_str=False):
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        # Optionally convert parentheses
        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r" \1 \2 ", text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r" \1 \2 ", text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self._contractions.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text if return_str else text.split()




#korean 

class Enum(object):
    def __init__(self, names): 
        for value, name in enumerate(names.split()): setattr(self, name, value)

class ko_tokenize():

    def tokenize(target, encoding='utf8'):
        """ Word Tokenizer
        
        단어 단위로 Tokenizing 한다.
        
        인자값 목록 (모든 변수가 반드시 필요):
        
        target : Tokenizing 하고자 하는 대상 문자열
        
        결과값 : 토크나이징 결과를 list 자료형으로 넘김
        
        """
        isHangulSyllables = lambda x: unicodedata.name(x).find("HANGUL SYLLABLE") == 0
        isHanjaSyllables = lambda x: unicodedata.name(x).find("CJK") == 0
        isNumber = lambda x: unicodedata.name(x).find("FULLWIDTH DIGIT") == 0 or unicodedata.name(x).find("DIGIT") == 0
        isAlphabet = lambda x: unicodedata.name(x).find("FULLWIDTH LATIN") == 0 or unicodedata.name(x).find("LATIN") == 0
        isAlphabet_Connection = lambda x: x in (".", "-", "_", "|")
        isNumber_Connection = lambda x: x in (".", ",")
        isPunctuation = lambda x: unicodedata.category(x)[0] == "P"
        isSymbol = lambda x: unicodedata.category(x)[0] == "S"
        getCategory = lambda x: unicodedata.category(x)
        
        TYPE = Enum("UNKNOWN SYMBOL NUMBER PUNCTUATION ALPHABET HANJA HANGUL")
        
        buf = str()
        type_prev = 0
        type_cur = 0
        
        if type(target) == str:
            target = target

        for i in range(len(target)):
            ch = target[i]
            ca = str()
            try:
                if isHangulSyllables(ch): type_cur = TYPE.HANGUL
                elif isHanjaSyllables(ch): type_cur = TYPE.HANJA
                elif isNumber(ch): type_cur = TYPE.NUMBER
                elif isAlphabet(ch): type_cur = TYPE.ALPHABET
                elif isAlphabet_Connection(ch) and type_prev == TYPE.ALPHABET:
                    if i+1 < len(target) and not isAlphabet(target[i+1]): type_cur = TYPE.SYMBOL
                    else: type_cur = TYPE.ALPHABET
                elif isNumber_Connection(ch) and type_prev == TYPE.NUMBER:
                    if i+1 < len(target) and not isNumber(target[i+1]): type_cur = TYPE.SYMBOL
                    elif i+1 == len(target): type_cur = TYPE.SYMBOL
                    else: type_cur = TYPE.NUMBER
                elif isPunctuation(ch): type_cur = TYPE.PUNCTUATION
                elif isSymbol(ch): type_cur = TYPE.SYMBOL
                else: type_cur = TYPE.UNKNOWN
                ca = getCategory(ch)
            except:
                type_cur = TYPE.UNKNOWN
            if type_cur == TYPE.PUNCTUATION:
                if ca in ("Ps", "Pe"): buf += " "
                elif i >= 0 and i < len(target) and target[i-1] != target[i]: buf += " "
            elif type_cur != type_prev: buf += " "
            buf += ch
            type_prev = type_cur
        return buf.split()



    def eumjeol_tokenize(text,blank):
        """
        음절 토크나이저

        음절단위로 tokenizing

        박찬양
        """
        emjeol_list = list()
        for emjeol in text:

          if blank and (emjeol not in ['\n']):
           emjeol_list.append(emjeol)

          elif emjeol not in [' ', '\n']:
           emjeol_list.append(emjeol)

        return emjeol_list


    def sentence_tokenize(text):
        """
        문장 토크나이저

        문장단위로 tokenizing
        """
        txt=text.replace("\n"," ")
        p=re.compile(r'(?<!\w\.\w.)(?<=\.|\?|\!)\s').split(txt)
        result=[]
        for tmp in p:
            if (tmp == ' ' or tmp== ''):
                continue
            else:result.append(tmp.strip(" "))

        return result

