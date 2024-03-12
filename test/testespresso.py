from nltk.tag import EspressoTagger

if __name__ == '__main__':
    sent = "나는 배가 고프다. 나는 아름다운 강산에 살고있다." 
    tagger = EspressoTagger()
    print()
    print(tagger.tag('pos', sent))
    print("dependency :")
    print(tagger.tag('dependency', sent))
    print('ner :')
    ner = tagger.tag('ner', sent)
    print(ner)
    print()
    print()
    print('wsd :')
    print(tagger.tag('wsd', sent))
    print()
    #print('srl :')
    #print(tagger.tag('srl', sent))
