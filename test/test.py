from nltk.alignment import NeedlemanWunsch, SmithWaterman, Hirschberg, LongestCommonSubsequence, LongestCommonSubstring, DTW
from nltk.distance import LevenshteinEditDistance, HammingDistance, DamerauLevenshteinDistance, WassersteinDistance
from nltk.similarity import CosineSimilarity, LCSubstringSimilarity, LCSubsequenceSimilarity, JaroSimilarity
from nltk.tokenize import sent_tokenize, word_tokenize, syllable_tokenize
from nltk.search import NaiveSearch, RabinKarpSearch, KMPSearch, BoyerMooreSearch, FaissSearch
from nltk.metrics import BERTScore, BARTScore, DefaultMetric
from nltk import pos_tag, nouns, word_segmentor, pos_tag_with_verb_form
import numpy as np
from typing import List
import torch

def demo():
    str1 = '기존에 제품이 장기간 사용으로 손상'
    str2 = '장기간 사용으로 제품이 손상'

    # result1, result2 = NeedlemanWunsch().get_alignment(str1, str2)
    # print(result1, '\n', result2)

    result1, result2 = SmithWaterman().get_alignment(str1, str2)
    print(f"{result1}\n{result2}")

    # result1, result2 = Hirschberg().get_alignment(str1, str2)
    # print(f"{result1}\n{result2}")

    # result = DTW().get_alignment_path(str1, str2)
    # print(result)

    # result = LongestCommonSubsequence().compute(str1, str2)
    # print(result)

    # result = LongestCommonSubstring().compute(str1, str2)
    # print("-------LongestCommonSubstring-------")
    # print(result)
    # print("------------------------------------")
    # print()

def demo2():
    str1 = '나는 학생이다.'
    str2 = '그는 선생님이다.'

    result = BARTScore().compute([str1], [str2])
    print("-------BARTScore-------")
    print(result)
    print("-----------------------")
    print()

def demo3():
    str1 = '나는 학생이다.'
    str2 = '그는 선생님이다.'
    model_name = 'bert-base-uncased'
    result = BERTScore(model_name_or_path=model_name, lang='kor', num_layers=12).compute([str1], [str2])

    print("model name: ", model_name)
    print("-------BERTScore-------")
    print(result)
    print("-----------------------")
    print()

def demo4():
    demo_setences = ['제가 나와 있는 곳은 경남 거제시 옥포동 덕포 해수욕장에 나와 있습니다.']
    for sen in demo_setences:
        print(word_tokenize(sen, "korean"))
        print(pos_tag(sen, lang='kor'))

def demo5():
    str1 = '나는 학생이다.'
    str2 = '그는 선생님이다.'

    # result = LevenshteinEditDistance().compute(str1, str2)

    # result = HammingDistance().compute(str1, str2)


    result = DamerauLevenshteinDistance().compute(str1, str2)

    print("-------DamerauLevenshteinDistance-------")
    print(result)
    print("----------------------------------------")
    print()

def demo6():
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([3, 7, 8, 3, 1])

    result = CosineSimilarity().compute(x1, x2)

    print("-------CosineSimilarity-------")
    print(result)
    print("------------------------------")
    print()

def demo7():
    str1 = '나는 학생이다.'
    str2 = '그는 선생님이다.'

    result = LCSubstringSimilarity().compute(str1, str2)

    print("-------LCSubstringSimilarity-------")
    print(result)
    print("-----------------------------------")
    print()

    result = LCSubsequenceSimilarity().compute(str1, str2)

    print("-------LCSubsequenceSimilarity-------")
    print(result)
    print("--------------------------------------")
    print()

    result = JaroSimilarity().compute(str1, str2)

    print("-------JaroSimilarity-------")
    print(result)
    print("----------------------------")
    print()


def demo8():
    pattern = "학생"
    str1 = '나는 학생이다.'

    result = NaiveSearch().search(pattern, str1)
    print(result)

    result = RabinKarpSearch().search(pattern, str1)
    print(result)

    result = KMPSearch().search(pattern, str1)
    print(result)

    result = BoyerMooreSearch().search(pattern, str1)
    print(result)

def demo9():
    faiss = FaissSearch(model_name_or_path = 'skt/kobert-base-v1', tokenizer_name_or_path = 'skt/kobert-base-v1')
    corpus = {
        'text': [
                "오늘은 날씨가 매우 덥습니다.",
                "저는 음악을 듣는 것을 좋아합니다.",
                "한국 음식 중에서 떡볶이가 제일 맛있습니다.",
                "도서관에서 책을 읽는 건 좋은 취미입니다.",
                "내일은 친구와 영화를 보러 갈 거예요.",
                "여름 휴가 때 해변에 가서 수영하고 싶어요.",
                "한국의 문화는 다양하고 흥미로워요.",
                "피아노 연주는 나를 편안하게 해줍니다.",
                "공원에서 산책하면 스트레스가 풀립니다.",
                "요즘 드라마를 많이 시청하고 있어요.",
                "커피가 일상에서 필수입니다.",
                "새로운 언어를 배우는 것은 어려운 일이에요.",
                "가을에 단풍 구경을 가고 싶어요.",
                "요리를 만들면 집안이 좋아보입니다.",
                "휴대폰 없이 하루를 보내는 것이 쉽지 않아요.",
                "스포츠를 하면 건강에 좋습니다.",
                "고양이와 개 중에 어떤 동물을 좋아하세요?"
                "천천히 걸어가면서 풍경을 감상하는 것이 좋아요.",
                "일주일에 한 번은 가족과 모임을 가요.",
                "공부할 때 집중력을 높이는 방법이 있을까요?",
                "봄에 꽃들이 피어날 때가 기대되요.",
                "여행 가방을 챙기고 싶어서 설레여요.",
                "사진 찍는 걸 좋아하는데, 카메라가 필요해요.",
                "다음 주에 시험이 있어서 공부해야 해요.",
                "운동을 하면 몸이 가벼워집니다.",
                "좋은 책을 읽으면 마음이 풍요로워져요.",
                "새로운 음악을 발견하면 기분이 좋아져요.",
                "미술 전시회에 가면 예술을 감상할 수 있어요.",
                "친구들과 함께 시간을 보내는 건 즐거워요.",
                "자전거 타면 바람을 맞으면서 즐거워집니다."
        ],
    }
    print(faiss.initialize_corpus(corpus=corpus, section='text', embedding_type='mean_pooling', save_path='/Users/dowon/Test/test.json'))
    query = "오늘은 날씨가 매우 춥다."
    top_k = 5
    result = faiss.search(query, top_k)
    print(result)

def faiss_test():
    faiss = FaissSearch(model_name_or_path = 'klue/bert-base')
    result = TextReader("/Users/dowon/Test/sentence1.txt").read()
    id = 0
    
    for i in result:
        print(i)
        i = i.replace('\n', '')
        print(i)
        i = "i am test"
        print(faiss.get_embeddings(text=i, num_workers=10).detach().cpu().numpy())
        id += 1
        if id ==3:
            break

def faiss_save_test():
    faiss = FaissSearch(model_name_or_path = '/Users/dowon/test_model/trained_model/', tokenizer_name_or_path = '/Users/dowon/test_model/trained_model/')
    faiss.load_dataset_from_json('/Users/dowon/Test/test.json')
    faiss.embedding_type = 'mean_pooling'
    # faiss.load_faiss_index(index_name='embeddings',file_path='/Users/dowon/Test/test_index.json')
    faiss.add_faiss_index(column_name='embeddings')
    query = "오늘은 날시가 매우 춥다."
    top_k = 5
    result = faiss.search(query, top_k)
    print(result)
    

def demo10():
    metric = DefaultMetric()
    y_true = [1, 3, 3, 5, 5,1]
    y_pred = [1, 2, 3, 4, 5,2]
    str1 = "i am teacher"
    str2 = "he is student"
    print(metric.precision_score(y_true, y_pred, "macro"))

def demo11():
    print("\nBegin Wasserstein distance demo ")

    P =  np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    Q1 = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
    Q2 = np.array([0.1, 0.1, 0.1, 0.1, 0.6])

    P = torch.from_numpy(P)
    Q1 = torch.from_numpy(Q1)
    Q2 = torch.from_numpy(Q2)
    kl_p_q1 = WassersteinDistance().compute_kullback(P, Q1)
    kl_p_q2 = WassersteinDistance().compute_kullback(P, Q2)

    wass_p_q1 = WassersteinDistance().compute_wasserstein(P, Q1)
    wass_p_q2 = WassersteinDistance().compute_wasserstein(P, Q2)

    jesson_p_q1 = WassersteinDistance().compute_jesson_shannon(P, Q1)
    jesson_p_q2 = WassersteinDistance().compute_jesson_shannon(P, Q2)


    print("\nKullback-Leibler distances: ")
    print("P to Q1 : %0.4f " % kl_p_q1)
    print("P to Q2 : %0.4f " % kl_p_q2)

    print("\nWasserstein distances: ")
    print("P to Q1 : %0.4f " % wass_p_q1)
    print("P to Q2 : %0.4f " % wass_p_q2)

    print("\nJesson-Shannon distances: ")
    print("P to Q1 : %0.4f " % jesson_p_q1)
    print("P to Q2 : %0.4f " % jesson_p_q2)
    
    print("\nEnd demo ")

def demo12():
	y_pred = [5, 2, 4, 1, 3, 2, 5, 6, 7]
	y_true = [1, 3, 6, 7, 1, 5]

	user = [[5, 3, 2], [9, 1, 2], [3, 5, 6], [7, 2, 1]]
	h_pred = [[15, 6, 21, 3], [15, 77, 23, 14], [51, 23, 21, 2], [53, 2, 1, 5]]

	metric = DefaultMetric()
	print(metric.precision_at_k(y_true,  y_pred, 3))
	print(metric.recall_at_k(y_true,y_pred, 3))
	print(metric.hit_rate_at_k(user, h_pred, 1))



class TextReader:
    def __init__(self, path: str):
        self.path = path

    def read(self) -> List[str]:
        with open(self.path, 'r') as f:
            return f.readlines()


if __name__=="__main__":
    # demo()
    # demo2()
    # demo3()
    #demo4()
    # demo5()
    # demo6()
    # demo7()
    # demo8()
    # demo9()
    # faiss_test()
    # faiss_save_test()
    # demo10()
    demo11()
    #demo12()
