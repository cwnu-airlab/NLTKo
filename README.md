# Manual [NLTKo]

## **Revision History**

| 번호 | 내용                                                         | 작성자 | 날짜     |
| ---- | ------------------------------------------------------------ | ------ | -------- |
| 1    | NLTKo (version1.0.1) : ssem, word_tokenize                   | 홍성태 | 20.12.28 |
| 2    | NLTKo (version1.0.2) : tokenize 완성, korChar 추가           | 홍성태 | 20.12.31 |
| 3    | NLTKo (version1.0.3) : korChar.py 수정 (naming convention)   | 홍성태 | 21.01.05 |
| 4    | NLTKo (version1.0.4) <br />metric.py (accuracy, recall, precision,f_score) 추가<br />eval.py (bleu, rouge, wer, rouge) 추가 | 홍성태 | 21.02.01 |
| 5    | NLTKo (version1.0.5)<br />eval.py (rouge-s, rouge-l) & README.md 수정 | 홍성태 | 21.02.02 |
| 6    | NLTKo (version1.0.6)<br />eval.py (cider) 추가 & README.md 수정 | 홍성태 | 21.02.19 |
| 7    | NLTKo (version1.0.7)<br />metric.py (pos_eval) 추가 & README.md 수정 | 홍성태 | 21.03.31 |
| 8    | NLTKo (version1.0.8)<br />metric.py (pos_eval) 수정 & README.md 수정 | 홍성태 | 21.04.02 |
| 9    | NLTKo(version 1.0.9)<br />Tag(형태소분석) 관련 함수 추가 & README.md 수정 | 홍성태 | 21.06.21 |
| 10   | NLTKo(version 1.1.0)<br />eval.py (Meteor 추가) &  parser.py 추가 & README.md 수정 | 홍성태 | 21.07.20 |
| 11   | NLTKo(version 1.1.1)<br />trans.py (papago 추가) & parser ➔ etc 변경 & 음절 토크나이저 blank 수정 | 홍성태 | 21.12.09 |
| 12   | NLTKo(version 1.1.2)<br />trans.py (papago key auto update) & pos_tag engine Espresso5로 변경 | 홍성태 | 22.07.08 |
| 13   | NLTKo(version 1.1.3)<br />pos_tag 모듈 수정(nouns, word_segmentor) | 홍성태 | 22.07.29 |
| 14   | NLTKo(version 1.1.4)<br />형태소분석기 동적라이브러리 생성을 위한 setup.py 수정 | 홍성태 | 22.08.19 |
| 15   | NLTKo(version 1.1.5)<br />Python3.8에서 실행을 위해 setup.py 수정 | 김도원 | 23.07.20 |
| 16   | NLTKo(version 1.2.0)<br />string2string 기능 추가 | 김도원 | 23.08.16 |
| 17   | NLTKo(version 1.2.1)<br />Metrics 통합 | 김도원 | 23.08.24 |
| 18   | NLTKo(version 1.2.2)<br />Metrics 클래스 명칭 정리 | 김도원 | 23.08.24 |
| 19   | NLTKo(version 1.2.3)<br />Espresso5 pos tag 추가 | 김도원 | 23.08.31 |
| 20   | NLTKo(version 1.2.4)<br />Espresso5 'ner tag', 'wsd tag', 'dependency parser' 추가 | 김도원 | 23.09.11 |
| 21   | NLTKo(version 1.2.5)<br />Kobert Tokenizer 추가, MAUVE 평가 지표 추가 | 김도원 | 23.11.17 |
| 22   | NLTKo(version 1.2.6)<br />BERTScore, BARTScore 위치 변경, 'dependency parser'출력 형식 변경 | 김도원 | 23.11.17 |
| 23   | NLTKo(version 1.2.7)<br />Precision@K, Recall@K, HitRate@K, Wasserstein Distance 추가 | 김도원 | 23.11.23 |
| 24   | NLTKo(version 1.2.8)<br />Jesson-Shannon distance 추가, Espresso5 ubuntu환경에서 실행 가능하도록 변경 | 김도원 | 23.11.30 |
| 25   | NLTKo(version 1.2.9)<br />METEOR Score 복원 | 김도원 | 24.02.22 |
| 26   | NLTKo(version 1.2.10)<br />METEOR Score 오류 수정, SRL tagger 추가 | 김도원 | 24.02.22 |


<div style="page-break-after: always;"></div>

## 목차



* [1.목적](#1.-목적)
* [2. 시스템 기능 정의](#2-시스템-기능-정의)
  + [2.1 토크나이저](#21-토크나이저)
  + [2.2 세종전자사전 활용](#22-세종전자사전-활용)
  + [2.3 한국어 전처리](#23-한국어-전처리)
  + [2.4 분류 모델 평가](#24-분류-모델-평가)
  + [2.5 기계 번역 평가](#25-기계-번역-평가)
  + [2.6 pos_tag](#26-pos_tag)
  + [2.7 Translate](#27-translate)
  * [2.8 string2string](#28-string2string)
* [3. 사용 환경](#3-사용-환경)
  * [3.1 라이브러리 설치](#31-라이브러리-설치)
	* [3.1.1 설치 도중 오류 발생시 해결 방법](#311-설치-도중-오류-발생시-해결-방법)
* [4.실행](#4-실행)
  * [4.1 토크나이저 (tokenizer)](#41-토크나이저-tokenizer)
  * [4.2 세종전자사전 (ssem)](#42-세종전자사전-ssem)
    * [4.2.1 객체 확인 방법](#421-객체-확인-방법 )
    * [4.2.2 entry 접근법](#422-entry-접근법)
    * [4.2.3 sense 접근법](#423-sense-접근법)
    * [4.2.4 entry 함수 사용법 & 결과](#424-entry-함수-사용법--결과)
    * [4.2.5 sense 함수 사용법 & 결과](#425-sense-함수-사용법--결과)
  * [4.3 한국어 전처리 (korchar)](#43-한국어전처리-korchar)
  * [4.4 분류 모델 평가](#44-분류모델평가)
    * [4.4.1 DefaultMetric](#441-defaultmetric)
        * macro
        * accuracy
        * precision
        * recall
        * f1_score
        * precision@k
        * recall@k
        * hitrate@k
    * [4.4.2 MAUVE](#442-mauve)
    * [4.4.3 BERT Score](#443-bert-score)
    * [4.4.4 BART Score](#444-bart-score)
  * [4.5 기계 번역 평가 (StringMetric)](#45-기계번역평가-stringmetric)
    * [4.5.1 WER/CER](#451-wercer)
    * [4.5.2 BLEU](#452-bleu)
    * [4.5.3 ROUGE](#453-rouge)
    * [4.5.4 CIDER](#454-cider)
    * [4.5.5 METEOR](#455-meteor)
  * [4.6 Espresso5](#46-espresso5)
	  * [4.6.1 pos tag](#461-pos-tag)
	  * [4.6.2 dependency parse](#462-dependency-parse)
	  * [4.6.3 wsd tag](#463-wsd-tag)
	  * [4.6.4 ner tag](#464-ner-tag)
    * [4.6.5 srl tag](#465-srl-tag)
  * [4.7 Translate](#47-translate)
  * [4.8 정렬 (alignment)](#48-정렬-alignment)
	  * [4.8.1 Needleman-Wunsch 알고리즘](#481-needleman-wunsch-알고리즘)
	  * [4.8.2 Hirschberg 알고리즘](#482-hirschberg-알고리즘)
	  * [4.8.3 Smith-Waterman 알고리즘](#483-smith-waterman-알고리즘)
	  * [4.8.4 DTW](#484-dtw)
	  * [4.8.5 Longest Common Subsequence](#485-longest-common-subsequence)
	  * [4.8.6 Longest Common Substring](#486-longest-common-substring)
  * [4.9 거리 (distance)](#49-거리-distance)
	  * [4.9.1 Levenshtein Edit Distance](#491-levenshtein-edit-distance)
	  * [4.9.2 Hamming Distance](#492-hamming-distance)
	  * [4.9.3 Damerau-Levenshtein Distance](#493-damerau-levenshtein-distance)
    * [4.9.4 Wasserstein Distance](#494-wasserstein-distance)
      * Wasserstein Distance
      * Kullback-Leibler Distance
      * Jensen-Shannon Distance
  * [4.10 유사도 (similarity)](#410-유사도-similarity)
	  * [4.10.1 코사인 유사도 (Cosine Similarity)](#4101-코사인-유사도-cosine-similarity)
	  * [4.10.2 LCSubstring Similarity](#4102-lcsubstring-similarity)
	  * [4.10.3 LCSubsequence Similarity](#4103-lcsubsequence-similarity)
	  * [4.10.4 Jaro Similarity](#4104-jaro-similarity)
  * [4.11 검색 (search)](#411-검색-search)
	  * [4.11.1 Naive Search](#4111-naive-search)
	  * [4.11.2 Rabin-Karp 검색](#4112-rabin-karp-검색)
	  * [4.11.3 KMP 검색 알고리즘](#4113-kmp-검색)
	  * [4.11.4 Boyer-Moore 검색 알고리즘](#4114-boyer-moore-검색)
	  * [4.11.5 Faiss-Semantic 검색](#4115-faiss-semantic-검색)
  * [4.12 etc](#412-etc)
* [5.사용예제 코드](#5-사용예제-코드)
  * [5.1 세종전자사전 예제 코드](#51-세종전자사전-예제-코드)
  * [5.2 한국어 전처리 예제 코드](#52-한국어-전처리-예제-코드)
  * [5.3 분류 모델 평가 예제 코드](#53-분류-모델-평가-예제-코드)
  * [5.4 기계 번역 평가 예제 코드](#54-기계-번역-평가-예제-코드)
  * [5.5 Tag 인터페이스 예제 코드](#55-tag-인터페이스-예제-코드)
* [6.부록](#6-부록)

<div style="page-break-after: always;"></div>

## 1. 목적

   NLTKo는 한국어를 위한 NLTK이며 기존의 영어에서 사용하는 WordNet 대신 세종전자사전을 사용한다. NLTK와 동일한 함수명을 사용하여 한국어 정보들을 손쉽게 사용하는데 목적이 있다. 설치 방법은 [다음](#31-라이브러리-설치)와 같다.

## 2. 시스템 기능 정의

### 2.1 토크나이저

​	자연어 문서를 분석하기 위해서 긴 문자열을 나누어야 한다. 이 문자열 단위를 토큰 (token) 이라고 하고  문자열을 토큰으로 나누는 작업을 토큰 생성(tokenizing) 이라 한다.  현재 ko_nltk에서는 사용자가 분석에 필요한 작업 토큰의 단위에 따라 **문장, 어절, 음절 토크나이징**이 모두 가능하다.

### 2.2 세종전자사전 활용

​	세종 전자사전의 단어 XML 파일을 이용하여 **단어의 파생어, 복합어, 숙어, 동의어, 반의어, 동위어, 상위어, 하위어, 전체어, 부분어, 관련어, 영어, 예시, 형용사 결합, 명사 결합, 동사 결합, 선택제약, 경로, 유사도 정보**의 리스트를 반환한다. 아래의 XML tag에 따라 ko_nltk에서 사용가능한 기능들을 구분하였으며, 아래 그림은 해당 entry의 대표 구조도와 같다.  

~~~h
entrys(word)
  ⨽ entry 1
        ⨽  idm : 파생어
        ⨽  comp : 복합어
        ⨽  der : 숙어
        ⨽senses 
            ⨽sense 1
                ⨽  sem : sem
               	⨽  syn : 동의어
                ⨽  ant : 반의어
                ⨽  coord : 동위어
                ⨽  hyper : 상위어
             		⨽  hypo : 하위어
                ⨽  holo : 전체어
                ⨽  mero : 부분어
                ⨽  rel : 관련어
                ⨽  trans : 영어
                ⨽  rel : 예시
                ⨽  comb_aj : 형용사 결합
                ⨽  comb_n : 명사 결합
                ⨽  comb_v : 동사 결합
                ⨽  sel_rst : 선택 제약
                ⨽  sem_path : 경로
                ⨽  wup_similarity : 유사도
            ⨽sense2
  ⨽ entry 2
  ⨽ ...
~~~

<div style="page-break-after: always;"></div>

### 2.3 한국어 전처리

​	각 문자를 한글, 영어, 한자, 숫자, 기호, 구두점, 연결문자로 판별하는 기능과 입력 한글 문자의 분할, 초/중/종성의 결합, 입력 두 문자의 기능을 추가하였으며 총 **12개의 메소드**를 제공한다.

### 2.4 분류 모델 평가

​	분류를 수행할 수 있는 기계 학습 알고리즘을 만들고 나면, 분류기의 예측력을 검증/평가 한다. 분류 모델 성능을 평가하는 평가 지표 4가지 **(Accuracy, Precision, Recall, F1_score, POS_eval)**를 제공한다.

### 2.5 기계 번역 평가

​	 **BLEU, ROUGE, METEOR, WER/CER, CIDER**는 기계 번역과 같이 문장간의 품질을 측정하는 데 사용되며 평가할 문장 결과의 정확성을 입증하는데 일반적으로 사용하는 평가 지표들이다. 

### 2.6 pos_tag

​	 의미가 있는 최소한의 단위인 형태소로 분리하는 작업인 형태소 분석을 수행하며 관련 함수 총 4가지 함수를 제공하며 품사태거는 nlpnet의 코드를 차용하여 한국어에 맞게 수정하였다.

### 2.7 Translate

​	 파파고를 이용한 번역을 제공한다.

### 2.8 string2string
	
	string2string에 있는 기능을 제공한다.

## 3. 사용 환경

- 운영체제 : ubuntu 18.04, ubuntu 22.04, MacOS
- 언어 : `python3.8`, `python3.9`, `python3.10`, `python3.11`
- 라이브러리 : nltk==1.1.3, numpy==1.23, faiss-cpu=1.7.3   **※ 해당 NLTKo는 영어 NLTK를 포함하고 있음 ※**

**주의사항**
- Espresso5의 EspressoTagger의 사용가능 환경은 다음과 같다.

| OS | python | 아키텍처 |
|----| ------|------|
| Mac | python3.8 | arm64 |
| ubuntu | python3.8 python3.9 python3.10 python3.11 | arm64, x86_64 |




### 3.1 라이브러리 설치

   해당 라이브러리를 설치하기 위해서 아래와 동일하게 명령어 라인에서 입력하여 다운로드 받을 때, 사용자의  2가지  정보가 필요하다. 'modi.changwon.ac.kr' 내 사용하는 **사용자의 ID와 PW를 입력**해주어야만 다운로드가 가능하다.


~~~h
$ git config --global http.sslVerify false
$ pip install git+https://github.com/cwnu-airlab/NLTKo
 
~~~

##### 3.1.1. 설치 도중 오류 발생시 해결 방법
- 만약 ubuntu에서 `pip install`을 진행하였을 때, 오류가 발생하여 제대로 설치가 되지않는다면, 아래의 명령어들을 실행하여 해결할 수 있다.

~~~h
apt update
apt-get install g++
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-cache policy python3.8
apt install python3.8
apt install python3.8-dev
apt-get install python3.8-distutils
apt install git

~~~

- `apt install pythonx.x-dev` (x.x는 사용자의 python 버전에 맞게 입력)
- `apt-get install pythonx.x-distutils` (x.x는 사용자의 python 버전에 맞게 입력)

<div style="page-break-after: always;"></div>

## 4. 실행

### 4.1 토크나이저 (tokenizer)

   nltk의 sent_tokenize(), word_tokenize() 사용 방법과 동일하게 사용가능하며, 2개의 인자가 필요하다. 첫번째 인자는 tokenizer의 입력 텍스트이며 두번째 인자로 "korean"를 입력 하면 각 tokenize의 결과를 리스트 형태로 반환한다. 한국어에서 사용가능한 음절 토크나이저를 추가 하였다.

**사용법 & 결과**

```python
>>> from nltk.tokenize import sent_tokenize, word_tokenize, syllable_tokenize
>>> text="안녕하세요 저는 OOO입니다. 창원대학교에 재학 중입니다."

# 문장 토크나이징
>>> sent_tokenize(text,'korean')
['안녕하세요 저는 OOO입니다.', '창원대학교에 재학 중입니다.']

# 어절 토크나이징
>>> word_tokenize(text,"korean")
['안녕하세요', '저는', 'OOO입니다.', '창원대학교에', '재학', '중입니다.']

# 음절 토크나이징 [default=False]
>>> syllable_tokenize(text,"korean")
['안', '녕', '하', '세', '요', '저', '는', 'O', 'O', 'O', '입', '니', '다', '.', '창', '원', '대', '학', '교', '에', '재', '학', '중', '입', '니', '다', '.']

#음절 토크나이징 (blank 포함)
>>> syllable_tokenize(text,"korean",True)
['안', '녕', '하', '세', '요', ' ', '저', '는', ' ', 'O', 'O', 'O', '입', '니', '다', '.', ' ', '창', '원', '대', '학', '교', '에', ' ', '재', '학', ' ', '중', '입', '니', '다', '.']

```

### 4.2 세종전자사전 (ssem)

   우선 해당 기능을 사용하기 전에 인자 포맷에 대해 설명한다. 인자는 **entrys, entry, sense** 함수에서 사용한다. 인자 포맷을 설명하기 위해 예제는 체언의 '눈'과 용언의 '감다'를 이용하였다. 사전 구조도는 [2.2 세종전자사전 활용](#22-세종전자사전-활용)에서 확인할 수 있다.

<center><b>인자 포맷 : ' 단어(.형태소)((.entry번호).sense번호) '</b></center>

```h
entrys : '단어' 명시
ex) entrys('눈'), entrys('감다')
  
entry : '단어 & 형태소 & entry 번호' 명시
ex) entry('눈.nng_s.1'), entry('감다.vv.1')
  
sense : '단어 & 형태소 & entry 번호 & sense 번호' 명시
ex) sense('눈.nng_s.1.1'), sense('감다.vv.1.01')
```

|   분 류    | 설 명                                                        |
| :--------: | :----------------------------------------------------------- |
|    단어    | 사용자 검색 단어                                             |
|   형태소   | 검색 단어의 형태소 <br>**체언 : nng_s(단일어 명사 : '강'), nng_c(복합어 명사 : '강물') / 용언 : vv,va** |
| entry 번호 | 해당 단어와 형태소의 entry 번호 <br>(entry에 직접 접근을 위해 필요)<br>**체언 , 용언 : 1,2,3,...,n** |
| sense 번호 | 해당 단어와 형태소, entry이하 sense 번호<br>(sense에 직접 접근을 위하여 필요)<br>**체언 : 1,2,3, ...,n (한 자리 숫자로 적용) / 용언 : 01, 02, 03, ...,n (두 자리 숫자로 적용)** |

##### 4.2.1 객체 확인 방법

   filecheck() 함수를 통하여 해당 단어가 세종전자사전의 존재 여부를 확인할 수 있다. 세종전자사전 내 단어가 포함되어 있으면 해당 파일의 위치의 리스트를 반환하고, 포함되지 않는다면 빈 리스트를 반환한다.

```python
>>>from nltk.sejong import ssem

# 세종전자사전 내 단어 포함여부 확인
>>>check=ssem.filecheck('눈')
['./01. 체언_상세/눈.xml']

>>>check=ssem.filecheck('한라봉')
[]
```

##### 4.2.2. entry 접근법

 아래는 entry에 직•간접적으로 접근하기 위한 방법이다. entrys()를 통해 접근 하거나 entry()를 통해 바로 접근이 가능하다.

```python
>>>from nltk.sejong import ssem

# 해당 단어가 가지는 entrys 리스트 반환
>>>entrys=ssem.entrys('눈')
[Entry('눈.nng_s.1'), Entry('눈.nng_s.2'), Entry('눈.nng_s.3'), Entry('눈.nng_s.4'), Entry('눈.nng_s.5')]

# entry 객체에 직접 접근
>>>entry=ssem.entry('눈.nng_s.1')
Entry('눈.nng_s.1')

# entry 객체에 간접 접근
>>>entrys=ssem.entrys('눈')[0]
Entry('눈.nng_s.1')
```

##### 4.2.3. sense 접근법

 아래는 sense에 직•간접적 접근하기 위한 방법이다. senses()를 통해 접근하거나, sense()를 통해 바로 접근이 가능하다. 

```python
>>>from nltk.sejong import ssem

entrys=ssem.entrys('눈')
entry=entrys[0]

# 해당 entry가 가지는 sense리스트 반환
>>>senses=entry.senses()
[Sense('눈.nng_s.1.1'),Sense('눈.nng_s.1.2'),Sense('눈.nng_s.1.3'),..., Sense('눈.nng_s.1.7')]

# sense 객체에 간접 접근
>>> sense=senses[0]
Sense('눈.nng_s.1.1')

# sense 객체에 직접 접근
>>>sense=ssem.sense('눈.nng_s.1.1') #체언
Sense('눈.nng_s.1.1')

>>>sense=ssem.sense('있다.va.1.01')  #용언(형용사)
Sense('있다.va.1.01')

>>>sense=ssem.sense('먹다.vv.1.01')  #용언(동사)
Sense('먹다.vv.1.01')
```

##### 4.2.4. entry 함수 사용법 & 결과

   entry에 직•간접적 접근을 통해 사용할 수 있는 3가지 함수(파생어, 복합어, 파생어)의 결과를 리스트로 반환한다. 

~~~ python
>>> from nltk.sejong import ssem

# entry 객체에 직접 접근
>>> entry=ssem.entry('눈.nng_s.2')

# 해당 entry 객체의 숙어 리스트 반환
>>>entry.idm()
['눈이 오나 비가 오나']

# 해당 entry 객체의 복합어 리스트 반환
>>>entry.comp()
['눈사람', '눈길', '눈송이', '눈싸움', '눈더미', '함박눈', '싸락눈']

# 해당 entry 객체의 파생어 리스트 반환
>>>entry.der()
['눈발']
~~~

##### 4.2.5. sense 함수 사용법 & 결과

  sense의 메소드를 사용하기 위해서는 sense에 직접 접근하는 방법과 그 외 entrys, entry에서 접근하는 방법이 있다.

  아래는 직접 접근하여  sense의 메소드를 알아보는 예이다. 해당 결과가 없다면 빈 리스트를 반환한다. 단어의 sem, 동의어, 반의어, 동위어, 상위어, 하위어, 전체어, 부분어, 관련어, 영어, 예시, 형용사 결합, 명사 결합, 동사 결합, 선택제약, 경로, 유사도 반환 함수 17개를 포함한다. wup_similarity()는 두 단어의 유사도 비교를 위하여 target이 되는 단어가 인자로 들어오게 된다.

```python
from nltk.sejong import ssem

# sense 접근 방법 1 (직접 접근)
>>>sense=ssem.sense('눈.nng_s.1.1')
Sense('눈.nng_s.1.1')

# sense 접근 방법 2 (간접 접근 1)
>>>entrys=ssem.entrys('눈')
[Entry('눈.nng_s.1'), Entry('눈.nng_s.2'), Entry('눈.nng_s.3'), Entry('눈.nng_s.4'), Entry('눈.nng_s.5')]
>>>sense = entrys[0].senses()[0]
Sense('눈.nng_s.1.1')

# sense 접근 방법 3 (간접 접근 2)
>>>sense=ssem.entry('눈.nng_s.1').senses()[0]
Sense('눈.nng_s.1.1')

# 해당 sense의 sem 확인
>>>sense.sem()
['신체부위']

# 해당 sense의 동의어 리스트 반환
>>>sense.syn()
[]

# 해당 sense의 반의어 리스트 반환
>>>sense.ant()
[]

# 해당 sense의 동위어 리스트 반환
>>>sense.coord()
['위', '입', '코', '이마', '머리', '볼']

# 해당 sense의 상위어 리스트 반환
>>>sense.hyper()
['신체부위']

# 해당 sense의 하위어 리스트 반환
>>>sense.hypo()
[]

# 해당 sense의 전체어 리스트 반환
>>>sense.holo()
['얼굴']

# 해당 sense의 부분어 리스트 반환
>>>sense.mero()
['눈동자', '눈시울', '눈망울', '눈꺼풀']  

# 해당 sense의 관련어 리스트 반환
>>>sense.rel()
[]

# 해당 sense의 영어 리스트 반환
>>>sense.trans()
['eye']

# 해당 sense의 예시 리스트 반환
>>>sense.example()
['그는 눈을 크게 뜨고 나를 바라보았다.', '그 아이는 엄마를 닮아 눈이 아주 컸다.']

# 해당 sense의 형용사결합 리스트 반환
>>>sense.comb_aj()
['눈이 아름답다', '눈이 초롱초롱하다', '눈이 아프다', '눈이 맑다', '눈이 크다', '눈이 작다', '눈이 가늘다']

# 해당 sense의 명사결합 리스트 반환
>>>sense.comb_n()
[]

# 해당 sense의 동사결합 리스트 반환
>>>sense.comb_v()
['눈을 뜨다', '눈을 감다', '눈을 흘기다', '눈이 감기다', '눈이 휘둥그레지다']

# 해당 sense의 선택제약 리스트 반환
>>>sense.sel_rst()
[]

# 해당 sense의 상위 경로 리스트 반환
>>>sense.sem_path()
['1_구체물', '1.4_관계구체물', '1.4.1_부분', '1.4.1.2_신체부위']

# semclass 기반의 두 단어 유사도 계산 반환하며, 0~1의 값을 가진다.
# 'target': 유사도를 계산하기 위한 단어
>>>target=ssem.sense('발.nng_s.1.1')
>>>sense.wup_similarity(target)
0.8
```



### 4.3 한국어전처리 (korChar)

 그외 제공하는 한국어 전처리 함수를 설명한다. 총 12가지의 메소드가 있으며 각각 기능에 따라 인자의 개수에 유의해야 한다. 판별 기능의 결과는 True/False로 반환하며 분할, 결합은 문자 튜플과 문자, 비교는 -1,0,1로 반환한다. kor_check()은 모든 한글 자,모음 결합문자를 판별할 수 있지만 kor_syllable()의 경우 **음가 문자**만 판별 가능하므로 주의하여야 한다.

| 함수명                           | 설명                                                         | e.g.                |    결과    |
| :------------------------------- | :----------------------------------------------------------- | ------------------- | :--------: |
| kor_check(char)                  | 입력한 문자가 한글 문자인지 판별                             | ㄱ, ㄲ, ㅜ, 김, ... | True/False |
| kor_split(char)                  | 입력한 문자를 초/중/종성으로 분할                            | 김, ...             | (ㄱ,ㅣ,ㅁ) |
| kor_join(char x, char y, char z) | 입력한 초/중/종성을 하나의 문자로 결합<br>**x: 초성, y: 중성, z: 종성** | (ㄱ,ㅣ,ㅁ), ...     |    '김'    |
| kor_cmp(char x, char y)          | 입력한 두 문자를 비교<br>**x: 비교 대상 문자1, y: 비교 대상 문자2** | x='가'<br>y='라'    |  -1, 0, 1  |
| kor_syllable(char )              | 입력 문자가 한글 음가 문자인지 판별                          | 가, ..., 힣         | True/False |
| hanja_syllable(char)             | 입력 문자가 한자 음가 문자인지 판별                          | 成, 泰, ...         | True/False |
| num_syllable(char)               | 입력 문자가 숫자인지 판별                                    | 1, 2, 3, ...        | True/False |
| eng_syllable(char)               | 입력 문자가 영어 알파벳 문자인지 판별                        | A, a, B, b, ...     | True/False |
| symbol_check(char)               | 입력 문자가 기호인지 판별                                    | ★, !, @, $, ...     | True/False |
| punctuation_check()              | 입력 문자가 구두점인지 판별                                  | ., !, ?, ...        | True/False |
| engConnection_check()            | 입력 문자가 영여 연결 문자인지 판별                          | ., -, _, \|         | True/False |
| numConnection_check()            | 입력 문자가 숫자 연결 문자인지 판별                          | ., , ,              | True/False |

<div style="page-break-after: always;"></div>

**4.3.1. korChar 사용법 & 결과**

~~~python
>>> from nltk import korChar

# 한글 문자 판별
>>> print(korChar.kor_check('ㄲ'))
>>> True

>>> print(korChar.kor_check('A'))
>>> False


# 초/중/종성 분할
>>> print(korChar.kor_split('가'))
>>> ('ㄱ','ㅏ','')


# 초/중/종성 결합
>>> print(korChar.kor_join('ㄲ','ㅜ','ㅁ'))
>>> 꿈


# 한글 비교
>>> print(korChar.kor_cmp('가','라'))
>>> -1

>>> print(korChar.kor_cmp('마','라'))
>>> 1

>>> print(korChar.kor_cmp('라','라'))
>>> 0


# 한글 문자 판별
>>> print(korChar.kor_syllable('ㄱ'))
>>> True

>>> print(korChar.kor_syllable('!'))
>>> False


# 한자 문자 판별
>>> print(korChar.hanja_syllable('韓'))
>>> True

>>> print(korChar.hanja_syllable('ㄲ'))
>>> False


# 숫자 판별
>>> print(korChar.num_syllable('1'))
>>> True

>>> print(korChar.num_syllable('@'))
>>> False


# 영어 알파벳 문자 판별
>>> print(korChar.eng_syllable('a'))
>>> True

>>> print(korChar.eng_syllable('ㄱ'))
>>> False


# 기호 판별
>>> print(korChar.symbol_check('★'))
>>> True

>>> print(korChar.symbol_check('ㄱ'))
>>> False


# 구두점 판별
>>> print(korChar.punctuation_check('.'))
>>> True

>>> print(korChar.punctuation_check('@'))
>>> False


# 영어 알파벳 문자 판별
>>> print(korChar.engConnection_check('_'))
>>> True

>>> print(korChar.engConnection_check('!'))
>>> False


# 숫자 연결 문자 판별
>>> print(korChar.numConnection_check('.'))
>>> True

>>> print(korChar.numConnection_check('!'))
>>> False
~~~

<div style="page-break-after: always;"></div>

### 4.4 분류모델평가

​	분류 모델에 대하여 성능을 평가할 수 있는 대표적인 방법 Accuracy, Precision, Recall, F1_score 사용이 가능하며 이진 데이터와 그 이상의 데이터 모두 사용 가능하다. 입력 형식은 리스트이며 정답과 예측 리스트의 길이가 같아야 한다. Accuracy를 제외하고 평균('micro'/'macro')을 선택할 수 있으며 default는 micro로 설정되어있다.

##### 4.4.1. DefaultMetric

* micro : 전체 값들의 평균
* macro : 각 분류 카테고리 평균들의 평균 
* accuracy : 정확도 반환 [ 전체에서 바르게 분류한 비율 ] 
* precision : 정밀도 반환 [ 정밀도란 모델이 True라고 분류한 것 중 실제 True의 비율 ] 
* recall : 재현율 반환 [ 재현율이란 실제 True인 것 중 모델이 True라고 예측한 것의 비율 ]
* f1_score: Precision과 Recall의 조화평균 값 반환
* pos_eval: 형태소 분석 결과 스코어를 계산하여 값 반환 (예제확인)
* precision@k : precision@k는 k개 추천 결과에 대한 Precision을 계산한 것으로, 모델이 추천한 아이템 k개 중에 실제 사용자가 관심있는 아이템의 비율을 의미
* recall@k : recall@k는 k개 추천 결과에 대한 Recall을 계산한 것으로, 사용자가 관심있는 모든 아이템 중에서 모델의 추천한 아이템 k개가 얼마나 포함되는지 비율을 의미
* hitrate@k : 전체 사용자들의 추천 결과에 대해서 상위 k 안에 선호 아이템이 있는 추천 결과 개수를 전체 사용자의 수로 나누어 평균을 계산한다.

**사용법 & 결과**

```python
from nltk.metrics import DefaultMetric

# 정답 & 예측 데이터 
# y_true=[1,1,1,0,1],  y_pred=[1,0,1,1,0]
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]

# accuracy
>>> DefaultMetric().accuracy_score(y_true,y_pred)
0.3333333333333333

# precision
>>> DefaultMetric().precision_score(y_true, y_pred,'micro')
0.3333333333333333

>>> DefaultMetric().precision_score(y_true, y_pred,'macro')
0.2222222222222222

# recall
>>> DefaultMetric().recall_score(y_true,y_pred,'micro')
0.3333333333333333

>>> DefaultMetric().recall_score(y_true,y_pred,'macro')
0.3333333333333333

# f1_score
>>> DefaultMetric().f1_score(y_true,y_pred,'micro')
0.3333333333333333

>>> DefaultMetric().f1_score(y_true,y_pred,'macro')
0.26666666666666666

>>> y_pred = [5, 2, 4, 1, 3, 2, 5, 6, 7]
>>> y_true = [1, 3, 6, 7, 1, 5]

>>> DefaultMetric().precision_at_k(y_true,  y_pred, 5)
0.8
>>> DefaultMetric().recall_at_k(y_true,y_pred, 5)
0.6666666

>>> user = [[5, 3, 2], [9, 1, 2], [3, 5, 6], [7, 2, 1]]
>>> h_pred = [[15, 6, 21, 3], [15, 77, 23, 14], [51, 23, 21, 2], [53, 2, 1, 5]]

>>> DefaultMetric().hit_rate_at_k(user, h_pred, 3)
0.25

'''
우리	우리/NP	우리/NP
나라에	나라/NNG+에/JKB	나라/NNG+에/JKB
있는	있/VA+는/ETM	있/VA+는/ETM
식물의	식물/NNG+의/JKG	식물/NNG+의/JKG
종수만	종/NNG+수/NNG+만/JX	종/ETM+수/NNB+만/JX
하여도	하/VV+여도/EC	하/VV+여도/EC
수천종이나	수천/NR+종/NNG+이나/JX	수천/XSN+종/NNG+이나/JX
되며	되/VV+며/EC	되/VV+며/EC
그중에는	그중/NNG+에/JKB+는/JX	그중/NNG+에/JKB+는/JX
경제적가치가	경제적가치/NNG+가/JKS	경제적가치/NNG+가/JKS
'''

>>> DefaultMetric().pos_eval(test.txt)
'''
입력 텍스트 파일 형식
: 어절	정답	결과

반환 값
:Eojeol Accuracy, Token precision, Token recall, Token f1
:어절 정확도, 토큰 예측율, 토큰 재현율, 토큰 조화평균

'''
(0.8, 0.8636363636363636, 0.8636363636363636, 0.8636363636363636)
```

##### 4.4.2. MAUVE

개방형 텍스트 생성의 뉴럴 텍스트와 인간 텍스트 비교 측정 지표이다. <br/><br/>
참고 논문 : https://arxiv.org/abs/2102.01454

**주의 사항**
한국어의 경우 p와 q의 문장 개수가 각각 최소 50개 이상이여야 제대로 된 결과가 나옵니다.

* __init__(model_name_or_path: str) -> None : 토크나이징과 임베딩을 진행할 모델을 입력받아 Mauve 클래스를 초기화 한다.
  * model은 현재 huggingface에서 제공되는 모델만 사용가능합니다. 로컬 모델은 안됩니다.
* compute(self,
            p_features=None, q_features=None,
            p_tokens=None, q_tokens=None,
            p_text=None, q_text=None,
            num_buckets='auto', pca_max_data=-1, kmeans_explained_var=0.9,
            kmeans_num_redo=5, kmeans_max_iter=500,
            device_id=-1, max_text_length=1024,
            divergence_curve_discretization_size=25, mauve_scaling_factor=5,
            verbose=False, seed=25, batch_size=1, use_float64=False,
  ) -> SimpleNamespace(mauve, frontier_integral, p_hist, q_hist, divergence_curve)
  * ``p_features``: (n, d) 모양의 ``numpy.ndarray``, 여기서 n은 생성 개수
  * ``q_features``: (n, d) 모양의 ``numpy.ndarray``, 여기서 n은 생성 개수
  * ``p_tokens``: 길이 n의 리스트, 각 항목은 모양 (1, 길이)의 torch.LongTensor
  * ``q_tokens``: 길이 n의 리스트, 각 항목은 모양 (1, 길이)의 torch.LongTensor
  * ``p_text``: 길이가 n인 리스트, 각 항목은 문자열
  * ``q_text``: 길이가 n인 리스트, 각 항목은 문자열
  * ``num_buckets``: P와 Q를 양자화할 히스토그램의 크기, Options: ``'auto'`` (default, n/10를 뜻함) 또는 정수
  * ``pca_max_data``: PCA에 사용할 데이터 포인터의 수, ``-1``이면 모든 데이터를 사용, Default -1
  * ``kmeans_explained_var``: PCA에 의한 차원 축소를 유지하기 위한 데이터 분산의 양, Default 0.9
  * ``kmeans_num_redo``: k-평균 클러스터링을 다시 실행하는 횟수(최상의 목표가 유지됨), Default 5
  * ``kmeans_max_iter``: k-평균 반복의 최대 횟수, Default 500
  * ``device_id``: 기능화를 위한 장치. GPU를 사용하려면 gpu_id(예: 0 또는 3)를 제공, CPU를 사용하려면 -1
  * ``max_text_length``: 고려해야 할 최대 토큰 수, Default 1024
  * ``divergence_curve_discretization_size``: 발산 곡선에서 고려해야 할 점의 수. Default 25.
  * ``mauve_scaling_factor``: 논문의 상수``c`` Default 5
  * ``verbose``: True인 경우 실행 시간 업데이트를 화면에 출력
  * ``seed``: k-평균 클러스터 할당을 초기화하기 위한 무작위 시드
  * ``batch_size``: 특징 추출을 위한 배치 크기

  :return
  * ``out.mauve``, MAUVE 점수인 0에서 1 사이의 숫자, 값이 높을수록 P가 Q에 더 가깝다는 의미
  * ``out.frontier_integral``, 0과 1 사이의 숫자, 값이 낮을수록 P가 Q에 더 가깝다는 것을 의미
  * ``out.p_hist``, P에 대해 얻은 히스토그램, ``out.q_hist``와 동일
  * ``out.divergence_curve``에는 발산 곡선의 점이 포함, 모양은 (m, 2), 여기서 m은 ``divergence_curve_discretization_size``

**사용법 & 결과**

```python
>>> from nltk.metrics import Mauve

>>> p_ko = ['누나가 동생을 등에 업는다.',
'나는 퀴즈의 정답을 맞혔다.',
'입에 고기를 마구 욱여넣었다.',
'마음이 너무 설렌다.', ...

'봄눈이 녹고 있다.',
'길이 얼어 있다.',
] # total 88

>>> q_ko = ['안녕하세요! 오늘은 어떤 날씨인가요?',
'한국 음식 중 어떤 것이 제일 좋아하세요?',
'학교에서 친구들과 함께 공부하는 게 즐거워요.',
'주말에는 가족과 함께 시간을 보내는 것이 좋아요.', ...

'좋아하는 책을 읽으면서 여유로운 주말을 보내는 것이 행복해요.',
'한국의 다양한 음식을 맛보는 것이 여행의 매력 중 하나에요.',
] # total 88

>>> result = Mauve('skt/kobert-base-v1').compute(p_text=p_ko, q_text=q_ko, device_id = 0, max_text_length=256, verbose=False)
>>> print(result.mauve)
0.6333866808068385

```

##### 4.4.3. BERT Score

* __init__(model_name_or_path: str | None = None, lang: str | None = None, num_layers: int | None = None, all_layers: bool = False, use_fast_tokenizer: bool = False, device: str = 'cpu', baseline_path: str | None = None) -> None : BERT Score를 초기화하는 생성자입니다.
	* model_name_or_path : BERT 모델의 이름 또는 경로 (huggingface.co에서 가져옵니다.)
	* lang : BERT 모델의 언어 (kor | eng)
	* num_layers : BERT 모델의 레이어 수
	* device : BERT 모델을 실행할 장치 (cpu | cuda)
* compute(source_sentences: List[str], target_sentences: List[str] | List[List[str]], batch_size: int = 4, idf: bool = False, nthreads: int = 4, return_hash: bool = False, rescale_with_baseline: bool = False, verbose: bool = False) -> dict | str | None : 두 문장의 BERT Score를 계산한다.
* 모델은 huggingface.co에서 다운받습니다. (https://huggingface.co/bert-base-uncased)
	* model_name_or_path 파라미터에는 hunggingface.co/ 뒷부분을 넣어줍니다. `model_name_or_path = 'bert-base-uncased'` (https://huggingface.co/<mark>bert-base-uncased</mark>)


**사용법 & 결과**
~~~python
>>> from nltk.metrics import BERTScore
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = BERTScore(model_name_or_path='skt/kobert-base-v1', lang='kor', num_layers=12).compute([sent1], [sent2])
>>> print(result)
{'precision': array([0.78243864], dtype=float32), 'recall': array([0.78243864], dtype=float32), 'f1': array([0.78243864], dtype=float32)}
~~~


##### 4.4.4. BART Score

* __init__(model_name_or_path='facebook/bart-large-cnn', tokenizer_name_or_path: str | None = None, device: str = 'cpu', max_length=1024) -> None : BART Score를 초기화하는 생성자입니다.
	* model_name_or_path : BART 모델의 이름 또는 경로 (huggingface.co에서 가져옵니다.)
	* device : BART 모델을 실행할 장치 (cpu | cuda)
* compute(source_sentences: List[str], target_sentences: List[str] | List[List[str]], batch_size: int = 4, agg: str = 'mean') -> Dict[str, List[float]] : 두 문장의 BART Score를 계산한다.
* 모델은 huggingface.co에서 다운받습니다. (https://huggingface.co/facebook/bart-large-cnn)
	* model_name_or_path 파라미터에는 hunggingface.co/ 뒷부분을 넣어줍니다. `model_name_or_path = 'facebook/bart-large-cnn'` (https://huggingface.co/<mark>facebook/bart-large-cnn</mark>)

**사용법 & 결과**
~~~python
>>> from nltk.metrics import BARTScore
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = BARTScore().compute([sent1], [sent2])
>>> print(result)
{'score': array([-2.97229409])}
~~~


<div style="page-break-after: always;"></div>

### 4.5 기계번역평가 (StringMetric)

​	문장 간 평가를 위한 방법인 WER/CER, BLEU, ROUGE, CIDER 사용이 가능하다. 각 평가 방법마다 입력 형식이 다르므로 주의하여 사용해야 한다. **각 평가 방법에 대한 자세한 설명과 논문은 부록에서 확인할 수 있다.**



##### 4.5.1. WER/CER

* wer (단어 오류율) : 두 입력 문장 사이의 단어 오류율 반환
* cer  (음절 오류율) : 두 입력 문장 사이의 문자(음절) 오류율 반환 



**(WER/CER)사용법 & 결과**

```python
from nltk.metrics import StringMetric

'''
Args
	reference : str
	hypothesis : str
	

Returns
	scores : flaot
'''

>>> ref="신세계그룹이 인천 SK와이번스 프로야구단을 인수했다"
>>> hyp="신세계 SK와이번스 프로야구단 인수"


# CER
>>> StringMetric().cer(ref,hyp)
0.3333333333333333

# WER
>>> StringMetric().wer(ref,hyp)
0.8
```

<div style="page-break-after: always;"></div>

##### 4.5.2. BLEU

* bleu_n : bleu-n(1,2,3,4) 스코어 반환

  각 n만 고려한 스코어

* bleu  : bleu score 값 반환 (N=4)

  (1~4)-gram을 모두 고려한 스코어이며 일반적인 의미의 bleu 스코어 (0.25,0.25,0.25,0.25)

**BLEU 사용법 & 결과**

```python
>>> from nltk.metrics import StringMetric
'''
Args
	reference : list of str
	candidate : list
	n : int

Returns
	scores : flaot
::
		candidate=[sentence]
	
		multi_reference=[
					ref1 sentence,
					ref2 sentence,
					...]
'''
>>> can=['빛을 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다']
>>> ref=['빛을 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 기회가 훨씬 높았다']

# BLEU-N
>>> StringMetric().bleu_n(ref,can,1)
0.714285714285714

>>> StringMetric().bleu_n(ref,can,2)
0.384615384615385

>>> StringMetric().bleu_n(ref,can,3)
0.166666666666667

>>> StringMetric().bleu_n(ref,can,4)
0.090909090909091

# BLEU_Score
>>> StringMetric().bleu(ref,can)
0.25400289715191
```



##### 4.5.3. ROUGE

​	**※ rouge는 recall based score이며 L, S는 f-measure를 사용하며 N은 recall score이다.**

* rouge-n: rouge-n(1,2,3) 스코어 반환 

  unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표 

* rouge-l : rouge-lcs 스코어 반환

   LCS를 이용하여 최장길이로 매칭되는 문자열을 측정한다. ROUGE-2와 같이 단어들의 연속적 매칭을 요구하지 않고, 문자열 내에서 발생하는 매칭을 측정하기 때문에 유연한 성능비교가 가능

* rouge-s: rouge-s(1,2,3) 스코어 반환

  Window size 내에 위치하는 단어쌍들을 묶어 해당 단어쌍들이 얼마나 중복되어 나타나는지 측정

  

**ROUGE 사용법 & 결과**

```python
from nltk.metrics import StringMetric

'''
Args
	reference : list of str(sentences)
	hypothesis: str (sentences)
	n : int

Returns
	rouge score : flaot
	
::
		hypothesis=hyp summary

		multi_reference=[
					ref1_summary,
					ref2_summary,
					...]
'''


>>> ref_list=["아이폰 앱스토어에 올라와 있는 앱 개발사들은 이제 어느 정보를 가져갈 것인지 정확하게 공지해야함과 동시에 이용자의 승인까지 받아야 합니다."]

>>> hyp="아이폰 앱스토어에 올라와 있는 앱 개발사들은 이제 어느 정보를 가져갈 것인지 공지해야함과 동시에 승인까지 받아야 합니다."


# rouge_n
>>> StringMetric().rouge_n(ref_list,hyp,1)
0.8888888888888888
>>> StringMetric().rouge_n(ref_list,hyp,2)
0.7647058823529411
>>> StringMetric().rouge_n(ref_list,hyp,3)
0.625


# rouge_l
>>> StringMetric().rouge_l(ref_list,hyp)
0.9411764705882353


# rouge_s
>>> StringMetric().rouge_s(ref_list,hyp,1)
0.8064516129032258
>>> StringMetric().rouge_s(ref_list,hyp,2)
0.8222222222222222
>>> StringMetric().rouge_s(ref_list,hyp,3)
0.8275862068965517
```

<div style="page-break-after: always;"></div>

##### 4.5.4. CIDER

* cider : CIDER 스코어 반환

  TF-IDF를 n-gram에 대한 가중치로 계산하고 참조 캡션과 생성 캡션에 대한 유사도를 측정

  

**CIDER 사용법 & 결과**

```python
from nltk.metrics import StringMetric

'''
Args
	reference : list of str(sentences)
	hypothesis: list (sentence)

Returns
	cider score : flaot
	
::
		hypothesis=[hyp sentence]

		multi_reference=[
					ref1_sentence,
					ref2_sentence,
					...]
'''

>>> ref1=['뿔 달린 소 한마리가 초원 위에 서있다']
>>> ref2=['뿔과 긴 머리로 얼굴을 덮은 소 한마리가 초원 위에 있다']

>>> hyp=['긴 머리를 가진 소가 초원 위에 서있다']

# cider (single reference)
>>> StringMetric().cider(ref1,hyp)
0.2404762
>>> StringMetric().cider(ref2,hyp)
0.1091321


>>> ref_list=['뿔 달린 소 한마리가 초원 위에 서있다','뿔과 긴 머리로 얼굴을 덮은 소 한마리가 초원 위에 있다']

# cider (multiple references)
>>> StringMetric().cider(ref_list,hyp)
0.1933312
```

<div style="page-break-after: always;"></div> 

##### 4.5.5. METEOR

* METEOR (Meter For Evaluation of Translation with Explicit Ordering )

  : unigram precision, recall, f-mean을 기반으로 하며, 어간 (Stemming), 동의어 (synonym)일치를 포함한다. 

  : 평가의 기본단위는 문장이다.

* **동의어 추출은 세종의미사전을 활용하였으며 본 라이브러리의 형태소 분석기를 이용하였다.** 

**METEOR 사용법 & 결과**

```python
>>> from nltk.metrics import StringMetric

'''
Args
	reference : list of str(sentences)
	hypothesis: str (sentence)

Returns
	meteor score : flaot
	
::
		hypothesis=hyp sentence

		multi_reference=[
					ref1_sentence,
					ref2_sentence,
					...]
'''


>>>hyp=['봉준호 감독이 아카데미에서 국제영화상을 수상했다.']
>>>ref=['봉준호가 아카데미에서 각본상을 탔다.']

# Meteor (single reference)
>>> StringMetric().meteor(ref,hyp)
0.4536585365853659

>>>hyp=['현재 식량이 매우 부족하다.']
>>>ref=['오늘 매우 양식이 부족하였다.']


>>> StringMetric().meteor(ref,hyp)
0.5645569620253165

>>>hyp=['현재 식량이 매우 부족하다.']
>>>ref=['오늘 양식이 매우 부족하였다.', '오늘 매우 양식이 부족하였다.', '오늘 식량이 매우 부족하였다.']

# Meteor (Multiple reference) : return max score
>>> StringMetric().meteor(ref,hyp)
0.6303797468354431
```

<div style="page-break-after: always;"></div> 

### 4.6. Espresso5

Espresso5 모델을 사용한 tagger를 사용할 수 있다.

* tag(task: str, sentence: str) -> List[str] : 문장의 각 토큰에 대한 task의 태깅 결과를 반환한다.

##### 4.6.1. pos tag

**사용법 & 결과**

~~~python
>>> from nltk.tag import EspressoTagger
>>> sent = "나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger()
>>> print(tagger.tag('pos', sent))
['나_NN', '는_JJ', ' _SP', '아름답_VB', 'ㄴ_EE', ' _SP', '강산_NN', '에_JJ', ' _SP', '살_VB', '고_EE', '있_VB', '다_EE', '._SY']
~~~

##### 4.6.2. dependency parse

**사용법 & 결과**

~~~python
>>> from nltk.tag import EspressoTagger
>>> sent = "나는 배가 고프다. 나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger()
>>> print(tagger.tag('dependency', sent))
[[(1, '나는', '3', 'NP_SBJ'), (2, '배가', '3', 'NP_SBJ'), (3, '고프다', '0', 'VP')], [(1, '나는', '4', 'NP_SBJ'), (2, '아름답ㄴ', '3', 'VP_MOD'), (3, '강산에', '4', 'NP_AJT'), (4, '살고있다', '0', 'VP')]]
~~~

##### 4.6.3. wsd tag

**사용법 & 결과**

~~~python
>>> from nltk.tag import EspressoTagger
>>> sent = "나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger()
>>> print(tagger.tag('wsd', sent))
['나_*', '는_*', '아름답_*', 'ㄴ_*', '강산_01', '에_*', '살_01', '고_*', '있_*', '다_*', '._*']
~~~

##### 4.6.4. ner tag

**사용법 & 결과**

~~~python
>>> from nltk.tag import EspressoTagger
>>> sent = "나는 배가 고프다."

>>> tagger = EspressoTagger()
>>> print(tagger.tag('ner', sent))
['나_*', '는_*', '배_AM-S', '가_*', '고프_*', '다_*', '._*']
~~~

##### 4.6.5 srl tag

**사용법 & 결과**
~~~python
>>> from nltk.tag import EspressoTagger
>>> sent = "나는 배가 고프다. 나는 아름다운 강산에 살고있다."

>>> tagger = EspressoTagger()
>>> print(tagger.tag('srl', sent))
[('ARG0', '나는'), ('ARG1', '배가'), ('ARG0', '나는'), ('ARG1', '강산에')]
~~~


### 4.7. Translate 

파파고 번역기를 이용한 한/영간 번역 기능의 함수이다.

##### 4.7.1. 

* e2k : 영어 ➔ 한국어 변환
* k2e : 한국어 ➔ 영어 변환

**사용법 & 결과**

~~~python
>>> from nltk import trans
>>> papago = trans.papago()
'''

e2k[k2e](sentence(s))

	:: Args
			sentence(s) : list of str
			
	:: Returns 
			translation sentence(s) : list of str
'''

>>> sent_list = ['넷플릭스를 통해 전 세계 동시 방영돼 큰 인기를 끈 드라마 시리즈 ‘오징어 게임’의 인기가 구글 인기 검색어로도 확인됐다.']
>>> papago.k2e(sent_list)
['The popularity of the drama series "Squid Game," which was aired simultaneously around the world through Netflix, has also been confirmed as a popular search term for Google.']

>>> sent_list = ['The popularity of the drama series "Squid Game," which was aired simultaneously around the world through Netflix, has also been confirmed as a popular search term for Google.']
>>> papago.e2k(sent_list)
["넷플릭스를 통해 전 세계에 동시 방송된 드라마 시리즈 '오징어 게임'의 인기가 구글의 인기 검색어로도 확인됐다."]
~~~

### 4.8. 정렬 (alignment)

두 문장의 정렬 결과를 반환하는 함수이다.
해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

##### 4.8.1. Needleman-Wunsch 알고리즘

* get_alignment(str1: str|List[str], str2: str|List[str], return_score_matrix: bool = False) -> Tuple(str|List[str], str|List[str], ndarray|None) : 두 문장의 글로벌 정렬 결과를 반환한다.

**사용법 & 결과**

~~~python
>>> from nltk.alignment import NeedlemanWunsch

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result1, result2 = NeedlemanWunsch().get_alignment(sent1, sent2)
>>> print(result1, '\n', result2)
나 | 는 |   | 학 | 생 | - | 이 | 다 | .
그 | 는 |   | 선 | 생 | 님 | 이 | 다 | .
~~~

##### 4.8.2. Hirschberg 알고리즘

* get_alignment(str1: str|List[str], str2: str|List[str]) -> Tuple(str|List[str], str|List[str]) : 두 문장의 글로벌 정렬 결과를 반환한다.

**사용법 & 결과**
~~~python
>>> from nltk.alignment import Hirschberg

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result1, result2 = Hirschberg().get_alignment(sent1, sent2)
>>> print(result1, '\n', result2)
나 | 는 |   | 학 | 생 | - | 이 | 다 | .
그 | 는 |   | 선 | 생 | 님 | 이 | 다 | .
~~~

##### 4.8.3. Smith-Waterman 알고리즘

* get_alignment(str1: str | List[str], str2: str | List[str], return_score_matrix: bool = False) -> Tuple[str | List[str], str | List[str]] : 두 문장의 로컬 정렬 결과를 반환한다.

**사용법 & 결과**
~~~python
>>> from nltk.alignment import SmithWaterman

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result1, result2 = SmithWaterman().get_alignment(sent1, sent2)
>>> print(f"{result1}\n{result2}")
는 |   | 학 | 생 | - | 이 | 다 | .
는 |   | 선 | 생 | 님 | 이 | 다 | .
~~~

##### 4.8.4. DTW

* get_alignment_path(sequence1: str | List[str] | int | List[int] | float | List[float] | ndarray, sequence2: str | List[str] | int | List[int] | float | List[float] | ndarray, distance='absolute_difference', p_value: int | None = None) -> List[Tuple[int, int]] : DTW 알고리즘을 사용하여 두 시퀀스의 정렬 인덱스를 반환한다.

**사용법 & 결과**
~~~python
>>> from nltk.alignment import DTW

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = DTW().get_alignment_path(sent1, sent2)
>>> print(result)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
~~~

##### 4.8.5. Longest Common Subsequence

* compute(str1: str | List[str], str2: str | List[str], returnCandidates: bool = False) -> Tuple[float, List[str] | List[List[str]]] : 두 문자열의 가장 긴 공통 서브시퀀스를 계산한다.

**사용법 & 결과**
~~~python
>>> from nltk.alignment import LongestCommonSubsequence

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = LongestCommonSubsequence().compute(sent1, sent2)
>>> print(result)
(6.0, None)
~~~

##### 4.8.1.6. Longest Common Substring

* compute(str1: str | List[str], str2: str | List[str], returnCandidates: bool = False) -> Tuple[float, List[str] | List[List[str]]] : 두 문자열의 가장 긴 공통 서브스트링을 계산한다.

**사용법 & 결과**
~~~python
>>> from nltk.alignment import LongestCommonSubstring

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = LongestCommonSubstring().compute(sent1, sent2)
>>> print(result)
(3, None)
~~~

### 4.9. 거리 (Distance)

해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

##### 4.9.1. Levenshtein Edit Distance

* compute(str1: str | List[str], str2: str | List[str], method: str = 'dynamic-programming')→ float : 두 문자열의 Levenshtein 거리를 계산한다.
	* method : 'dynamic-programming' | 'recursive' | 'recursive-memoization'

**사용법 & 결과**
~~~python
>>> from nltk.distance import LevenshteinEditDistance

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = LevenshteinEditDistance().compute(sent1, sent2)
>>> print(result)
3.0
~~~

##### 4.9.2. Hamming Distance

* compute(str1: str | List[str], str2: str | List[str])→ float : 두 문자열의 Hamming 거리를 계산한다.

**사용법 & 결과**
~~~python
>>> from nltk.distance import HammingDistance

>>> sent1 = '나는 학생이었다.'
>>> sent2 = '그는 선생님이다.'

>>> result = HammingDistance().compute(sent1, sent2)
>>> print(result)
4.0
~~~

##### 4.9.3. Damereau-Levenshtein Distance

* compute(str1: str | List[str], str2: str | List[str]) -> float : 두 문자열의 Damereau-Levenshtein 거리를 계산한다.

**사용법 & 결과**
~~~python
>>> from nltk.distance import DamereauLevenshteinDistance

>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = DamereauLevenshteinDistance().compute(sent1, sent2)
>>> print(result)
3.0
~~~

##### 4.9.4. Wasserstein Distance

* compute_kullback(p: np.ndarray | torch.Tensor, q: np.ndarray | torch.Tensor) -> float : 두 Tensor간의 Kullback-Leibler 거리를 계산한다.
* compute_wasserstein(p: np.ndarray | torch.Tensor, q: np.ndarray | torch.Tensor) -> float : 두 Tensor간의 Wasserstein 거리를 계산한다.

**사용법 & 결과**
~~~python
>>> from nltk.distance import WassersteinDistance
>>> import torch

>>> P =  np.array([0.6, 0.1, 0.1, 0.1, 0.1])
>>> Q1 = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
>>> Q2 = np.array([0.1, 0.1, 0.1, 0.1, 0.6])

# Tensor도 입력 가능
>>> P = torch.from_numpy(P) # numpy convert to Tensor
>>> Q1 = torch.from_numpy(Q1) # numpy convert to Tensor
>>> Q2 = torch.from_numpy(Q2) # numpy convert to Tensor

>>> kl_p_q1 = WassersteinDistance().compute_kullback(P, Q1)
>>> kl_p_q2 = WassersteinDistance().compute_kullback(P, Q2)

>>> wass_p_q1 = WassersteinDistance().compute_wasserstein(P, Q1)
>>> wass_p_q2 = WassersteinDistance().compute_wasserstein(P, Q2)

>>> print("\nKullback-Leibler distances: ")
>>> print("P to Q1 : %0.4f " % kl_p_q1)
>>> print("P to Q2 : %0.4f " % kl_p_q2)
Kullback-Leibler distances:
P to Q1 : 1.7918
P to Q2 : 1.7918

>>> print("\nWasserstein distances: ")
>>> print("P to Q1 : %0.4f " % wass_p_q1)
>>> print("P to Q2 : %0.4f " % wass_p_q2)
Wasserstein distances:
P to Q1 : 1.0000
P to Q2 : 2.0000

>>> jesson_p_q1 = WassersteinDistance().compute_jesson_shannon(P, Q1)
>>> jesson_p_q2 = WassersteinDistance().compute_jesson_shannon(P, Q2)

>>> print("\nJesson-Shannon distances: ")
>>> print("P to Q1 : %0.4f " % jesson_p_q1)
>>> print("P to Q2 : %0.4f " % jesson_p_q2)
Jesson-Shannon distances:
P to Q1 : 0.1981
P to Q2 : 0.1981

~~~

### 4.10. 유사도 (Similarity)

해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

##### 4.10.1. 코사인 유사도 (Cosine Similarity)

* compute(x1: Tensor | ndarray, x2: Tensor | ndarray, dim: int = 0, eps: float = 1e-08) -> Tensor | ndarray : 두 벡터의 코사인 유사도를 계산한다.

**사용법 & 결과**
~~~python
>>> from nltk.similarity import CosineSimilarity
>>> import numpy as np

>>> x1 = np.array([1, 2, 3, 4, 5])
>>> x2 = np.array([3, 7, 8, 3, 1])

>>> result = CosineSimilarity().compute(x1, x2)
>>> print(result)
0.6807061638788793
~~~


##### 4.10.2. LCSubstring Similarity

* compute(str1: str | List[str], str2: str | List[str], denominator: str = 'max') -> float : 두 문자열의 LCSubstring 유사도를 계산한다.
	* denominator : (max | sum)

**사용법 & 결과**
~~~python
>>> from nltk.similarity import LCSubstringSimilarity
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = LCSubstringSimilarity().compute(sent1, sent2)
>>> print(result)
0.3333333333333333
~~~

##### 4.10.3. LCSubsequence Similarity

* compute(str1: str | List[str], str2: str | List[str], denominator: str = 'max') -> float : 두 문자열의 LCSubsequence 유사도를 계산한다.
	* denominator : (max | sum)

**사용법 & 결과**
~~~python
>>> from nltk.similarity import LCSubsequenceSimilarity
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = LCSubsequenceSimilarity().compute(sent1, sent2)
>>> print(result)
0.6666666666666666
~~~

##### 4.10.4. Jaro Similarity

* compute(str1: str | List[str], str2: str | List[str])→ float : 두 문자열의 Jaro 유사도를 반환한다.

**사용법 & 결과**
~~~python
>>> from nltk.similarity import JaroSimilarity
>>> sent1 = '나는 학생이다.'
>>> sent2 = '그는 선생님이다.'

>>> result = JaroSimilarity().compute(sent1, sent2)
>>> print(result)
0.8055555555555555
~~~

### 4.11. 검색 (Search)

해당 함수는 sting2sting의 코드를 참고하거나 포함하고 있다. (https://github.com/stanfordnlp/string2string)

##### 4.11.1. Navie Search

* search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

**사용법 & 결과**
~~~python
>>> from nltk.search import NaiveSearch
>>> pattern = "학생"
>>> text = "나는 학생이다."

>>> result = NaiveSearch().search(pattern, text)
>>> print(result)
3
~~~

##### 4.11.2. Rabin-Karp 검색

* search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

**사용법 & 결과**
~~~python
>>> from nltk.search import RabinKarpSearch
>>> pattern = "학생"
>>> text = "나는 학생이다."

>>> result = RabinKarpSearch().search(pattern, text)
>>> print(result)
3
~~~

##### 4.11.3. KMP 검색

* search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

**사용법 & 결과**
~~~python
>>> from nltk.search import KMPSearch
>>> pattern = "학생"
>>> text = "나는 학생이다."

>>> result = KMPSearch().search(pattern, text)
>>> print(result)
3
~~~

##### 4.11.4. Boyer-Moore 검색

* search(pattern: str, text: str) -> int : 텍스트에서 패턴을 검색한다.

**사용법 & 결과**
~~~python
>>> from nltk.search import BoyerMooreSearch
>>> pattern = "학생"
>>> text = "나는 학생이다."

>>> result = BoyerMooreSearch().search(pattern, text)
>>> print(result)
3
~~~

##### 4.11.5. Faiss-Semantic 검색

* __init__(model_name_or_path: str = 'facebook/bart-large', tokenizer_name_or_path: str = 'facebook/bart-large', device: str = 'cpu')→ None : FaissSearh를 초기화 합니다.
* add_faiss_index(column_name: str = 'embeddings', metric_type: int | None = None, batch_size: int = 8, **kwargs)→ None : FAISS index를 dataset에 추가합니다.
* get_embeddings(text: str | List[str], embedding_type: str = 'last_hidden_state', batch_size: int = 8, num_workers: int = 4)→ Tensor : 텍스트를 임베딩합니다.
* get_last_hidden_state(embeddings: Tensor)→ Tensor : 임베딩된 텍스트의 last hidden state를 반환합니다.
* get_mean_pooling(embeddings: Tensor)→ Tensor : 입력 임베딩의 mean pooling을 반환합니다.
* initialize_corpus(corpus: Dict[str, List[str]] | DataFrame | Dataset, section: str = 'text', index_column_name: str = 'embeddings', embedding_type: str = 'last_hidden_state', batch_size: int | None = None, num_workers: int | None = None, save_path: str | None = None)→ Dataset : 데이터셋을 초기화합니다.
* load_dataset_from_json(json_path: str)→ Dataset : json 파일에서 데이터셋을 로드합니다.
* load_faiss_index(index_name: str, file_path: str, device: str = 'cpu')→ None : FAISS index를 로드합니다.
* save_faiss_index(index_name: str, file_path: str)→ None : 특정한 파일 경로로 FAISS index를 저장합니다.
* search(query: str, k: int = 1, index_column_name: str = 'embeddings')→ DataFrame : 데이터셋에서 쿼리를 검색합니다.

**사용법 & 결과**
~~~python
>>> from nltk.search import FaissSearch
>>> faiss = FaissSearch(model_name_or_path = 'facebook/bart-large')
>>> corpus = {
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
>>> faiss.initialize_corpus(corpus=corpus, section='text', embedding_type='mean_pooling')
>>> query = "오늘은 날시가 매우 춥다."
>>> top_k = 5
>>> result = faiss.search(query, top_k)
>>> print(result)
                       text                                         embeddings      score
0          오늘은 날씨가 매우 덥습니다.  [-0.2576247453689575, 0.47791656851768494, -1....  14.051050
1  한국 음식 중에서 떡볶이가 제일 맛있습니다.  [-0.2623925805091858, 0.46345704793930054, -1....  28.752083
2      요즘 드라마를 많이 시청하고 있어요.  [-0.2683958113193512, 0.6801461577415466, -1.1...  29.339230
3    다음 주에 시험이 있어서 공부해야 해요.  [-0.20012563467025757, 0.5758355855941772, -1....  31.358824
4     피아노 연주는 나를 편안하게 해줍니다.  [-0.24231986701488495, 0.6492734551429749, -1....  34.069862
~~~

* faiss 검색을 매번 initialize 하지 않고, 미리 initialize 해놓은 후 검색을 수행할 수 있습니다.

**사용법 & 결과**
~~~python
>>> from nltk.search import FaissSearch

# if you use model and tokenizer in local 
# faiss = FaissSearch(model_name_or_path = '~/test_model/trained_model/', tokenizer_name_or_path = '~/test_model/trained_model/')

>>> faiss = FaissSearch(model_name_or_path = 'facebook/bart-large')
>>> corpus = {
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
>>> faiss.initialize_corpus(corpus=corpus, section='text', embedding_type='mean_pooling', save_path='./test.json')
~~~

- `initialize_corpus()` 메소드 실행시 `save_path`를 지정하면, 해당 경로에 임베딩된 Dataset이 json형식으로 저장됩니다.

~~~python
>>> from nltk.search import FaissSearch

>>> faiss = FaissSearch(model_name_or_path = 'facebook/bart-large')
>>> faiss.load_dataset_from_json('./test.json')
>>> faiss.embedding_type = 'mean_pooling' # initalize_corpus() 메소드 실행시 지정한 embedding_type과 동일하게 지정해야 합니다.
>>> faiss,add_faiss_index(colum_name = 'embeddings')
>>> query = '오늘은 날씨가 매우 춥다.'
>>> top_k = 5
>>> result = faiss.search(query=query, top_k=top_k)
>>> print(result)
Downloading data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 6052.39it/s]
Extracting data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 333.68it/s]
Generating train split: 29 examples [00:00, 2969.89 examples/s]
Adding FAISS index...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 3575.71it/s]
                       text                                         embeddings      score
0          오늘은 날씨가 매우 덥습니다.  [-0.2576247454, 0.4779165685, -1.3404077291, 0...  14.051050
1  한국 음식 중에서 떡볶이가 제일 맛있습니다.  [-0.2623925805, 0.4634570479, -1.1966289282, 0...  28.752083
2      요즘 드라마를 많이 시청하고 있어요.  [-0.2683958113, 0.6801461577, -1.1375769377, 0...  29.339230
3    다음 주에 시험이 있어서 공부해야 해요.  [-0.2001256347, 0.5758355856, -1.0528291464, 0...  31.358824
4     피아노 연주는 나를 편안하게 해줍니다.  [-0.242319867, 0.6492734551, -1.4172941446, 0....  34.069862


~~~



### 4.12 etc

##### 4.12.1. 

* parse_morph : 예제 확인

* TBA

**사용법 & 결과**

~~~python
>>> from nltk import etc

'''

parse_morphs(target)
	* args
			target : 'word1/pos + word2/pos+...'

	* return : [('word1',pos), ('word2',pos)...]
'''

>>> org = '하늘 / NNG + 을 / JKO + 날 / VV + 는 / ETD + 자동차 / NNG'
>>> etc.parse_morph(org)
[('하늘', 'NNG'), ('을', 'JKO'), ('날', 'VV'), ('는', 'ETD'), ('자동차', 'NNG')]


>>> org = '아버지/NNG + 가/JKS + 방/NNG + 에/JKB + 들어가/VV + 신다/EP+EC'
>>> etc.parse_morph(org)
[('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('신다', 'EP')]


~~~



## 5. 사용예제 코드

##### 5.1 세종전자사전 예제 코드

```python
from nltk.sejong import ssem

word = input("input : ")
entrys=ssem.entrys(word)

target = ssem.sense('빵.nng_s.1.1')

for entry in entrys:
	print("\n")
	print("-> @",entry,"@")
	print("-->파생어 : ",entry.der())
	print("-->복합어 : ", entry.comp())
	print("-->숙  어 : ", entry.idm())
	for sense in entry.senses():
		print("\t----> @@",sense,"@@")
		print("\t\t------> s e m : ",sense.sem())
		print("\t\t------> 동의어 : ",sense.syn())		
		print("\t\t------> 반의어 : ",sense.ant())
		print("\t\t------> 동위어 : ",sense.coord())
		print("\t\t------> 상위어 : ",sense.hyper())
		print("\t\t------> 하위어 : ",sense.hypo())
		print("\t\t------> 전체어 : ",sense.holo())
		print("\t\t------> 부분어 : ",sense.mero())
		print("\t\t------> 관련어 : ",sense.rel())
		print("\t\t------> 영  어 : ",sense.trans())
		print("\t\t------> 예  시 : ",sense.example())
		print("\t\t------> 형용사결합 : ",sense.comb_aj())
		print("\t\t------> 명사결합 : ",sense.comb_n())
		print("\t\t------> 동사결합 : ",sense.comb_v())
		print("\t\t------> 선택제약 : ",sense.sel_rst())
		print("\t\t------> 경	로 : ",sense.sem_path()) 
		print("\t\t------> wup유사도: ",sense.wup_similarity(target))
```

<div style="page-break-after: always;"></div>

**세종전자사전 예제 코드 결과**

~~~ h
-> @ Entry('동행.nng_s.1') @
-->파생어 :  ['동행인', '동행자']
-->복합어 :  ['동행하다', '동행시키다']
-->숙  어 :  []
	----> @@ Sense('동행.nng_s.1.1') @@
		------> s e m :  ['대칭적행위']
		------> 동의어 :  []
		------> 반의어 :  []
		------> 동위어 :  []
		------> 상위어 :  []
		------> 하위어 :  []
		------> 전체어 :  []
		------> 부분어 :  []
		------> 관련어 :  []
		------> 영  어 :  ['going together']
		------> 예  시 :  ['경찰서까지 저와 동행을 해 주십시오.']
		------> 형용사결합 :  []
		------> 명사결합 :  []
		------> 동사결합 :  ['동행을 하다', '동행이 있다', '동행이 되다', '동행을 요구하다', '동행을 부탁하다', '동행을 거부하다', '동행을 승낙하다', '동행을 구하다', '동행을 만나다', '동행을 정하다']
		------> 선택제약 :  []
		------> 경	로 :  ['5_사태', '5.2_행위', '5.2.1_물리적행위', '5.2.1.3_대칭적행위']
		------> wup유사도:  0.0
	----> @@ Sense('동행.nng_s.1.2') @@
		------> s e m :  ['행위인간']
		------> 동의어 :  ['일행']
		------> 반의어 :  []
		------> 동위어 :  []
		------> 상위어 :  []
		------> 하위어 :  []
		------> 전체어 :  []
		------> 부분어 :  []
		------> 관련어 :  ['여행']
		------> 영  어 :  ['traveling companion']
		------> 예  시 :  ['우연히 학교 후배가 이번 여행길에 동행이 되었다.']
		------> 형용사결합 :  ['동행이 있다', '동행이 없다']
		------> 명사결합 :  []
		------> 동사결합 :  []
		------> 선택제약 :  []
		------> 경	로 :  ['1_구체물', '1.1_구체자연물', '1.1.2_생물', '1.1.2.5_인간', '1.1.2.5.7_행위인간']
		------> wup유사도:  0.18181818181818182
~~~

<div style="page-break-after: always;"></div>

##### 5.2 한국어 전처리 예제 코드

~~~python
from nltk import korChar

inputSet = ["홝","ㄹ","ㅗ","1","１","韓","A","Ａ","★","."]
print("1. kor_check: ", [korChar.kor_check(x) for x in inputSet])
print("2. kor_split: ", [korChar.kor_split(x) for x in inputSet])
print("3. kor_join: ", [korChar.kor_join(*korChar.kor_split(x)) for x in inputSet])
print("4. kor_cmp: ", [korChar.kor_cmp(x, "가") for x in inputSet])
print("5. kor_syllable: ", [korChar.kor_syllable(x) for x in inputSet])
print("6. hanja_syllable: ", [korChar.hanja_syllable(x) for x in inputSet])
print("7. num_syllable: ", [korChar.num_syllable(x) for x in inputSet])
print("8. eng_syllable: ", [korChar.eng_syllable(x) for x in inputSet])
print("9. symbol_check: ", [korChar.symbol_check(x) for x in inputSet])
print("10. punctuation_check: ", [korChar.punctuation_check(x) for x in inputSet])
print("11. engConnection_check: ", [korChar.engConnection_check(x) for x in inputSet])
print("12. numConnection_check: ", [korChar.numConnection_check(x) for x in inputSet])
~~~

**한국어 전처리 예제 코드 결과**

~~~h
1. kor_check:  [True, True, True, False, False, False, False, False, False, False]
2. kor_split:  [('ㅎ', 'ㅘ', 'ㄺ'), ('ㄹ', '', ''), ('', 'ㅗ', ''), ('1', '', ''), ('１', '', ''), ('韓', '', ''), ('A', '', ''), ('Ａ', '', ''), ('★', '', ''), ('.', '', '')]
3. kor_join:  ['홝', 'ㄹ', 'ㅗ', '1', '１', '韓', 'A', 'Ａ', '★', '.']
4. kor_cmp:  [1, 1, -1, -1, 1, 1, -1, 1, -1, -1]
5. kor_syllable:  [True, False, False, False, False, False, False, False, False, False]
6. hanja_syllable:  [False, False, False, False, False, True, False, False, False, False]
7. num_syllable:  [False, False, False, True, True, False, False, False, False, False]
8. eng_syllable:  [False, False, False, False, False, False, True, True, False, False]
9. symbol_check:  [False, False, False, False, False, False, False, False, True, False]
10. punctuation_check:  [False, False, False, False, False, False, False, False, False, True]
11. engConnection_check:  [False, False, False, False, False, False, False, False, False, True]
12. numConnection_check:  [False, False, False, False, False, False, False, False, False, True]
~~~

<div style="page-break-after: always;"></div>

##### 5.3 분류 모델 평가 예제 코드

~~~python
from nltk import metric

y_true=[0,0,1,1,2,0,1,2]
y_pred=[0,1,0,2,0,0,2,2]

#default로 'micro'설정
print("accuracy : ",metric.accuracy_score(y_true,y_pred))
print("precision: ",metric.precision_score(y_true,y_pred))
print("recall   : ",metric.recall_score(y_true,y_pred))
print("f1_score : ",metric.f1_score(y_true,y_pred))

#'macro'사용
print("precision: ",metric.precision_score(y_true,y_pred,avg="macro"))
print("recall   : ",metric.recall_score(y_true,y_pred,avg="macro"))
print("f1_score : ",metric.f1_score(y_true,y_pred,avg="macro"))
~~~

**분류 모델 평가 예제 코드 결과**

~~~h
accuracy :  0.375
precision:  0.375
recall   :  0.375
f1_score :  0.375

precision:  0.27777777777777773
recall   :  0.38888888888888884
f1_score :  0.32407407407407407
~~~

<div style="page-break-after: always;"></div>

##### 5.4 기계 번역 평가 예제 코드

~~~python
from nltk import StringMetric

can='빛을 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다'
ref='빛을 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 기회가 훨씬 높았다'
# BLEU SCORE
print('bleu-1 : ',StringMetric().bleu_n([ref],[can],1))
print('bleu-2 : ',StringMetric().bleu_n([ref],[can],2))
print('bleu-3 :',StringMetric().bleu_n([ref],[can],3))
print('bleu-4 : ',StringMetric().bleu_n([ref],[can],4))
print('bleu   : ',StringMetric().bleu([ref],[can]))
# WER/CER
print('wer    : ',StringMetric().wer(ref,can))
print('cer    : ',StringMetric().cer(ref,can))
# ROUGE SCORE
print('rouge-n(1) : ',StringMetric().rouge_n([ref],can,1))
print('rouge-n(2) : ',StringMetric().rouge_n([ref],can,2))
print('rouge-n(3) : ',StringMetric().rouge_n([ref],can,3))
print('rouge-l    : ',StringMetric().rouge_l([ref],can))
print('rouge-s(1) : ',StringMetric().rouge_s([ref],can,1))
print('rouge-s(2) : ',StringMetric().rouge_s([ref],can,2))
# CIDER SCORE
print('CIDER : ',StringMetric().cider([ref],[can]))
~~~



**기계 번역 평가 예제 코드 결과**

~~~h
bleu-1 :  0.785714285714286
bleu-2 :  0.538461538461538
bleu-3 : 0.333333333333333
bleu-4 :  0.181818181818182
bleu   :  0.40016016019225

wer    :  0.21428571428571427
cer    :  0.18421052631578946

rouge-n(1) :  0.7857142857142857
rouge-n(2) :  0.5384615384615384
rouge-n(3) :  0.3333333333333333
rouge-l    :  0.7857142857142857
rouge-s(1) :  0.56
rouge-s(2) :  0.5555555555555556 

cider    :  0.4598318
~~~



##### 5.5 Tag 인터페이스 예제 코드

```python
from nltk import pos_tag, nouns, word_segmentor
from nltk.tokenize import syllable_tokenize

sent="오픈소스에 관심 많은 멋진 개발자님들!"

#품사 태깅
tagged=pos_tag(sent,lang='kor')

#명사 추출
nouns_list=nouns(sent)
nouns_list

# 어절 분리
sent="오픈소스에관심많은멋진개발자님들!"
seg=word_segmentor(sent)

print(tagged)
print(nouns_list)
print(seg)
```



**Tag 인터페이스 실행 결과**

~~~h
[('오픈소스', 'NN'), ('에', 'JJ'), (' ', 'SP'), ('관심', 'NN'), (' ', 'SP'), ('많', 'VB'), ('은', 'EE'), (' ', 'SP'), ('멋지', 'VB'), ('ㄴ', 'EE'), (' ', 'SP'), ('개발자', 'NN'), ('님들', 'XN'), ('!', 'SY')]

['오픈소스', '관심', '개발자']

['오픈', '소스에', '관심', '많은', '멋진', '개발자님들']
~~~



<div style="page-break-after: always;"></div>

## 6. 부록 



### 세종전자사전 단어 파일 개수

| 분 류  | 갯 수  |
| :----: | :----: |
|  명사  | 23,017 |
| 형용사 | 3,451  |
|  동사  | 12,102 |
| 합 계  | 38,570 |



### 평가 방법

* **METRIC**

  description : https://modi.changwon.ac.kr/air_cwnu/nlp_tool/nltk_ko/issues/3

* **WER/CER**

  description : https://modi.changwon.ac.kr/air_cwnu/nlp_tool/nltk_ko/issues/4

* **BLEU**

  decription : https://modi.changwon.ac.kr/air_cwnu/nlp_tool/nltk_ko/issues/5

  paper : https://www.aclweb.org/anthology/P02-1040/

* **METEOR**

  description : https://modi.changwon.ac.kr/air_cwnu/nlp_tool/nltk_ko/issues/6

  paper : https://www.aclweb.org/anthology/W05-0909/

* **ROUGE**

  description : https://modi.changwon.ac.kr/air_cwnu/nlp_tool/nltk_ko/issues/7

  paper : https://www.aclweb.org/anthology/W04-1013/

* **CIDER**

  paper : https://ieeexplore.ieee.org/document/7299087

