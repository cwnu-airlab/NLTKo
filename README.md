# Manual [NLTKo][^3]

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

[^3]: https://koreascience.kr/article/CFKO202130060745839.view?orgId=anpor&hide=breadcrumb,journalinfo

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
  + [2.7 Translate](#27-Translate)
* [3. 사용 환경](#3-사용-환경)
  * [3.1 라이브러리 설치](#31-라이브러리-설치)
* [4.실행](#4-실행)
  * [4.1 토크나이저 (tokenizer)](#41-토크나이저-tokenizer)
  * [4.2 세종전자사전 (ssem)](#42-세종전자사전-ssem)
    * [4.2.1 객체 확인 방법](#421-객체-확인-방법 )
    * [4.2.2 entry 접근법](#422-entry-접근법)
    * [4.2.3 sense 접근법](#423-sense-접근법)
    * [4.2.4 entry 함수 사용법 & 결과](#424-entry-함수-사용법--결과)
    * [4.2.5 sense 함수 사용법 & 결과](#425-sense-함수-사용법--결과)
  * [4.3 한국어 전처리 (korchar)](#43-한국어전처리-korchar)
  * [4.4 분류 모델 평가 (metric)](#44-분류모델평가-metric)
  * [4.5 기계 번역 평가 (eval)](#45-기계번역평가-eval)
    * [4.5.1 WER/CER](#451-WER/CER)
    * [4.5.2 BLEU](#452-bleu)
    * [4.5.3 ROUGE](#453-rouge)
    * [4.5.4 CIDER](#454-cider)
    * [4.5.5 METEOR](#455-meteor)
  * [4.6 pos-tag](#46-pos_tag)
  * [4.7 Translate](#47-Translate)
  * [4.8 etc](#48-etc)
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

​	자연어 문서를 분석하기 위해서 긴 문자열을 나누어야 한다. 이 문자열 단위를 토큰 (token) 이라고 하고  문자열을 토큰으로 나누는 작업을 토큰 생성(tokenizing) 이라 한다.  현재 NLTKo에서는 사용자가 분석에 필요한 작업 토큰의 단위에 따라 **문장, 어절, 음절 토크나이징**이 모두 가능하다.

### 2.2 세종전자사전 활용[^1]  

​	세종 전자사전의 단어 XML 파일을 이용하여 **단어의 파생어, 복합어, 숙어, 동의어, 반의어, 동위어, 상위어, 하위어, 전체어, 부분어, 관련어, 영어, 예시, 형용사 결합, 명사 결합, 동사 결합, 선택제약, 경로, 유사도 정보**의 리스트를 반환한다. 아래의 XML tag에 따라 ko_nltk에서 사용가능한 기능들을 구분하였으며, 아래 그림은 해당 entry의 대표 구조도와 같다. 

[^1]: 세종의미사전에 대한 국립국어원의 공개 협의를 마침

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

### 2.6 pos_tag[^2]

​	 의미가 있는 최소한의 단위인 형태소로 분리하는 작업인 형태소 분석을 수행하며 관련 함수 총 4가지 함수를 제공한다.

[^2]: [nlpnet](http://nilc.icmc.usp.br/nlpnet/)의 코드를 차용하여 한국어에 맞게 수정하였다.

### 2.7 Translate

​	 파파고를 이용한 번역을 제공한다.

## 3. 사용 환경

- 운영체제 : Ubuntu 18.04
- 언어 : python3
- 라이브러리 : nltk (version 1.1.4)  **※ 해당 nltk는 영어 nltk를 내포하고 있음 ※**

### 3.1 라이브러리 설치

   해당 라이브러리를 설치하기 위해서 아래와 동일하게 명령어 라인에서 입력하여 다운로드 가능하다.

```h
$ git config --global http.sslVerify false
$ pip install git+https://github.com/cwnu-airlab/NLTKo


Collecting git+https://github.com/cwnu-airlab/NLTKo
  Cloning https://github.com/cwnu-airlab/NLTKo to /tmp/pip-req-build-_u64swgw
  Running command git clone --filter=blob:none -q https://github.com/cwnu-airlab/NLTKo /tmp/pip-req-build-_u64swgw
  Resolved https://github.com/cwnu-airlab/NLTKo to commit 9a95d89ee954af3610a7a61295cd5c1b40a9b33f
  Preparing metadata (setup.py) ... done
Requirement already satisfied: regex==2020.7.14 in /mnt/data4/ghdchlwls123/virtual_env/real/lib/python3.6/site-packages (from nltk==1.1.4) (2020.7.14)
Requirement already satisfied: numpy in /mnt/data4/ghdchlwls123/virtual_env/real/lib/python3.6/site-packages (from nltk==1.1.4) (1.19.5)
... 
Requirement already satisfied: certifi>=2017.4.17 in /mnt/data4/ghdchlwls123/virtual_env/real/lib/python3.6/site-packages (from requests->nltk==1.1.4) (2022.6.15)
Building wheels for collected packages: nltk
  Building wheel for nltk (setup.py) ... done
  Created wheel for nltk: filename=nltk-1.1.4-cp36-cp36m-linux_x86_64.whl size=55523456 sha256=0c8b9d1c9d901d0db7fb7baea5c67a2aea9b1e1d5edc62e9ae32a46bdfe31351
  Stored in directory: /tmp/pip-ephem-wheel-cache-49yua750/wheels/b0/8d/9a/424cab1626f590f8f204c9376833c732276777a79f4906120e
Successfully built nltk
Installing collected packages: nltk
Successfully installed nltk-1.1.4
```

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

### 4.4 분류모델평가 (metric)

​	분류 모델에 대하여 성능을 평가할 수 있는 대표적인 방법 Accuracy, Precision, Recall, F1_score 사용이 가능하며 이진 데이터와 그 이상의 데이터 모두 사용 가능하다. 입력 형식은 리스트이며 정답과 예측 리스트의 길이가 같아야 한다. Accuracy를 제외하고 평균('micro'/'macro')을 선택할 수 있으며 default는 micro로 설정되어있다.

**4.4.1. 전체 개념**

* micro : 전체 값들의 평균
* macro : 각 분류 카테고리 평균들의 평균 
* accuracy : 정확도 반환 [ 전체에서 바르게 분류한 비율 ] 
* precision : 정밀도 반환 [ 정밀도란 모델이 True라고 분류한 것 중 실제 True의 비율 ] 
* recall : 재현율 반환 [ 재현율이란 실제 True인 것 중 모델이 True라고 예측한 것의 비율 ]
* f1_score: Precision과 Recall의 조화평균 값 반환
* pos_eval: 형태소 분석 결과 스코어를 계산하여 값 반환 (예제확인)

**4.4.2. metric.py 사용법 & 결과**

```python
from nltk import metric

# 정답 & 예측 데이터 
# y_true=[1,1,1,0,1],  y_pred=[1,0,1,1,0]
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]

# accuracy
>>> metric.accuracy_score(y_true,y_pred)
0.3333333333333333

# precision
>>> metric.precision_score(y_true, y_pred,'micro')
0.3333333333333333

>>> metric.precision_score(y_true, y_pred,'macro')
0.2222222222222222

# recall
>>> metric.recall_score(y_true,y_pred,'micro')
0.3333333333333333

>>> metric.recall_score(y_true,y_pred,'macro')
0.3333333333333333

# f1_score
>>> metric.f1_score(y_true,y_pred,'micro')
0.3333333333333333

>>> metric.f1_score(y_true,y_pred,'macro')
0.26666666666666666

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

>>> metric.pos_eval(test.txt)
'''
입력 텍스트 파일 형식
: 어절	정답	결과

반환 값
:Eojeol Accuracy, Token precision, Token recall, Token f1
:어절 정확도, 토큰 예측율, 토큰 재현율, 토큰 조화평균

'''
(0.8, 0.8636363636363636, 0.8636363636363636, 0.8636363636363636)
```

<div style="page-break-after: always;"></div>

### 4.5 기계번역평가 (eval)

​	문장 간 평가를 위한 방법인 WER/CER, BLEU, ROUGE, CIDER 사용이 가능하다. 각 평가 방법마다 입력 형식이 다르므로 주의하여 사용해야 한다. **각 평가 방법에 대한 자세한 설명과 논문은 부록에서 확인할 수 있다.**



##### 4.5.1. WER/CER

* wer (단어 오류율) : 두 입력 문장 사이의 단어 오류율 반환
* cer  (음절 오류율) : 두 입력 문장 사이의 문자(음절) 오류율 반환 



**eval.py (WER/CER)사용법 & 결과**

```python
from nltk import eval

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
>>> eval.cer(ref,hyp)
0.3333333333333333

# WER
>>>eval.wer(ref,hyp)
0.8
```

<div style="page-break-after: always;"></div>

##### 4.5.2. BLEU

* bleu_n : bleu-n(1,2,3,4) 스코어 반환

  각 n만 고려한 스코어

* bleu  : bleu score 값 반환 (N=4)

  (1~4)-gram을 모두 고려한 스코어이며 일반적인 의미의 bleu 스코어 (0.25,0.25,0.25,0.25)

**eval.py (BLEU)사용법 & 결과**

```python
>>> from nltk import eval
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
>>> eval.bleu_n(ref,can,1)
0.714285714285714

>>> eval.bleu_n(ref,can,2)
0.384615384615385

>>> eval.bleu_n(ref,can,3)
0.166666666666667

>>> eval.bleu_n(ref,can,4)
0.090909090909091

# BLEU_Score
>>> eval.bleu(ref,can)
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

  

**eval.py (ROUGE)사용법 & 결과**

```python
from nltk import eval

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
>>> eval.rouge_n(ref_list,hyp,1)
0.8888888888888888
>>> eval.rouge_n(ref_list,hyp,2)
0.7647058823529411
>>> eval.rouge_n(ref_list,hyp,3)
0.625


# rouge_l
>>> eval.rouge_l(ref_list,hyp)
0.9411764705882353


# rouge_s
>>> eval.rouge_s(ref_list,hyp,1)
0.8064516129032258
>>> eval.rouge_s(ref_list,hyp,2)
0.8222222222222222
>>> eval.rouge_s(ref_list,hyp,3)
0.8275862068965517
```

<div style="page-break-after: always;"></div>

##### 4.5.4. CIDER

* cider : CIDER 스코어 반환

  TF-IDF를 n-gram에 대한 가중치로 계산하고 참조 캡션과 생성 캡션에 대한 유사도를 측정

  

**eval.py (CIDER)사용법 & 결과**

```python
from nltk import eval

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
>>> eval.cider(ref1,hyp)
0.2404762
>>> eval.cider(ref2,hyp)
0.1091321


>>> ref_list=['뿔 달린 소 한마리가 초원 위에 서있다','뿔과 긴 머리로 얼굴을 덮은 소 한마리가 초원 위에 있다']

# cider (multiple references)
>>> eval.cider(ref_list,hyp)
0.1933312
```

<div style="page-break-after: always;"></div> 

##### 4.5.5. METEOR

* METEOR (Meter For Evaluation of Translation with Explicit Ordering )

  : unigram precision, recall, f-mean을 기반으로 하며, 어간 (Stemming), 동의어 (synonym)일치를 포함한다. 

  : 평가의 기본단위는 문장이다.

* **동의어 추출은 세종의미사전을 활용하였으며 본 라이브러리의 형태소 분석기를 이용하였다.** 

**eval.py (METEOR)사용법 & 결과**

```python
>>> from nltk import eval

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


>>> hyp = '현재 식량이 매우 부족하다.'
>>> ref = ['오늘 매우 양식이 부족하였다.']


# Meteor (single reference)
>>> eval.meteor(ref,hyp)
0.5

>>> hyp='현재 식량이 매우 부족하다.'
>>> ref=['오늘 양식이 매우 부족하였다.', '오늘 매우 양식이 부족하였다.']


# Meteor (Multiple reference) : return max score
>>> eval.meteor(ref,hyp)
0.9921875

>>> hyp='봉준호 감독이 아카데미에서 국제영화상을 수상했다.'
>>> ref=['봉준호가 아카데미에서 각본상을 탔다.']

>>> eval.meteor(ref,hyp)
0.36585365
```

<div style="page-break-after: always;"></div> 

### 4.6 pos_tag 

형태소 분석 결과로, 품사와 같은 추가 정보를 이용하여  문장의 각 토큰에 태그를 지정하는 인터페이스이며 기능은 아래와 같다.

##### 4.6.1. 

* pos_tag : 품사 태깅 결과 반환 (신)
* nouns : 명사 추출 결과 반환 (신)
* pos_tag_with_verb_form : 보조 동사 결합 결과 반환 (구)
* word_segmentor : 구분되지 않은 문장 입력을 분리하여 반환 (구)

**사용법 & 결과**

~~~python
>>> from nltk import pos_tag, nouns, word_segmentor, pos_tag_with_verb_form
>>> from nltk.tokenize import syllable_tokenize
'''
pos_tag(token, tagset=None, lang='eng') //nltk와 함수 공유
	* args
			tokens : syllable_tokens (list)
			tagset : None 
			lang : 'kor' //한국어 품사 태깅		
	* return : pos-tagging (tuple of list)

pos_tag_with_verb_form(sent)
	* args : 문장 (str)
	* return : pos_tag_with_verb_form (list)

nouns(sent)
	* args : 문장 (str)
	* return : 명사 리스트(list)
	
word_segmentor(sent)
	* args : 문장 (str)
	* return : 어절 리스트(list)
'''

>>> sent="오픈소스에 관심 많은 멋진 개발자님들!"

>>> tagged=pos_tag(sent,lang='kor')
>>> tagged
[('오픈', 'NN'), ('소스', 'NN'), ('에', 'JJ'), ('관심', 'NN'), ('많', 'VB'), ('은', 'EE'), ('멋지', 'VB'), ('ㄴ', 'EE'), ('개발자', 'NN'), ('님들', 'XN'), ('!', 'SY')]

>>> nouns_list=nouns(sent)
>>> nouns_list
['오픈', '소스', '관심', '개발자']

>>> sent="오픈소스에관심많은멋진개발자님들!"
>>> seg = word_segmentor(sent)
>>> seg
['오픈', '소스에', '관심', '많은', '멋진', '개발자님들']


>>> sent="우리 다시 사랑하자"
>>> token=syllable_tokenize(sent)

>>> pos_tag(token,lang='kor')
[('우리', 'NN'), ('다시', 'MA'), ('사랑', 'NN'), ('하', 'XV'), ('자', 'EE')]

>>> pos_tag_with_verb_form(sent)
[('우리', 'NN'), ('다시', 'MA'), ('사랑하', 'VB'), ('자', 'EE')]
~~~



### 4.7 Translate

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



### 4.8 etc

##### 4.8.1. 

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
from nltk import eval

can='빛을 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다'
ref='빛을 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 기회가 훨씬 높았다'
# BLEU SCORE
print('bleu-1 : ',eval.bleu_n([ref],[can],1))
print('bleu-2 : ',eval.bleu_n([ref],[can],2))
print('bleu-3 :',eval.bleu_n([ref],[can],3))
print('bleu-4 : ',eval.bleu_n([ref],[can],4))
print('bleu   : ',eval.bleu([ref],[can]))
# WER/CER
print('wer    : ',eval.wer(ref,can))
print('cer    : ',eval.cer(ref,can))
# ROUGE SCORE
print('rouge-n(1) : ',eval.rouge_n([ref],can,1))
print('rouge-n(2) : ',eval.rouge_n([ref],can,2))
print('rouge-n(3) : ',eval.rouge_n([ref],can,3))
print('rouge-l    : ',eval.rouge_l([ref],can))
print('rouge-s(1) : ',eval.rouge_s([ref],can,1))
print('rouge-s(2) : ',eval.rouge_s([ref],can,2))
# CIDER SCORE
print('CIDER : ',eval.cider([ref],[can]))
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



