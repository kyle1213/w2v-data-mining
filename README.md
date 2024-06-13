## 워드2벡터를 사용한 데이터 단어 임베딩 프로젝트  
임베딩 된 모델을 사용해서 키워드와 연관도가 높은 문장을 추출  
추출된 문장들의 워드 클라우드를 나타내고, 해당 단어들을 임베딩 벡터로 클러스터링

1. 데이터 수집(ex. 런닝화)
2. Okt 토크나이저로 데이터 처리
3. 문장 별로 토크나이즈 된 데이터들을 FastText(W2V)로 학습 + 학습된 임베딩 모델로 단어를 입력했을 때 유사한 다른 단어를 확인할 수 있음
4. 프로그램에서 원하는 키워드 입력(여러 개 가능, ex.나이키, 런닝화..)
5. 학습된 임베딩 모델로 각 데이터(문서 or 글)에서 k1 개 만큼의 문장을 key sentence로 선별 + 각 문장의 단어들을 키워드와 비교하여 가장 유사도가 높은 단어 k2 개의 유사도 평균을 문장과 키워드의 유사도로 사용
6. 데이터 별 key sentences를 사용하여 워드클라우드 생성
7. 워드 클라우드에 나타난(빈도수가 높은) 단어들의 임베딩 벡터를 PCA를 거쳐 k-clustering 을 거쳐 유사 단어 군집 확인

얻을 수 있는 결과
- key sentences에 대한 워드 클라우드
- 데이터로 학습된 w2v 모델
- 워드 클라우드의 단어에 대한 클러스터
- key sentencces

학습된 모델로 추출된 key sentences with score  
![](https://github.com/kyle1213/w2v-data-mining/blob/master/imgs/key_sentences.png)

key sentences로 생성된 wordcloud  
![](https://github.com/kyle1213/w2v-data-mining/blob/master/imgs/word_cloud.png)

wordcloud의 단어와 임베딩 모델을 사용한 클러스터 결과  
![](https://github.com/kyle1213/w2v-data-mining/blob/master/imgs/cluster.png)
