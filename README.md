## 워드2벡터를 사용한 데이터 단어 임베딩 프로젝트  
gpu 서버나 데이터 라벨없이 글의 key sentence를 뽑는 프로젝트
임베딩 된 모델을 사용해서 키워드와 연관도가 높은 문장을 추출  
추가로 추출된 문장들의 워드 클라우드를 나타내고, 해당 단어들을 임베딩 벡터로 클러스터링

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


## 결과 및 해석
직접 구성한 토크나이저가 아닌 다른 오픈소스의 토크나이저를 사용했기 때문에 단어 토큰들이 정확하게 나누어지지 않았다(고유 명사 등).  
이로 인해 보고자 하던 단어(브랜드명, 제품명처럼)가 워드클라우드에 잡히지 않거나 임베딩의 유사 단어를 확인할 때, 즉 늦게 이를 알아차리면 사전에 해당 단어를 추가하고 다시 토크나이징을 하는 문제가 있다.  

key sentence 추출 과정을 통해 raw 데이터에 워드클라우드를 진행한 것 보다 훨씬 노이즈가 적고 중요한 정보가 워드클라우드에 나올 수 있었다.
또한 key sentence 추출로 document를 대표할 문장을 뽑을 수 있어 데이터를 효율적으로 저장할 수 있었다.  

물론 key sentence를 key word와 비교하여 특정 threshold를 넘는 문장을 고르는 방식이 아닌(도메인마다 threshold가 달리질테고, 한 도메인에서도 이를 정하느 것은 어려움), 가장 유사도가 높은 k1 개(ex. 3 개)만큼 출력하게 구현을 했다.  
이로 인해 key sentence 여도 뽑히지 못하는 경우나, key sentence는 이미 다 뽑았는데 상관없는 다른 문장을 초과해서 추출하는 경우도 있었다.(결과만 봤을 때 큰 악영향은 없지만, 개선할 부분이라고 생각한다)  

main 목표인 key word와 유사한 key sentence 추출을 위해 학습된 w2v 모델(FastText)을 사용해 키워드 뿐 아니라 다른 단어와 유사한 단어도 확인할 수 있었다.  

key sentence 추출로 정제된 데이터에 대해 워드클라우드를 진행하여 더 나은 결과를 볼 수 있었고, 해당 단어들과 임베딩 모델을 활용해 상위 단어들의 클러스터를 볼 수 있었다.  
비슷한 성격의 단어들이 cluster되는 모습을 보고 어떤 특징이 있는지 알 수 있었다.
