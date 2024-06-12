import os
from gensim.models.fasttext import FastText

from utils import str_to_lst


def model_train_save(tokenized_data_, model_path, vector_size, window_size):
    # 데이터를 fasttext를 사용해서 임베딩 벡터 학습 및 모델 저장 함수

    print("모델 학습 시작")
    model = FastText(sentences=tokenized_data_, vector_size=vector_size, window=window_size, min_count=7, workers=4)
    model.save(model_path)
    print("학습 완료")


def w2v_model_train(path, df):
    if not os.path.exists(path + '/model'):
        os.mkdir(path + '/model')
        print(f"Directory '{path + '/model'}' created successfully.")
    else:
        print(f"Directory '{path + '/model'}' already exists.")

    tokenized_data = str_to_lst(df_series=df['sentences_with_words'])
    model_train_save(tokenized_data, model_path=path + '/model/fasttext.model', vector_size=128, window_size=3)


def model_test(model, keywords, top_n):
    # 학습 된 fasttext 로 키워드와 유사한 단어 n개 print

    print('model.wv.vectors.shape: ', model.wv.vectors.shape)

    for keyword in keywords:
        print(model.wv.most_similar(keyword, topn=top_n))


def similar_word_search(path):
    # 모델로부터 입력 단어와 유사 단어 k개 출력하는 함수

    model = FastText.load(path + '\\model\\fasttext.model')
    done = 1
    while done:
        try:
            done = int(input("프로그램 종료=0, 데이터 추출=1 \n입력: "))
            if done == 0:
                print("프로그램 종료")
                break

            keyword = str(input('키워드를 입력하면 모델의 유사 단어 k개를 출력합니다.(ex. 공기청정기 15): ')).split()
            print(model.wv.most_similar(keyword[0], topn=int(keyword[1])))

        except:
            print('에러 발생\n예상 원인: 오입력 or W2V에 존재하지 않는 단어 사용')
