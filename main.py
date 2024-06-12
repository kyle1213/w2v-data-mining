from utils import config_setting, length_check, long2noise, sentence_search
from model import w2v_model_train, similar_word_search
from data_mining import get_key_sentences, make_keyword_cloud, auto_cluster, keyword_cluster

# matplotlib.use('Agg')


if __name__ == "__main__":
    main_loop = True

    path, df = config_setting()

    while main_loop:
        options = str(input('\n*** 데이터 마이닝 툴 ***\n0. 환경변수, path 변경\n1. 데이터 EDA\n2. 데이터 길이 확인\n'
                            '3. 긴 데이터 노이즈 처리\n4. 워드2벡터 모델 학습\n5. W2V 유사단어 검색\n6. 키 문장 출력\n'
                            '7. 워드 클라우드 생성\n8. 자동 클러스터\n9. 키워드 클러스터\n10. 단어로 데이터 search\n999. 종료\n입력: '))

        if options == '0':  # option 0일 때 오입력 예외처리
            path, df = config_setting()

        elif options == '2':
            print("\n*** 2. 데이터 길이 확인 ***")
            length_check(df)

        elif options == '3':
            print("\n*** 3. 긴 데이터 노이즈 처리 ***")
            long2noise(df)

        elif options == '4':
            print("\n*** 4. 워드2벡터 모델 학습 ***")
            w2v_model_train(path, df)

        elif options == '5':
            print("\n*** 5. W2V 유사단어 검색 ***")
            similar_word_search(path)

        elif options == '6':
            print("\n*** 6. 키 문장 출력 ***")
            get_key_sentences(path, df)

        elif options == '7':
            print("\n*** 7. 워드 클라우드 생성 ***")
            make_keyword_cloud(path)

        elif options == '8':
            print("\n*** 8. 자동 클러스터 ***")
            auto_cluster(path)

        elif options == '9':
            print("\n*** 9. 키워드 클러스터 ***")
            keyword_cluster(path)

        elif options == '10':
            print("\n*** 10. 단어로 데이터 search ***")
            sentence_search(df)

        elif options == '999':
            print("\n*** 999. 프로그램 종료 ***")
            main_loop = False
        else:
            print('invalid option')

