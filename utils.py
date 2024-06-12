import pandas as pd
import os
import ast
import pymysql
import matplotlib.pyplot as plt
import heapq
import re
from tqdm import tqdm


def long2noise(df):
    # DB의 데이터가 일정 길이 이상이라면 noise 처리하는 코드
    # 마케팅 도메인에서 글이 너무 긴 경우 노이즈로 여기는 경우가 있음
    # 데이터가 노이즈라면, DB의 flag col 의 값을 수정시켰음

    length = int(input('noise 기준 길이 입력: '))

    conn = pymysql.connect()
    cursor = conn.cursor()
    for index, data in tqdm(df.iterrows()):
        if not data['sentences_with_words']:
            continue
        sentences = ast.literal_eval(data['sentences_with_words'])
        for i, sentence in enumerate(sentences):
            if len(sentence) > length:
                sql = 'UPDATE '
                query = str(input('input update query: '))
                sql += query
                cursor.execute(sql)
                conn.commit()
    conn.close()


def str_to_lst(df_series):
    # pandas series 인 df['sentences_with_words']와 같은 문장들을 받아온다.
    # sentences_with_words 의 경우, 문장을 토크나이저로 단어 단위로 자른 데이터로, DB에 저장할 때
    # 2중 리스트 형태의 문자열로 저장된다.
    # 이를 사용하기 편하게 1차원 리스트로 변환하는 함수

    tokenized_data = []
    for words in tqdm(df_series):
        if not words:
            continue
        words = ast.literal_eval(words)
        words = sum(words, [])
        tokenized_data.append(words)
    return tokenized_data


def len_of_tkn_data(tokenized_data, flag):
    # 문서 또는 문장 단위로 tokenized 된 데이터의 최대, 평균 길이를 구하는 함수

    if flag == 0:  # tokenized_data 가 문서 단위 / 문장 통계량
        print('문서 당 최대 문장 양 :', max(len(d) for d in tokenized_data))
        print('문서 당 평균 문장 양 :', sum(map(len, tokenized_data)) / len(tokenized_data))

    elif flag == 1:  # tokenized_data 가 문장 단위 / 단어 통계량
        print('문장 당 최대 단어 양 :', max(len(d) for d in tokenized_data))
        print('문장 당 평균 단어 양 :', sum(map(len, tokenized_data)) / len(tokenized_data))

    plt.hist([len(d) for d in tokenized_data], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show()


def length_check(df):
    # 데이터 통계량 확인 함수

    print("sentences_with_words 파싱 중 . . .")
    tokenized_data = [ast.literal_eval(d) for d in df['sentences_with_words']]

    print('\n# of doc: ', len(tokenized_data))
    len_of_tkn_data(tokenized_data=tokenized_data, flag=0)

    tokenized_data = sum(tokenized_data, [])
    print('\nsent 총 갯수: ', len(tokenized_data))
    len_of_tkn_data(tokenized_data=tokenized_data, flag=1)


def min_values_by_index(arr, index, k):
    # min value 추출 함수(거리 최소 문장 추출할 때 사용)

    if index == -1:
        min_elements = heapq.nsmallest(k, arr)
    else:
        min_elements = heapq.nsmallest(k, arr, key=lambda x: x[index])

    return min_elements


def config_setting():
    # main 에서 코드를 돌리기 전 환경변수 세팅 함수

    try:
        print("\n*** 0. 환경변수, path 설정 ***")
        conn = pymysql.connect()
        query = str(input("사용할 데이터를 가져올 쿼리: "))
        df = pd.read_sql(query, conn)
        print("data read")
        conn.close()

        path = str(input('모델 save, load, 자료 저장할 path 입력(raw data위에 생성): '))
        path = 'C:\\rawdata\\' + path
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created successfully.")
        else:
            print(f"Directory '{path}' already exists.")

        return path, df

    except:
        print('에러 발생')
        print('query와 path를 다시 입력하세요')
        return 0, 0


def contains_only_letters(text):
    # 정규 표현식을 사용하여 문자열에 알파벳만 있는지 확인

    return not bool(re.search(r'[^a-zA-Z가-힣\s]', text))


def contains_only_digits(text):
    # 정규 표현식을 사용하여 문자열에 숫자만 있는지 확인

    return not bool(re.search(r'\D', text))


def contains_only_symbols(text):
    # 정규 표현식을 사용하여 문자열에 기호만 있는지 확인

    return bool(re.search(r'[^a-zA-Z가-힣0-9\s]', text))


def contains_only_numbers_or_symbols(text):
    # 숫자 또는 기호만 있는지 확인

    return contains_only_digits(text) or contains_only_symbols(text)


def sentence_search(df):
    try:
        print("sentences_with_words 파싱 중 . . .")
        while True:
            flag = 0
            opt = str(input('all 옵션은 0 / any 옵션은 1 입력: '))
            tmp = str(input('검색할 문장에 포함될 단어(0==종료): '))
            if tmp == '0':
                break
            else:
                inputs = tmp.split()
            flag_num = int(input('몇개: '))
            for i, sentences_with_words in enumerate(df['sentences_with_words']):
                if flag == flag_num:
                    break
                for j, sentence in enumerate(ast.literal_eval(sentences_with_words)):
                    if opt == '0':
                        if all(inp in sentence for inp in inputs):
                            print(ast.literal_eval(df['sentences'][i])[j], sentence)
                            print(df['url'][i] + '\n')
                            flag += 1
                            break  #
                    elif opt == '1':
                        if any(inp in sentence for inp in inputs):
                            print(ast.literal_eval(df['sentences'][i])[j], sentence)
                            print(df['url'][i] + '\n')
                            flag += 1
                            break
    except:
        print('에러발생')