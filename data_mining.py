import pandas as pd
import heapq
import ast
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from gensim.models.fasttext import FastText
from nltk.corpus import stopwords
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict

from utils import min_values_by_index, contains_only_numbers_or_symbols


def key_sentences(df, model, core_keywords, any_keywords, all_keywords, k1=7, k2=3):
    # 키워드와 sentences간의 거리 계산하고 가장 가까운 k2개 뽑는 함수

    new_df = pd.DataFrame(columns=['key_sentences'])

    for index, data in tqdm(df.iterrows()):
        if not data['sentences_with_words']:
            continue
        sentences = ast.literal_eval(data['sentences_with_words'])

        title_content = data['title'] + data['contents']

        if any_keywords:
            if not(any(item2 in title_content for item2 in any_keywords)):
                continue
        if all_keywords:
            if not(all(item3 in title_content for item3 in all_keywords)):
                continue

        for i, sentence in enumerate(sentences):
            distances = []
            if sentence:
                for word in sentence:
                    distance = []
                    for keyword in core_keywords + any_keywords + all_keywords:
                        distance.append(model.wv.distance(keyword, word))
                    distances.append(sum(distance)/len(distance))
                top_k_distances = min_values_by_index(distances, index=-1, k=k1)
                sentence.insert(0, sum(top_k_distances) / len(top_k_distances))
                sentence.insert(1, ast.literal_eval(data['sentences'])[i])

            else:  # null sentence
                sentence.insert(0, 999)
                sentence.insert(1, data['sentences'][i])

        mini = min_values_by_index(sentences, index=0, k=k2)
        for m in mini:
            m.append(df['url'][index])
        new_df.at[index, 'key_sentences'] = mini

    return new_df


def save_global_key_sentences(output_path, sentences, input_keywords, word_k):
    # document 별 key_sentences 를 global 하게 distance 순서로 정렬하여 저장

    smallest_k = heapq.nsmallest(int(len(sentences) * word_k[1]), sum(sentences, []), key=lambda x: x[0])
    tmp_df = pd.DataFrame(pd.Series(smallest_k))
    tmp_df.index = tmp_df.index + 1
    tmp_df.to_csv(output_path + '/' + str(input_keywords) + '_global_key_sentences.csv', encoding='ANSI', mode='w')
    print(output_path + '/' + str(input_keywords) + '_global_key_sentences.csv' + " 저장 완료")


def get_key_sentences(path, df):
    # key sentences 를 뽑고 global key sentences save 하는 함수

    model = FastText.load(path + '\\model\\fasttext.model')

    done = 1
    while done:
        try:
            done = int(input("프로그램 종료=0, 데이터 추출=1 \n입력: "))
            if done == 0:
                print("프로그램 종료")
                break

            core_keywords = input("핵심 키워드 키워드 리스트(공기청정기 공청기): ").split()
            any_keywords = input("최소(any, 이들 중 하나라도) 있어야 할 키워드 리스트: ").split()
            all_keywords = input("다(all, 이들 중 모두) 있어야 할 키워드 리스트: ").split()

            input_keywords = core_keywords + any_keywords + all_keywords
            word_k = input("키문장을 대표할 단어 수와 content에서 뽑을 키문장의 수를 입력(e.g. 7 3): ")
            word_k = word_k.split()
            print("content별 {0}와(과) 유사한 key sentence {1}개를 단어{2}개를 보고 추출 중 . . .".format(input_keywords, word_k[1],
                                                                                         word_k[0]))
            key_df = key_sentences(df=df, model=model, core_keywords=core_keywords, any_keywords=any_keywords,
                                   all_keywords=all_keywords, k1=int(word_k[0]), k2=int(word_k[1]))
            print("key sentence 추출 완료")
            print("전체 corpus에서 입력 키워드 {0}와(과) 가장 유사한 문장 저장 중 . . .".format(input_keywords))
            print("len(list(key_df[key_df['key_sentences'] != '']['key_sentences'])): ",
                  len(list(key_df[key_df['key_sentences'] != '']['key_sentences'])))
            save_global_key_sentences(path, list(key_df[key_df['key_sentences'] != '']['key_sentences']), input_keywords, word_k)
            print("global key sentences 저장 완료")
        except:
            print('에러 발생\n')


def make_word_cloud(sentences, RESULT_PATH):
    # wc 생성 함수. is_noun=명사만 할지 말지, sw=stopwords 할지 말지

    okt = Okt()
    word_list = []
    word_set_list = []
    stopword = ['원', '것', '하다', '있다', '좋다', '들다', '되다', '이다', '이다', '오다', '받다', '가다', '해주다', '보다', '그래도', '이고', '먼저',
                '여기', '이제', '추다', '라는', '따르다', '하지만', '라고', '경우', '마다', '말다', '판매', '싶다', '서다', '에는', '에서', '원래', '이나',
                'ml',
                '까지', '요즘', '때문', '바로', '가지', '자다', '너무', '라다', '이렇다', '이번', '이에요', '두다', '보이', '인데', '되어다', '그렇다',
                '주다', '정도', '없이', '같이', 'com', 'https', 'naver', 'kr', 'bit', '나다', '다른', '그래서', '대다', '어떻다', '처럼',
                '이렇게', '넘다', '이라고', '없다', '않다', '에게', '이랑', '부터', '으로', '하고', '그로','함께','하고','즉시','부터','에게','부터','에도',
                '그니','지금','같다','하나','어디','위해','서나','그리고','에게','통해','httpst','통해','으로','제외','위해','그로','대한','미리','리다',
                '오직','보고','브리','지다']
    stopwords_eng = set(stopwords.words('english'))
    stopword.extend(stopwords_eng)

    for s in tqdm(sentences):
        data = ast.literal_eval(s)[2:-1]  # key_sentences 의 인덱스0는 score, 인덱스2는 원문

        keyword_stop = [t for t in data if t not in stopword]
        keyword_from_data = [t for t in keyword_stop if (len(t) > 1 and not(contains_only_numbers_or_symbols(t)))]
        word_list.extend(keyword_from_data)
        word_set_list.extend(list(set(keyword_from_data)))

    c1 = Counter(word_list)
    print('len(Counter(word_list): ', len(c1))
    print(list(x for x, y in c1.most_common(200)))
    filtered_c1 = Counter()
    for word, count in c1.items():
        tmp = okt.pos(word)
        if len(tmp) == 1 and tmp[0][1] == 'Josa':
            pass
        else:
            filtered_c1[word] = count
    print('len(Counter(word_list): ', len(filtered_c1))
    print(list(x for x, y in filtered_c1.most_common(200)))

    c2 = Counter(word_set_list)
    print('len(Counter(word_set_list): ', len(c2))
    print(list(x for x, y in c2.most_common(200)))
    filtered_c2 = Counter()
    for word, count in c2.items():
        tmp = okt.pos(word)
        if len(tmp) == 1 and tmp[0][1] == 'Josa':
            pass
        else:
            filtered_c2[word] = count
    print('len(Counter(word_list): ', len(filtered_c2))
    print(list(x for x, y in filtered_c2.most_common(200)))

    except_words = str(input('제외 단어 입력: ')).split()

    c3 = Counter(word_set_list)
    print('len(Counter(word_except_list): ', len(c3))
    print(list(x for x, y in c3.most_common(200)))
    filtered_c3 = Counter()
    for word, count in c3.items():
        tmp = okt.pos(word)
        if (len(tmp) == 1 and tmp[0][1] == 'Josa') or word in except_words:
            pass
        else:
            filtered_c3[word] = count
    print('len(Counter(word_except_list): ', len(filtered_c3))
    print(list(x for x, y in filtered_c3.most_common(200)))

    c4 = Counter(word_set_list)
    print('len(Counter(word_set_except_list): ', len(c4))
    print(list(x for x, y in c4.most_common(200)))
    filtered_c4 = Counter()
    for word, count in c4.items():
        tmp = okt.pos(word)
        if (len(tmp) == 1 and tmp[0][1] == 'Josa') or word in except_words:
            pass
        else:
            filtered_c4[word] = count
    print('len(Counter(word_set_except_list): ', len(filtered_c4))
    print(list(x for x, y in filtered_c4.most_common(200)))

    dic1 = dict(filtered_c1)
    dic2 = dict(filtered_c2)
    dic3 = dict(filtered_c3)
    dic4 = dict(filtered_c4)

    # csv 파일 만들기
    word_count_df = pd.DataFrame(list(dic1.items()),columns=['word','counter'])
    word_count_df = word_count_df.sort_values('counter', ascending = False, ignore_index = True)
    word_count_df.index = word_count_df.index + 1
    word_count_df.to_csv(RESULT_PATH + '/' + "word_count.csv", encoding='ANSI', mode='w')
    # csv 파일 만들기
    word_set_count_df = pd.DataFrame(list(dic2.items()), columns=['word', 'counter'])
    word_set_count_df = word_set_count_df.sort_values('counter', ascending=False, ignore_index=True)
    word_set_count_df.index = word_set_count_df.index + 1
    word_set_count_df.to_csv(RESULT_PATH + '/' + "word_set_count.csv", encoding='ANSI', mode='w')
    # csv 파일 만들기
    word_except_count_df = pd.DataFrame(list(dic3.items()), columns=['word', 'counter'])
    word_except_count_df = word_except_count_df.sort_values('counter', ascending=False, ignore_index=True)
    word_except_count_df.index = word_except_count_df.index + 1
    word_except_count_df.to_csv(RESULT_PATH + '/' + "word_except_count.csv", encoding='ANSI', mode='w')
    # csv 파일 만들기
    word_except_set_count_df = pd.DataFrame(list(dic4.items()), columns=['word', 'counter'])
    word_except_set_count_df = word_except_set_count_df.sort_values('counter', ascending=False, ignore_index=True)
    word_except_set_count_df.index = word_except_set_count_df.index + 1
    word_except_set_count_df.to_csv(RESULT_PATH + '/' + "word_except_set_count.csv", encoding='ANSI', mode='w')

    wc = WordCloud(prefer_horizontal=True, font_path='malgun', width=600, height=400, scale=2.0,
                   background_color='white', max_font_size=250, max_words=300)

    gen = wc.generate_from_frequencies(filtered_c1)
    plt.figure()
    plt.imshow(gen)
    wc.to_file(RESULT_PATH + '/' + 'word_cloud.png')  # 파일이름 설정
    plt.close()

    gen = wc.generate_from_frequencies(filtered_c2)
    plt.figure()
    plt.imshow(gen)
    wc.to_file(RESULT_PATH + '/' + 'word_set_cloud.png')  # 파일이름 설정
    plt.close()

    gen = wc.generate_from_frequencies(filtered_c3)
    plt.figure()
    plt.imshow(gen)
    wc.to_file(RESULT_PATH + '/' + 'word_except_cloud.png')  # 파일이름 설정
    plt.close()

    gen = wc.generate_from_frequencies(filtered_c4)
    plt.figure()
    plt.imshow(gen)
    wc.to_file(RESULT_PATH + '/' + 'word_except_set_cloud.png')  # 파일이름 설정
    plt.close()


def make_keyword_cloud(path):
    try:
        csv_name = str(input('key sentence csv 파일 이름: '))
        key_sentences_path = (path + '\\' + csv_name + ".csv")
        key_df = pd.read_csv(key_sentences_path, encoding='cp949')
        make_word_cloud(key_df['0'], path)

    except:
        print('에러 발생')


def auto_cluster(path):
    model = FastText.load(path + '\\model\\fasttext.model')
    csv_name = str(input('word count csv 파일 이름: '))
    word_count_path = (path + '\\' + csv_name + ".csv")
    wordcloud_words = pd.read_csv(word_count_path, encoding='cp949')
    wordcloud_words = wordcloud_words['word'][:200]
    # 6. 워드클라우드의 단어의 임베딩 벡터 구하기
    words_vectors = np.array(get_words_vectors(model, wordcloud_words))
    std_words_vectors = (words_vectors - np.mean(words_vectors, axis=0))  # /np.std(words_vectors, axis=0)

    # do_pca(std_words_vectors)
    # n_components = float(input('enter n_components ratio(e.g. 0.4): '))
    n_components = 0.4
    if n_components > 1:
        n_components = int(n_components)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(std_words_vectors)
    principalDf = pd.DataFrame(data=principalComponents)
    wordDf = pd.DataFrame(data=wordcloud_words)

    # 8. 클러스터링
    k = int(input("클러스터 개수를 입력(e.g. 30)"))
    res = []
    res_tmp, res_df, avg_score = clustering(wordDf, principalDf, principalComponents, k)
    res.append(res_tmp[0].tolist())

    res_df.to_csv(path + '/' + 'cluster_df.csv', encoding='ANSI', mode='w')

    print("average_score: ", avg_score)
    plot_data = []
    plot_cluster = []
    plot_freq = []
    for r in res:
        for cluster_num in set(r):
            if cluster_num == -1:
                continue
            print("cluster num : {}".format(cluster_num))
            temp_df = res_df[res_df['cluster' + 'of' + str(k)] == cluster_num]  # cluster num 별로 조회
            print(temp_df)
            print(type(temp_df))
            plot_data.append(str(cluster_num))
            plot_cluster.append('')
            plot_freq.append(len(temp_df['word']))
            # cluster_res_df[cluster_num] = temp_df['word'].values
            for d in temp_df['word']:
                plot_data.append(str(d))
                plot_cluster.append(str(cluster_num))
                plot_freq.append(1)


def do_pca(words_vectors):
    pca = PCA().fit(words_vectors)

    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()

    xi = np.arange(1, words_vectors.shape[1] + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, words_vectors.shape[1] + 1,
                         step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.80, color='r', linestyle='-')
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.axhline(y=0.1, color='r', linestyle='-')
    plt.text(0.5, 0.81, '80% cut-off threshold', color='red', fontsize=8)
    plt.text(0.5, 0.51, '50% cut-off threshold', color='red', fontsize=8)
    plt.text(0.5, 0.11, '10% cut-off threshold', color='red', fontsize=8)

    ax.grid(axis='x')
    plt.show()  # 45


# wordDf, principalDf, principalComponents, k
def clustering(df_, tf_idf_df_, tf_idf_, k_):

    plt.plot(figsize=(8 * 1, 8), nrows=1, ncols=1)  # change col
    ind = 0
    result = []
    average_score = []

    model = AgglomerativeClustering(n_clusters=k_)
    clusters = model.fit(tf_idf_df_)
    n_cluster = len(set(clusters.labels_))

    result.append(clusters.labels_)
    df_['cluster' + 'of' + str(k_)] = result[ind]
    score_samples = silhouette_samples(tf_idf_, df_['cluster' + 'of' + str(k_)])
    df_['silhouette_coeff' + 'of' + str(k_)] = score_samples
    silhouette_s = silhouette_score(tf_idf_, df_['cluster' + 'of' + str(k_)])
    temp = 0
    for p in df_.groupby('cluster' + 'of' + str(k_))[
        'silhouette_coeff' + 'of' + str(k_)].mean():
        temp += p
    average_score.append(temp / len(set(clusters.labels_)))

    y_lower = 10

    plt.title(
        'Number of Cluster : ' + str(n_cluster) + '\n' + 'Silhouette Score :' + str(round(silhouette_s, 3)))
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.xlim([-0.1, 1])
    plt.ylim([0, len(tf_idf_df_) + (n_cluster + 1) * 10])
    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
    for j in range(-1, n_cluster - 1):
        ith_cluster_sil_values = score_samples[result[ind] == j]
        ith_cluster_sil_values.sort()

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(j) / n_cluster)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_s, color="red", linestyle="--")
    ind += 1
    plt.show()

    return result, df_, average_score


def get_words_vectors(model, wordcloud_words):
    words_vectors = []
    for w in wordcloud_words:
        words_vectors.append(model.wv.get_vector(w))

    words_vectors = np.array(words_vectors)

    return words_vectors


def keyword_cluster(path):
    try:
        model = FastText.load(path + '\\model\\fasttext.model')

        features = list(map(str, input('클러스터 단어들 입력: ').split()))

        path = str(input('word count 저장된 디렉토리: '))
        csv_name = str(input('word count csv 파일 이름: '))
        word_count_path = ()
        wordcloud_words = pd.read_csv(word_count_path, encoding='cp949')
        wordcloud_words = wordcloud_words['word'][:200]

        cluster_dict = defaultdict(list)
        words_vectors = np.array(get_words_vectors(model, wordcloud_words))
        features_vectors = [model.wv.get_vector(f) for f in features]

        for w, wv in zip(wordcloud_words, words_vectors):
            distances = model.wv.distances(wv, features)
            print(str(w) + '는 ' + str(features[np.argmin(np.array(distances))]) + '에 속합니다')
            cluster_dict[str(features[np.argmin(np.array(distances))])].append(str(w))

        for k in cluster_dict.keys():
            print(k, cluster_dict[k])
    except:
        print('에러발생(사전에 없는 단어를 사용했을 가능성이 높음)')