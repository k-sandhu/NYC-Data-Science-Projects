import datetime
import pandas as pd
import re
import nltk
import numpy as np
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_blobs

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

import xgboost as xgb
from xgboost import XGBClassifier

def get_data(rows=-1):
    #train_file = r"C:\Users\Kamal\Desktop\Bootcamp\Projects\Project 3\data\train.json"
    #predict_file = r"C:\Users\Kamal\Desktop\Bootcamp\Projects\Project 3\data\test.json"

    train_file = "https://storage.googleapis.com/twosigma-renthop/train.json"
    predict_file = "https://storage.googleapis.com/twosigma-renthop/test.json"

    print("Starting to read files")

    print("Reading train....")
    train_df = pd.read_json(train_file)
    train_df['which'] = "train"

    print("Reading test....")
    predict_df = pd.read_json(predict_file)
    predict_df['which'] = "predict"

    if rows != -1:
        train_df = train_df.sample(rows)
        predict_df = predict_df.sample(rows)

    df = pd.concat([train_df, predict_df])

    df = get_dummies_from_features(df)
    df = get_loc_image_setiment(df)
    df = description_quality(df)
    df = missing_address(df)
    df = preprocess_data(df)
    df = nums(df)
    df = words_to_vec(df)
    col_names_groups = col_name_groups(df)

    return df, col_names_groups

def _subparser(x):
    x = x.lower().replace('-', ' ').strip()
    if x[0] == '{':
        return [y.replace('"', '').strip() for y in re.findall(r'(?<=\d\s=\s)([^;]+);', x)]
    x = x.split(u'\u2022')
    return [z for y in x for z in re.split(r'[\.\s!;]!*\s+|\s+-\s+|\s*\*\s*', y)]

def _parser(x):
    return [z for z in [y.strip() for y in _subparser(x)] if len(z) > 0]

def _extract_features(features, feature_parser=lambda x: [x.lower()]):
    return [feature for ft in features for feature in feature_parser(ft)]

def _search_regex(regexes):
    if not isinstance(regexes, list):
        filter_fun = lambda x: re.search(regexes, x) is not None
    else:
        filter_fun = lambda x: re.search(regexes[0], x) is not None and re.search(regexes[1], x) is None
    return lambda x: 1.0 if np.any([filter_fun(ft) for ft in x]) else 0.0

def get_dummies_from_features(df, dtype=np.float32):
    # Clean up features
    FEATURES_MAP = {'elevator': 'elevator',
                    'cats allowed': r'(?<!\w)cats?(?!\w)|(?<!\w)(?<!no )pets?(?!\w)',
                    'dogs allowed': r'(?<!\w)dogs?(?!\w)|(?<!\w)(?<!no )pets?(?!\w)(?!: cats only)',
                    'hardwood floors': 'hardwood',
                    'doorman': r'(?<!virtual )doorman',
                    'dishwasher': 'dishwasher|dw(?!\w)',
                    'laundry': r'laundry(?! is on the blo)',
                    'no fee': 'no fee',
                    'fitness center': r'fitness(?! goals)|gym',
                    'pre war': r'pre\s?war',
                    'roof deck': 'roof',
                    'outdoor space': 'outdoor|garden|patio',
                    'dining room': 'dining',
                    'high speed internet': r'high.*internet',
                    'balcony': r'balcon(y|ies)|private.*terrace',
                    'terrace': 'terrace',
                    'swimming pool': r'pool(?! table)',
                    'new construction': 'new construction',
                    'exclusive': r'exclusive( rental)?$',
                    'loft': r'(?<!sleep )loft(?! bed)',
                    'wheelchair access': 'wheelchair',
                    'simplex': 'simplex',
                    'fireplace': ['fireplace(?! storage)', 'deco'],
                    # looks for first regex, excluding matches of the second regex
                    'lowrise': r'low\s?rise',
                    'garage': r'garage|indoor parking',
                    'reduced fee': r'(reduced|low) fee',
                    'furnished': ['(?<!un)furni', 'deck|inquire|terrace'],
                    'multi level': r'multi\s?level|duplex',
                    'high ceilings': r'(hig?h|tall) .*ceiling',
                    'super': r'(live|site).*super',
                    'parking': r'(?<!street )(?<!side )parking(?! available nearby)',
                    'renovated': 'renovated',
                    'green building': 'green building',
                    'storage': 'storage',
                    'stainless steel appliances': r'stainless.*(appliance|refrigerator)',
                    'concierge': 'concierge',
                    'light': r'(?<!\w)(sun)?light(?!\w)',
                    'exposed brick': 'exposed brick',
                    'eat in kitchen': r'eat.*kitchen',
                    'granite kitchen': 'granite kitchen',
                    'bike room': r'(?<!citi)(?<!citi )bike',
                    'walk in closet': r'walk.*closet',
                    'marble bath': r'marble.*bath',
                    'valet': 'valet',
                    'subway': r'subway|trains?(?!\w)',
                    'lounge': 'lounge',
                    'short term allowed': 'short term',
                    'children\'s playroom': r'(child|kid).*room',
                    'no pets': 'no pets',
                    'central a/c': r'central a|ac central',
                    'luxury building': 'luxur',
                    'view': r'(?<!\w)views?(?!\w)|skyline',
                    'virtual doorman': 'virtual d',
                    'courtyard': 'courtyard',
                    'microwave': 'microwave|mw',
                    'sauna': 'sauna'}

    series = df['features']
    series = series.apply(lambda x: _extract_features(x, _parser))
    dummies = np.zeros((len(series), len(FEATURES_MAP)), dtype=dtype)
    for i, (key, value) in enumerate(FEATURES_MAP.items()):
        dummies[:, i] = series.apply(_search_regex(value))

    dummies = pd.DataFrame(dummies, columns=['feature_' + key for key in FEATURES_MAP.keys()])
    df = pd.concat([df, dummies], axis=1, join_axes=[df.index])
    print("Returning: get_dummies_from_features")
    return df

def get_loc_image_setiment( df):

    #loc_df = pd.read_table(r"C:\Users\Kamal\Desktop\Bootcamp\Projects\Project 3\data\LocationClusters", sep='\s+')
    #train_sent = pd.read_csv(r"C:\Users\Kamal\Desktop\Bootcamp\Projects\Project 3\data\sentiment_train.csv")
    #test_sent = pd.read_csv(r"C:\Users\Kamal\Desktop\Bootcamp\Projects\Project 3\data\sentiment_test.csv")
    #image_data = pd.read_csv(r"C:\Users\Kamal\Desktop\Bootcamp\Projects\Project 3\data\images.csv")

    loc_df = pd.read_table("https://storage.googleapis.com/twosigma-renthop/LocationClusters", sep='\s+')
    train_sent = pd.read_csv("https://storage.googleapis.com/twosigma-renthop/sentiment_train.csv")
    test_sent = pd.read_csv("https://storage.googleapis.com/twosigma-renthop/sentiment_test.csv")
    image_data = pd.read_csv("https://storage.googleapis.com/twosigma-renthop/images.csv")

    print("Method: get_loc_image_setiment")

    df_col = df.columns
    loc_l = loc_df.columns
    loc_l = [f for f in loc_l if f not in df_col]
    loc_l.append('listing_id')
    loc_df = loc_df[loc_l]

    # clean and concatenate sentiment data from train and test into one dataframe
    df_sent = pd.concat([train_sent, test_sent], axis=0)

    # image data
    image_data = image_data.groupby('listing_id').mean()
    image_data = image_data.reset_index()
    image_data['widthScale'] = scale(image_data['width'].tolist())
    image_data['heightScale'] = scale(image_data['height'].tolist())

    df = pd.merge(df, loc_df, on='listing_id', how='left')
    df = pd.merge(df, df_sent, on='listing_id', how='left')
    df = pd.merge(df, image_data, on='listing_id', how='left')
    print("Method: get_loc_image_setiment. Returning Dataframe....")
    return df

def nums(df):
    print("Method: nums. Calculating some numbers....")

    features = df['features'].tolist()
    pictures = df['photos'].tolist()

    numPics = [len(l) for l in pictures]
    numFeatures = [len(l) for l in features]

    df['numPics'] = numPics
    df['numFeatures'] = numFeatures

    df['numPicsScale'] = scale(numPics)
    df['numFeaturesScale'] = scale(numFeatures)

    df['numPicsLog'] = df['numPics'].apply(np.ma.log)
    df['numFeaturesLog'] = df['numFeatures'].apply(np.ma.log)

    print("Method: nums. Returning some numbers....")
    return df

def description_quality(df, buzzwords=[]):
    print("Method: description_quality. Starting to work on descriptions....")
    descriptions = df['description'].tolist()

    numUpper = []
    titleQuality = []
    numUpperPercent = []
    numPara = []
    wordsPerPara = []
    buzzers = []
    numBuzzwordsbyWords = []
    numDesc = []
    desc = []

    buzzwords = list(set(['first month free', 'beautiful',
                          'amazing', 'great', 'luxurious', 'captivating', 'impeccable',
                          'stainless', 'landscaped', 'granite', 'remodel', 'clean',
                          'updated', 'upgraded', 'amenities', 'bright', 'charming',
                          'comfort', 'comfy', 'gorgeous', 'desirable', 'elegant', 'excellent',
                          'exclusive', 'inexpensive', 'low maintenance ', 'security',
                          'quiet', 'conveniently', 'storage space', 'large', 'available immediately',
                          'close to', 'outdoor space', 'bars', 'restaurants', 'cafes', 'shopping',
                          'spacious', 'walking distance', 'entertainment', 'premium', 'peaceful',
                          'modern', 'secure', 'spectacular', 'state-of-the-art', 'stunning',
                          'stylish', 'unique', 'upscale', 'vibrant', 'remodelled', 'lush', 'lovely',
                          'landmark', 'historic', 'graceful', 'gorgeous', 'highly', 'new', 'train',
                          'subway', 'walking distance', 'mint condition', 'public transport', 'gracious',
                          'serene', 'unbelievable', 'stunning']))
    i = 0
    print("Description Number: ")
    for description in descriptions:
        i = i + 1
        if i % 5000 == 0:
            print(i, ", ",)
        try:
            tempNumWords = len(description)
        except TypeError:
            tempNumWords = 0
        numDesc.extend([tempNumWords])  # counts number of words in the description as broken down by tokenizer

        # tokenize
        try:
            tokenizer = nltk.RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(description)
        except TypeError:
            tokens = '__error__'

        # count all caps
        upChars = [token for token in tokens if token.isupper()]
        numUpper.extend([-len(upChars)])

        title = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?A-Z]', description[:100])
        titleQuality.extend([-len(title) / 100])

        # count <br>. count number of paragraphs
        breaks = re.findall('[.](<br />){1,2}[A-Z]*', description)
        numPara.extend([len(breaks)])

        # someWords have wordCombinations with missingSpace between words
        temp_token = []
        for token in tokens:
            cap_it = lambda x: x[0].upper() + x[1:]
            cap_split = re.findall('[A-Z][^A-Z]*', cap_it(token))
            if cap_split != []:
                token = cap_split
                temp_token.extend(token)
            else:
                temp_token.extend(token)
        tokens = temp_token

        # count the number of buzzwords
        tokens = [token.lower() for token in tokens if len(token) > 2]

        sentence = ' '.join(tokens)
        desc.extend(['_'.join(tokens)])

        buzz = 0
        temp_buzzword = []

        for word in buzzwords:
            if word in sentence:
                buzz += 1
                temp_buzzword.extend([word])
        try:
            buzzers.extend(['_'.join(temp_buzzword)])
        except TypeError:
            buzzers.extend([temp_buzzword])

        try:
            numBuzzwordsbyWords.extend([buzz / tempNumWords])
        except ZeroDivisionError:
            numBuzzwordsbyWords.extend([0])

        # % of description as capital letters and words per paragraph
        try:
            wordsPerPara.extend([len(tokens) / len(breaks)])
        except ZeroDivisionError:
            wordsPerPara.extend([len(tokens)])

        try:
            numUpperPercent.extend([len(upChars) / len(tokens)])
        except ZeroDivisionError:
            numUpperPercent.extend([0])

    df['buzzers'] = buzzers
    df['numDesc'] = numDesc
    df['desc'] = desc

    df['numBuzzers'] = df['buzzers'].apply(len)
    df['numBuzzersLog'] = df['numBuzzers'].apply(np.ma.log)

    df['numUpper'] = numUpper
    df['titleQuality'] = titleQuality
    df['numUpperPercent'] = numUpperPercent
    df['numPara'] = numPara
    df['wordsPerPara'] = wordsPerPara
    df['numBuzzwordsbyWords'] = numBuzzwordsbyWords

    df['numUpperLog'] = df['numUpper'].apply(np.ma.log)
    df['numDescLog'] = df['numDesc'].apply(np.ma.log)
    df['titleQualityLog'] = df['titleQuality'].apply(np.ma.log)

    df['numUpperScale'] = scale(numUpper)
    df['titleQualityScale'] = scale(titleQuality)
    df['numUpperPercentScale'] = scale(numUpperPercent)
    df['numParaScale'] = scale(numPara)
    df['wordsPerParaScale'] = scale(wordsPerPara)
    df['numBuzzwordsbyWordsScale'] = scale(numBuzzwordsbyWords)

    print("Method: description_quality. Returning descriptions....")
    return df

def missing_address(df):
    print("Method: missing_address. Dummyfying missing addresses....")
    streetAddressGiven = []
    displayAddressGiven = []

    for rows, index in df.iterrows():
        if index['display_address'] == '':
            displayAddressGiven.extend([0])
        else:
            displayAddressGiven.extend([1])
        if index['street_address'] == '':
            streetAddressGiven.extend([0])
        else:
            streetAddressGiven.extend([1])

    df['displayAddressGiven'] = displayAddressGiven
    df['streetAddressGiven'] = streetAddressGiven

    return df

def preprocess_data(df):
    print("Method: preprocess_data. Preprocessing some things....")

    df['created'] = pd.to_datetime(df['created'])
    df['day'] = df['created'].dt.day
    df['month'] = df['created'].dt.month

    vals = {'low': 0, 'medium': 1, 'high': 2}
    df['interest_level'] = df['interest_level'].map(vals)

    building_id = LabelEncoder()
    manager_id = LabelEncoder()
    street_address = LabelEncoder()
    bedrooms = LabelEncoder()
    bathrooms = LabelEncoder()

    building_id.fit(df['building_id'])
    df['building_id'] = building_id.transform(df['building_id'])

    manager_id.fit(df['manager_id'])
    df['manager_id'] = manager_id.transform(df['manager_id'])

    street_address.fit(df['street_address'])
    df['street_address'] = street_address.transform(df['street_address'])

    bedrooms.fit(df['bedrooms'])
    df['bedrooms'] = bedrooms.transform(df['bedrooms'])

    bathrooms.fit(df['bathrooms'])
    df['bathrooms'] = bathrooms.transform(df['bathrooms'])

    price = df['price'].tolist()

    df['priceScale'] = scale(price)
    df['priceLog'] = np.ma.log(price)

    print("Method: preprocess_data. Done prereprocessing some things....")
    return df

def words_to_vec(df):
    print("Method: words_to_vec. Working on words to vecs....")

    buzzCount = CountVectorizer(stop_words='english', max_features=50, ngram_range=(1, 1), token_pattern=u'.*_.*')
    buzzCount_te_sparse = buzzCount.fit_transform(df["buzzers"])

    buzzTFid = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(2, 9))
    buzzTFid_te_sparse = buzzTFid.fit_transform(df["description"])
    _boltzman = BernoulliRBM(n_components=35)
    _boltzman.fit(buzzTFid_te_sparse)
    buzzTFid_boltzman = _boltzman.transform(buzzTFid_te_sparse)

    buzzCount_df = pd.DataFrame(buzzCount_te_sparse.toarray(), columns=buzzCount.get_feature_names())
    buzzTFid_boltzman_cols = ['buzz_boltz_' + str(ag) for ag in range(1, buzzTFid_boltzman.shape[1] + 1)]
    buzzTFid_boltzman_df = pd.DataFrame(buzzTFid_boltzman, columns=buzzTFid_boltzman_cols)
    df = pd.concat([df, buzzCount_df, buzzTFid_boltzman_df], axis=1)

    #fagg = FeatureAgglomeration(n_clusters=100)
    #fagg.fit(buzzTFid_te_sparse.toarray())
    #buzzTFid_fagg = fagg.transform(buzzTFid_te_sparse.toarray())
    #buzzCount_df = pd.DataFrame(buzzCount_te_sparse.toarray(), columns=buzzCount.get_feature_names())
    #buzzTFid_fagg_cols = ['buzz_fagg' + str(ag) for ag in range(1, buzzTFid_fagg.shape[1] + 1)]
    #buzzTFid_fagg_df = pd.DataFrame(buzzTFid_fagg, columns=buzzTFid_fagg_cols)
    #df = pd.concat([df, buzzTFid_fagg_df], axis=1)

    print("Method: words_to_vec. Returning words to vecs....")
    return df

def col_name_groups(df):

    scale_cols = [c for c in df.columns if re.search('.*Scale', c) != None]
    log_cols = [c for c in df.columns if re.search('.*Log', c) != None]
    feagg_cols = [c for c in df.columns if re.search('.*fagg_.*', c) != None]
    boltz_cols = [c for c in df.columns if re.search('.*bolt.*', c) != None]
    feagg_cols = [c for c in feagg_cols if c not in boltz_cols]

    col_names = {
    "drop_cols" : ['created', 'description', 'display_address', 'photos', 'street_address', 'desc', 'buzzers',
                 'features', 'latitude','longitude'],
    "dummy_cols" : ['manager_id', 'm50', 'm100', 'm150', 'm200', 'm500', 'm1000', 'building_id'],
    "emotion_cols" : ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'negative',
                    'positive'],
    "pic_cols" : ['width', 'height'],
    "num_cols" : ['numDesc', 'numBuzzers', 'numUpper', 'numPara', 'wordsPerPara', 'numBuzzwordsbyWords', 'numPics',
               'numFeatures'],
    "loc_cols" : ['m50', 'm100', 'm150', 'm500', 'm1000'],
    "scale_cols" : scale_cols,
    "log_cols" : log_cols,
    "fagg_cols" : feagg_cols,
    "boltz_cols" : boltz_cols
    }

    return col_names

def naive_bayes(X_train, y_train, X_predict, listing_id_predict):

    n_samples = X_train.shape[0]
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y = make_blobs(n_samples=n_samples, n_features=3, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)

    y[:n_samples // 3] = 0
    y[n_samples // 3:] = 1
    sample_weight = np.random.RandomState(42).rand(y.shape[0])

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X_train, y_train, sample_weight, test_size=0.9, random_state=42)

    clf_NB = GaussianNB()
    clf_NB.fit(X_train, y_train)

    print('Result prior to calibration using CalibratedClassifier')
    print(log_loss(y_test, clf_NB.predict_proba(X_test)))

    # Gaussian Naive-Bayes with sigmoid calibration
    clf_NB_sigmoid = CalibratedClassifierCV(clf_NB, cv=3, method='sigmoid')
    clf_NB_sigmoid.fit(X_train, y_train, sw_train)

    # Predict probabilities for both training and testing dataset
    loss_score = log_loss(y_test, clf_NB_sigmoid.predict_proba(X_test))
    print('Result after calibration using CalibratedClassifier', loss_score)

    predictions_NB = clf_NB_sigmoid.predict_proba(X_predict)
    predictions_NB_df = pd.DataFrame(data=predictions_NB,
                                  index = listing_id_predict,
                                   columns=['Low', 'Medium', 'High'])
    predictions_NB_train = clf_NB_sigmoid.predict_proba(np.append(X_train,X_test, axis=0))


    time = str(datetime.datetime.now().toordinal())
    file_name = time + 'predictions_NB.json'
    predictions_NB_df.to_json(file_name)
    joblib.dump(clf_NB_sigmoid, file_name + 'clf_NB.pkl')

    return predictions_NB_train, predictions_NB, loss_score, clf_NB_sigmoid

def mlp_classifier(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    _MLP = MLPClassifier(activation='relu',
                         alpha=1e-05, batch_size=256,
                         beta_1=0.9, beta_2=0.999, early_stopping=True,
                         epsilon=1e-08, hidden_layer_sizes=(6000, 2800, 60,), learning_rate='adaptive',
                         learning_rate_init=0.001, max_iter=250, momentum=0.9,
                         nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                         solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
                         warm_start=False)

    clf_MLP = _MLP.fit(X_train, y_train)

    dump_results(clf_MLP, X_test, y_test, X_predict, listing_id_predict, model = "mlp_classifier")

def gbc(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    _GBC = GradientBoostingClassifier(loss='deviance', learning_rate=0.01,
                                 n_estimators=100, subsample=1.0,
                                 criterion='friedman_mse', min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 max_depth=3, min_impurity_split=1e-07, init=None,
                                 random_state=None, max_features=None, verbose=1,
                                 max_leaf_nodes=None, warm_start=False, presort='auto')
    clf_GBC = _GBC.fit(X_train, y_train)

    dump_results(clf_GBC, X_test, y_test, X_predict, listing_id_predict, model="gbc_classifier")

def keras(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    _dropout_rate = .1
    input_dim = X_train.shape[1]
    _activation = 'relu'

    # create model
    model_KRS = Sequential()
    model_KRS.add(Dense(6000, input_dim=input_dim, init='uniform', activation=_activation))
    model_KRS.add(Dropout(_dropout_rate))
    model_KRS.add(BatchNormalization())
    model_KRS.add(Dense(2000, activation="tanh"))
    model_KRS.add(Dropout(_dropout_rate))
    model_KRS.add(BatchNormalization())
    model_KRS.add(Dense(3, activation='softmax'))

    model_KRS.compile(loss='categorical_crossentropy', optimizer='adam')

    model_KRS.fit(X_train, y_train, nb_epoch=50, verbose=1,validation_split=0.15)

    dump_results(model_KRS, X_test, y_test, X_predict, listing_id_predict, model="keras_classifier")

def keras_classifier_gridcv(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    _dropout_rate = .1
    input_dim = X_train.shape[1]
    _activation = 'tanh'

    def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
        model_KRS = Sequential()
        model_KRS.add(Dense(7000, input_dim=input_dim, init=init, activation=_activation))
        model_KRS.add(Dropout(_dropout_rate))
        model_KRS.add(BatchNormalization())
        model_KRS.add(Dense(3500, init=init,activation=_activation))
        model_KRS.add(Dropout(_dropout_rate))
        model_KRS.add(BatchNormalization())
        model_KRS.add(Dense(1500, init=init,activation=_activation))
        model_KRS.add(BatchNormalization())
        model_KRS.add(Dense(500, init=init,activation=_activation))
        model_KRS.add(BatchNormalization())
        model_KRS.add(Dense(75, init=init,activation=_activation))
        model_KRS.add(BatchNormalization())
        model_KRS.add(Dense(3, init=init, activation='softmax'))

        model_KRS.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model_KRS

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=1)
    # grid search epochs, batch size and optimizer
    optimizers = ['Nadam', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [256, 512, 1028]
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,verbose=1)
    grid_result = grid.fit(X_train, y_train)

    dump_results(grid_result, X_test, y_test, X_predict, listing_id_predict, model="keras_classifier")

def xgboost_train(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.016  # 0.016
    param['max_depth'] = 6  # 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = ["mlogloss","error"]
    param['min_child_weight'] = 1
    param['subsample'] = 0.75
    param['colsample_bytree'] = 0.75  # 0.75
    param['seed'] = 0
    param['gamma'] = 1
    param['early_stopping_rounds'] = 3,
    #param_list = list(param.items())

    num_rounds = 1650

    d_X_train = xgb.DMatrix(X_train, label=y_train)
    d_X_predict = xgb.DMatrix(X_predict)
    d_X_test = xgb.DMatrix(X_test)
    d_y_test = xgb.DMatrix(y_test)

    xgb_train = xgb.train(param, d_X_train, num_boost_round=num_rounds, verbose_eval=True)

    loss_score = log_loss(y_test, xgb_train.predict(d_X_test))
    print("Loss score for XGB train is:", loss_score)

    prediction_xgb_train = xgb_train.predict(d_X_predict)
    prediction_xgb_train = pd.DataFrame(data=prediction_xgb_train,
                                  index=listing_id_predict,
                                  columns=['Low', 'Medium', 'High'])

    time = str(datetime.datetime.now().toordinal())
    model_name = time + "xgboost_train" + str(loss_score)
    file_name = model_name + '_predictions.json'
    prediction_xgb_train.to_json(file_name)
    xgb_train.dump_model(model_name, with_stats=True)

def xgboost_cv(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.016  # 0.016
    param['max_depth'] = 6  # 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.75
    param['colsample_bytree'] = 0.75  # 0.75
    param['seed'] = 0
    param['gamma'] = 1

    num_rounds = 1650

    dtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(X_predict)
    X_test = xgb.DMatrix(X_test)
    y_test = xgb.DMatrix(y_test)

    xgb_cv = xgb.cv(param, dtrain, num_boost_round=num_rounds, verbose_eval=True)

    time = str(datetime.datetime.now().toordinal())
    model_name = time + "xgboost_cv"
    #file_name = model_name + '_predictions.json'
    #predictions_df.to_json(file_name)
    joblib.dump(xgb_cv, model_name + '.pkl')

    return xgb_cv

def xgboost_classifier(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    X_predict_shape = X_predict.shape

    _XGB = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100,
                         silent=False, objective='multi:softprob', nthread=-1,
                         gamma=1, min_child_weight=1, max_delta_step=0, subsample=.75,
                         colsample_bytree=.75, colsample_bylevel=1, reg_alpha=0,
                         reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

    d_X_train = xgb.DMatrix(X_train.reshape(X_train.shape), label=y_train.reshape(X_train.shape[0]))
    d_X_test = xgb.DMatrix(X_test.reshape(X_test.shape), label=y_test.reshape(X_test.shape[0]))

    _XGB_model = _XGB.fit(d_X_train, y = d_X_train.get_label())

    loss_score = log_loss(d_X_test.get_label(),_XGB.predict_proba(X_test))
    print("Loss score for XGB train is:", loss_score)

    prediction_XGB_model = _XGB.predict_proba(X_predict)
    prediction_XGB_model = pd.DataFrame(data=prediction_XGB_model,
                                  index=listing_id_predict,
                                  columns=['Low', 'Medium', 'High'])

    time = str(datetime.datetime.now().toordinal())
    model_name = time + "xgboost_classifier" + str(loss_score)
    file_name = model_name + '_predictions.json'
    prediction_XGB_model.to_json(file_name)
    joblib.dump(_XGB_model, "xgb_classifier" + '.pkl')

# Still need to write voting classifier and mltend
def xgboost_classifier_grid(X_train, y_train, X_test, y_test, X_predict, listing_id_predict):
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    X_predict_shape = X_predict.shape
    d_X_train = xgb.DMatrix(X_train.reshape(X_train_shape), label=y_train.reshape(X_train_shape[0], 1))
    d_X_predict = xgb.DMatrix(X_predict.reshape(X_predict_shape))
    d_X_test = xgb.DMatrix(X_test.reshape(X_test_shape), label=y_test.reshape(X_test_shape[0], 1))

    _XGB = xgb.XGBClassifier()

    # when in doubt, use xgboost
    param = {'nthread': [24],  # when use hyperthread, xgboost may become slower
                  'objective': ['binary:logistic'],
                  'learning_rate': [0.15],  # so called `eta` value
                  'max_depth': [8],
                  'min_child_weight': [3, 11],
                  'silent': [1],
                  'subsample': [0.9],
                  'colsample_bytree': [0.5],
                  'n_estimators': [300],  # number of trees
                  'seed': [1337]}

    # evaluate with roc_auc_truncated
    def _score_func(estimator, X, y):
        pred_probs = estimator.predict_proba(X)[:, 1]
        loss_score = log_loss(y, pred_probs)
        return loss_score

    # should evaluate by train_eval instead of the full dataset
    clf = GridSearchCV(_XGB, param, n_jobs=-1, cv=2, scoring=_score_func, verbose=1, refit=True)

    _XGB_grid = clf.fit(X_train, y = d_X_train.get_label())

    loss_score = log_loss(d_X_test.get_label(),_XGB_grid.predict_proba(X_test.reshape(X_test_shape)))
    print("Loss score for XGB train is:", loss_score)

    prediction_XGB_model = _XGB_grid.predict_proba(X_predict.reshape(X_predict_shape))
    prediction_XGB_model = pd.DataFrame(data=prediction_XGB_model,
                                  index=listing_id_predict,
                                  columns=['Low', 'Medium', 'High'])

    time = str(datetime.datetime.now().toordinal())
    model_name = time + "xgboost_classifier_grid" + str(loss_score)
    file_name = model_name + '_predictions.json'
    prediction_XGB_model.to_json(file_name)
    joblib.dump(_XGB_grid, "xgb_classifier_grid" + '.pkl')

def dump_results(clf, X_test, y_test, X_predict, listing_id_predict, model):
    loss_score = log_loss(y_test, clf.predict_proba(X_test))
    print(model,'loss score is:', loss_score)

    predictions = clf.predict_proba(X_predict)
    predictions_df = pd.DataFrame(data=predictions,
                                      index=listing_id_predict,
                                      columns=['Low', 'Medium', 'High'])

    time = str(datetime.datetime.now().toordinal())
    model_name = time + model + str(loss_score)
    file_name = model_name +'_predictions.json'
    predictions_df.to_json(file_name)
    joblib.dump(clf, model_name + '.pkl')

def main_ml_wrapper(df, col_name_groups):
    #define the columns that need to be dropped. Partial/manual feature engineering and cleaning
    columns_to_drop = [col_name_groups['drop_cols'],
                       col_name_groups['num_cols'],
                       col_name_groups['log_cols'],['m100', 'm150', 'm500', 'm1000']]
    columns_to_drop = [cols for cols in columns_to_drop for cols in cols]
    df = df.drop(columns_to_drop, axis=1)
    df = df.fillna(0)

    dummyfy = [cols for cols in col_name_groups['dummy_cols'] if cols not in columns_to_drop]
    print("Getting dummies....")
    df = pd.get_dummies(df, columns=dummyfy)

    print("Separate train and predict data")
    df_train = df[df['which'] == 'train']
    df_predict = df[df['which'] == 'predict']

    print("Separate X and Y columns. Create dummy and non-dummy y")
    y_train = df_train['interest_level']
    y_train_dum = pd.get_dummies(df_train['interest_level'])
    X_train = df_train.drop(['which', 'interest_level','listing_id'], axis=1)

    print("Get data that will be used for prediction")
    listing_id_predict = df_predict['listing_id'].tolist()
    X_predict = df_predict.drop(['which', 'interest_level','listing_id'], axis=1)

    print("Converting everything into matrices")
    #X_train = np.asmatrix(X_train, dtype='float32')
    #y_train = np.asmatrix(y_train, dtype='int32').transpose()
    #X_predict = np.asmatrix(X_predict, dtype='float32')

    X_train = scipy.sparse.csr_matrix(X_train.values)
    y_train = scipy.sparse.csr_matrix(y_train.values)
    X_predict = scipy.sparse.csr_matrix(X_predict.values)

    print("Splitting train and test data")
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, train_size=.80)

    #call Naive Bayes and combine predictions with the original results. Keep Original
    print("Calling Gaussian Naive Bayes")
    predictions_NB_train, predictions_NB, loss_score_NB, ml_model_NB = naive_bayes(X_train, y_train, X_predict, listing_id_predict)
    X_train_NB = np.append(X_train, predictions_NB_train, axis =1)
    X_predict_NB = np.append(X_predict, predictions_NB, axis =1)

    print("Splitting the post-Naive Bayes data into train and test")
    X_train_NB_split, X_test_NB_split, y_train_NB_split, y_test_NB_split = train_test_split(X_train_NB, y_train, train_size=.80)

    # call gbm and print results
    print("Calling GBC and sending original data splits to it")
    #gbc(X_train_split, y_train_split, X_test_split, y_test_split, X_predict, listing_id_predict)
    print("Calling GBC and sending post-NB data splits to it")
    #gbc(X_train_NB_split, y_train_NB_split, X_test_NB_split, y_test_NB_split, X_predict_NB, listing_id_predict)

    print("Calling XGBoost Classifier and sending original data splits to it")
    xgboost_classifier(X_train_split, y_train_split, X_test_split, y_test_split, X_predict, listing_id_predict)
    print("Calling XGBoost Classifier and sending post-NB data splits to it")
    xgboost_classifier(X_train_NB_split, y_train_NB_split, X_test_NB_split, y_test_NB_split, X_predict_NB, listing_id_predict)

    #xgboost_classifier_grid(X_train_split, y_train_split, X_test_split, y_test_split, X_predict, listing_id_predict)

    print("Calling XGBoost and sending original data splits to it")
    xgboost_train(X_train_split, y_train_split, X_test_split, y_test_split, X_predict, listing_id_predict)
    print("Calling XGBoost and sending post-NB data splits to it")
    xgboost_train(X_train_NB_split, y_train_NB_split, X_test_NB_split, y_test_NB_split, X_predict_NB,listing_id_predict)

    # call MLP and print results
    print("Calling Multi-Level Preceptron and sending original data splits to it")
    mlp_classifier(X_train_split, y_train_split, X_test_split, y_test_split, X_predict, listing_id_predict)
    print("Calling Multi-Level Preceptron and sending post-NB data splits to it")
    mlp_classifier(X_train_NB_split, y_train_NB_split, X_test_NB_split, y_test_NB_split, X_predict_NB, listing_id_predict)


    #call keras and print results
    print("Calling Keras and sending original data splits to it")
    y_train_dum = np.asmatrix(y_train_dum, dtype='int32')
    X_train_split_KRS, X_test_split_KRS, y_train_split_KRS, y_test_split_KRS = train_test_split(X_train, y_train_dum, train_size=.80)
    keras(X_train_split_KRS, y_train_split_KRS, X_test_split_KRS, y_test_split_KRS, X_predict, listing_id_predict)
    print("Calling Keras and sending post-NB data splits to it")
    X_train_NB_split_KRS, X_test_NB_split_KRS, y_train_NB_split_KRS, y_test_NB_split_KRS = train_test_split(X_train_NB, y_train_dum, train_size=.80)
    keras(X_train_NB_split_KRS, y_train_NB_split_KRS, X_test_NB_split_KRS, y_test_NB_split_KRS, X_predict,listing_id_predict)
    print("Calling Keras with GridSearchCV and sending original data splits to it")
    keras_classifier_gridcv(X_train_split_KRS, y_train_split_KRS, X_test_split_KRS, y_test_split_KRS, X_predict, listing_id_predict)
    


    print("Calling XGBoost with CV and sending post-NB data splits to it")
    xgboost_cv(X_train_NB_split, y_train_NB_split, X_test_NB_split, y_test_NB_split, X_predict_NB, listing_id_predict)
    print("Calling XGBoost with CV and sending original data splits to it")
    xgboost_cv(X_train_split, y_train_split, X_test_split, y_test_split, X_predict, listing_id_predict)

if __name__ == '__main__':
    df, col_name_groups = get_data()
    main_ml_wrapper(df, col_name_groups)
