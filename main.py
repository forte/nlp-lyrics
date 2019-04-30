import csv
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from wordcloud import WordCloud


class NLPLyrics:
    def __init__(self):
        # Create stoplist
        self.stoplist = stopwords.words('english')
        preset_steplist_upper =  [token.capitalize() for token in stopwords.words('english')]
        self.stoplist.extend(preset_steplist_upper)
        additional_stoplist = ['.', '!', '\'s', '?', ';', 'n\'t', '\'ll', 'would', '--', 'ta', 'wan', 'ai',
                               'na', 'ya', 'could', 'It', 'am', '\'m', ',', '\'', '\'re', 'u', '``', '\'\'',
                               'wa', 'ca', '\'em', '...', ':', 'em', 'wit', 'wo', 'ya', 'gon', 'y\'all', 
                               '\'ve', 'im', '\'cause', 'cause', '\'d', '-' 'ha', 'un']
        self.stoplist.extend(additional_stoplist)

        # Create word lemmatizer
        self.lemmatizer = WordNetLemmatizer()


    def normalize_data(self):
        genres = ['pop', 'rock', 'hiphop']
        songs = []

        for genre in genres:
            with open('lyrics_' + genre + '_final.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
                
                for num, record in enumerate(csv_reader): 
                    # logging purposes to check progress
                    if num%1000 == 0:
                        print(genre + ' - ' + str(num))

                    lyrics = self.clean_text(record[5])
                    tokens = nltk.word_tokenize(lyrics)
                    
                    clean_tokens = [x for x in (self.word_cleanup(token) for token in tokens) if x is not None]

                    if len(clean_tokens) > 0:
                        new_text = " ".join(clean_tokens)
                        songs.append([new_text, genre])

        # create datafram and export it to a CSV file
        df = pd.DataFrame(songs, columns = ['Text', 'Genre'])
        df.to_csv('final_dataframe.csv')
    

    def run_experiments(self):
        df = pd.read_csv('final_dataframe.csv')

        # -----SETTINGS-----
        min_df = 10  # min frequency
        ngram_range = (2, 3)
        classifier = MultinomialNB()
        # classifier = LinearSVC()
        # classifier = LogisticRegression(random_state=0)
        # -----END SETTINGS-----

        # Print settings:
        print('*****-----START-----*****')
        print('n-grams: ' + str(ngram_range))
        print('min frequency: ' + str(min_df))
        print(classifier)

        # create training and test sets with 80/20 split
        text_train, text_test, genre_train, genre_test = train_test_split(df.Text, df.Genre, test_size=.2)

        # create training features
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', ngram_range=ngram_range)
        features = tfidf.fit_transform(text_train)
        print('Features shape: ' + str(features.toarray().shape))


        # train system
        tfidf_transformer = TfidfTransformer()
        text_train_tfidf = tfidf_transformer.fit_transform(features)
        clf = classifier.fit(text_train_tfidf, genre_train)

        # calculate success of system with test data
        total = len(text_test)
        rock_total = 0
        rock_correct = 0
        hiphop_total = 0
        hiphop_correct = 0
        pop_total = 0
        pop_correct = 0
        genre_pred = []
        for text, genre in zip(text_test, genre_test):
            ans = clf.predict(tfidf.transform([text]))[0]
            genre_pred.append(ans)
            if ans == genre:
                if genre == 'hiphop':
                    hiphop_correct += 1
                if genre == 'pop':
                    pop_correct += 1
                if genre == 'rock':
                    rock_correct += 1

            if genre == 'hiphop':
                hiphop_total += 1
            if genre == 'pop':
                pop_total += 1
            if genre == 'rock':
                rock_total += 1

        total = rock_total + hiphop_total + pop_total
        correct = rock_correct + hiphop_correct + pop_correct
        conf_mat = confusion_matrix(genre_test, genre_pred)        
    
        # print results of system
        print(conf_mat)
        print('Hiphop accuracy: ' + str(hiphop_correct/float(hiphop_total)))
        print('Rock accuracy: ' + str(rock_correct/float(rock_total)))
        print('Pop accuracy: ' + str(pop_correct/float(pop_total)))
        print('Complete accuracy: ' + str(correct/float(total)))
        print('*****-----COMPLETE-----*****')

        # create confusion matrix with results
        self.create_confusion_matrix(df, 'Confusion Matrix', conf_mat)

            
    def word_cleanup(self, word):
        if re.match(r'[Nn]ig[agsz]{1,3}', word):
            return 'n-word'

        if re.match(r'[Oo]{1}h*', word):
            return None

        if re.match(r'[Ll]?[Aa]{1}', word):
            return None

        nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        words = ['zero','one','two','three','four','five','six','seven','eight','nine']

        if word in nums:
            return words[int(word)]

        new_word =  self.lemmatizer.lemmatize(word.lower())
        if new_word not in self.stoplist:
            return new_word


    @staticmethod
    def create_confusion_matrix(df, name, conf_mat):
        plt.subplots(figsize=(3,3))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=set(df.Genre.values), yticklabels=set(df.Genre.values), square=True)
        plt.title(name)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        fig_name = name + '.png'
        plt.savefig(fig_name)


    @staticmethod
    def clean_text(text):
        text = re.sub(r'\[.*\]|\(.*\)|\{.*\}', '', text)  # remove text in [ ], ( ), { }
        text = re.sub("\n", " ", text)
        return text


    @staticmethod
    def create_word_cloud(genre):
        # additional stoplist words for generating wordcloud
        wordcloud_stoplist = ['like', 'still', 'every', 'know', 'got', 'make', 'feel', 'go', 'get', 'yes', 'two', 'one',
                              'come', 'give', 'even', 'want', 'said', 'let', 'put', 'tell', 'say', 'said', 'gettin', 'ever',
                              'look', 'yeah', 'thing', 'good', 'everything', 'back', 'going', 'really', 'take', 'da', 'well',
                              'need', 'see', 'much', 'way', 'something', 'right', 'uh', 'doe', 'hey', 'think', 'time', 
                              'turn', 'whole', 'word', 'man', 'woman', 'thought', 'getting']

        df = pd.read_csv('final_dataframe.csv')
        count = 0
        words = ''
        for text, gen in zip(df.Text, df.Genre):
            if gen == genre:
                words += text

        tokens = nltk.word_tokenize(words)
 
        fdist = nltk.FreqDist([x for x in tokens if x not in wordcloud_stoplist])
    
        wc = WordCloud()
        wc.generate_from_frequencies(fdist)
        wc.to_file(genre + '_pic.png')

x = NLPLyrics()
x.run_experiments()