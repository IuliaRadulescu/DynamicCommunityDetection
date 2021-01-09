import pymongo
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import re
import seaborn as sns
sns.set_style('whitegrid')
import pyLDAvis
import pyLDAvis.sklearn

class TopicExtractor:

    def __init__(self, dataset):
        self.authors2Comments = dict(zip([x['author'] for x in dataset], [x['body'] for x in dataset]))
        self.comments = [x['body'] for x in dataset]

    # Helper function
    def plot_10_most_common_words(self, count_data, count_vectorizer):
        import matplotlib.pyplot as plt
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts+=t.toarray()[0]
        
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words)) 
        
        plt.figure(2, figsize=(15, 15/1.6180))
        plt.subplot(title='10 most common words')
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90) 
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.show()

    # Helper function
    def print_topics(self, model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def removeLinks(self):
        self.comments = list(map(lambda x: re.sub(r'(https?://[^\s]+)', '', x), self.comments))

    def removeRedditReferences(self):
        self.comments = list(map(lambda x: re.sub(r'(/r/[^\s]+)', '', x), self.comments))

    def removePunctuation(self):
        self.comments = list(map(lambda x: re.sub('[,.!?"\'\\n:*]', '', x), self.comments))

    def prepareForLDA(self):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.comments)

        # self.plot_10_most_common_words(X, vectorizer)

        return (vectorizer, X)

    def getTopics(self, noTopics, noWords):
        self.removeLinks()
        self.removeRedditReferences()
        self.removePunctuation()
        vectorizer, X = self.prepareForLDA()

        lda = LDA(n_components = noTopics, n_jobs = -1)
        lda.fit(X) # Print the topics found by the LDA model
        print("Topics found via LDA:")
        self.print_topics(lda, vectorizer, noWords)

        visData = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
        pyLDAvis.save_html(visData,'USAElections.html')
        

class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance != None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

        self.dbClient = pymongo.MongoClient('localhost', 27017)

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance == None:
            MongoDBClient()

        return MongoDBClient.__instance


dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSAElections

allComments = list(db.comments.find()) + list(db.submissions_1604775600_1604782800.find())

# filter nodes with inexisting parents
commentsWithoutParents = [comment for comment in allComments if ('parentId' in comment) and (comment['parentId'].split('_')[1] not in [comment['redditId'] for comment in allComments])]
print('Comments without parents ', len(commentsWithoutParents))

topicExtractor = TopicExtractor(allComments)
topicExtractor.getTopics(2, 10)
