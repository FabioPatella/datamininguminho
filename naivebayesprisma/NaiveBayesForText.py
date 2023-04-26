from collections import defaultdict

class NaiveBayesForText:
    """
    class for text classification using NaiveBayes
    """
    def fit(self,text,labels):
        """
        fit the model with a list of sentences
        :param text: list of pairs of string anf label
        :param labels: list of label
        """
        self.labelindex={}
        self.wordlabelcount=defaultdict(int)
        index=0
        for label in labels:
            self.labelindex[label]=index
            index=index+1
        numberofclasses=len(labels)
        labelcount=defaultdict(int)
        self.vocabolary = defaultdict(lambda: [0 for _ in range(numberofclasses)])
        for sentence_label in text:
            sentence=sentence_label[0]
            label=sentence_label[1]
            labelcount[label]=labelcount[label]+1
            listwords=sentence.split()
            for word in listwords:
                word=word.lower()
                self.vocabolary[word][self.labelindex[label]]=self.vocabolary[word][self.labelindex[label]]+1
                self.wordlabelcount[label]=self.wordlabelcount[label] +1

        self.cardinality=len(self.vocabolary)
        self.classprobability={}
        for label,count in labelcount.items():
            self.classprobability[label]=count/self.cardinality

    def predict(self,textstring):
        """
        :param textstring: string to predict
        :return: prediction for the string in input
        """
        classoutcomeprobility=defaultdict(float)
        for label,indexlabel in self.labelindex.items():
            classoutcomeprobility[label]=self.classprobability[label]
            listwords = textstring.split()
            for word in listwords:
                word=word.lower()
                classoutcomeprobility[label]=(self.vocabolary[word][indexlabel]+1)/(self.cardinality + self.wordlabelcount[label]) + classoutcomeprobility[label]
        prediction = max(classoutcomeprobility, key=classoutcomeprobility.get)

        return prediction


if __name__ == '__main__':
    text=[("a baixa do porto","Porto"),("o mercado do bolhão é no porto","Porto"),
          ("a câmara do porto fica no centro do porto","Porto"),("a baixa de lisboa","Lisboa"),
          ("o porto de lisboa","Lisboa")
          ]
    nvt=NaiveBayesForText()
    nvt.fit(text,["Porto","Lisboa"])
    prediction = nvt.predict("Porto è bella")
    print("the prediction is:",prediction)








