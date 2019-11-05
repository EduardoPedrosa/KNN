import math
import random
from functools import cmp_to_key
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k, testRate, simmilarity, file):
        self.k = k
        self.testRate = testRate    #quantidade de dados para teste
        self.simmilarity = simmilarity
        self.file = file
        self.classes = []      #as classes presentes no dataset
        self.df = []    #dataframe sem a ultima coluna que é a classe
        self.res = []   #ultima coluna do dataframe que é a classificação, se é ou não spam
        self.x_train = []   #dados para treino
        self.y_train = []   #classificação dos dados de treino
        self.x_test = []    #dados para teste
        self.y_test = []    #classificação dos dados de teste
        self.y_pred = []    #classificacao predita pelo algoritmo
        self.accuracy = 0

    def loadFile(self):
        f = open(self.file, "r")
        f1 = f.readlines()
        df = []
        res = []
        for line in f1:
            lString = line.split(',')
            l = []
            for i in range(len(lString)):
                if(i != len(lString)-1):
                    l.append(float(lString[i]))
                else:
                    res.append(lString[i])
            df.append(l)
        self.df = df
        self.res = res

    def populateClasses(self):
        for x in self.res:
            if(not(x in self.classes)):
                self.classes.append(x)

    def separateData(self):
        quantity = math.floor(self.testRate * len(self.df))
        x_test = []
        y_test = []
        x_train = []
        y_train = []
        df = self.df.copy()
        res = self.res.copy()
        while(len(x_test) != quantity):
            index = random.randint(0,len(df)-1)
            x_test.append(df[index])
            y_test.append(res[index])
            del(df[index])
            del(res[index])
        x_train = df
        y_train = res
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.populateClasses()

    def distance(self, a, b):
        if(self.simmilarity == 0):
            return self.euclidean(a,b)
        elif(self.simmilarity == 1):
            return self.cosine(a,b)

    def euclidean(self, a, b):
        distance = 0
        for i in range(len(a)):
            distance += (a[i]-b[i])**2
        return math.sqrt(distance)

    def cosine(self, a, b):
        numerator = 0
        normA = 0
        normB = 0
        for i in range(len(a)):
            numerator += a[i] * b[i]
            normA += a[i]**2
            normB += b[i]**2
        return (numerator / (math.sqrt(normA) * math.sqrt(normB)))

    def moreSimmilarThen(self, a, b):
        if(self.simmilarity == 0):
            return a < b
        else:
            return a > b

    def stackSort(self, a, b):  #compara qual das distancias é menor
        if(self.moreSimmilarThen(a[1], b[1])):
            return -1
        elif(self.moreSimmilarThen(b[1], a[1])):
            return 1
        else:
            return 0

    def kNeighbors(self, point):
        stack = []     #indices e valores da distancia
        for i in range(self.k): #iniciando a pilha de menores valores com os primeiros valores do array de testes
            item = (i, self.distance(point, self.x_train[i]))
            stack.append(item)
        stack.sort(key=cmp_to_key(self.stackSort))  #ordena em ordem crescente de distancias
        
        for i in range(self.k, len(self.x_train)):
            distance = self.distance(point, self.x_train[i])
            if(self.moreSimmilarThen(distance, stack[len(stack)-1][1])):  #se o elemento é maior que o ultimo da pilha então coloca ele no lugar deste elemento
                stack[len(stack)-1] = (i, distance)
                stack.sort(key=cmp_to_key(self.stackSort))  #ordena novamente a pilha
        
        return stack

    def metrics(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for x in self.classes:
            for i in range(len(self.y_pred)):
                if(self.y_pred[i] == x):
                    if(self.y_test[i] == self.y_pred[i]):
                        tp = tp + 1
                    else:
                        fp = fp + 1
                else:
                    if(self.y_test[i] != x):
                        tn = tn + 1
                    else:
                        fn = fn + 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = (tp) / (fn + tp)
        precision = (tp) / (fp + tp)
        fScore =  2 * ((recall * precision) / (recall + precision))
        fpr = (fp) / (tn + fp)
        tnr = (tn) / (tn + fp)
        fnr = (fn) / (fn + tp)
        print('---------------------------------')
        print('K:', self.k)
        if(self.simmilarity == 0):
            print('Simmilarity: Euclidean')
        else:
            print('Simmilarity: Cossine')
        print('True positive:', tp)
        print('False positive:', fp)
        print('True negative:', tn)
        print('False negative:', fn)
        print('Accuracy:', accuracy)
        print('Recall:', recall)
        print('Precision:', precision)
        print('F-Score:', fScore)
        print('False Positive Rate:', fpr)
        print('True Negative Rate:', tnr)
        print('False Negative Rate:', fnr)

        self.accuracy = accuracy

    def execute(self):
        self.loadFile()
        self.separateData()
        correct = 0
        y_pred = []
        for i in range(len(self.x_test)):
            kSimmilars = self.kNeighbors(self.x_test[i])  #retorna os k vizinhos mais proximos (indice, distancia)
            classes = []
            for j in range(len(self.classes)):
                classes.append(0)   # Iniciando a quantidade de vezes que a classe apareceu com 0
            for x in kSimmilars:
                xClass = self.y_train[x[0]]     # Qual a classe do elemento a ser testado 
                index = self.classes.index(xClass)  # Qual o indice dessa classe na lista de classes
                classes[index] = classes[index] + 1     # Acrescentando 1 ao numero de vezes que a classe apareceu
            classification = classes.index(max(classes))
            y_pred.append(self.classes[classification])
        self.y_pred = y_pred
        self.metrics()

def main():
    testRate = 0.25
    # file = "spambase.data"
    file = "Iris.csv"
    
    euclideanAccuracy = []  #guarda todas as acuracias obtidas nos testes com a distancia euclidiana
    cossineAccuracy = []    #guarda todas as acuracias obtidas nos testes com similaridade por cosseno
    ks = []     #todos os k usados para teste

    for k in range(1, 8, 2):
        for option in range(2):   # 0 caso para euclidiano e 1 para cosseno
            knn = KNN(k, testRate, option, file)
            knn.execute()
            if(option == 0):
                ks.append(k)
                euclideanAccuracy.append(knn.accuracy)
            else:
                cossineAccuracy.append(knn.accuracy)
    plt.plot(ks, euclideanAccuracy, label="Euclidiana")
    plt.ylabel('Acurácia')
    plt.plot(ks, cossineAccuracy, label="Cosseno")
    plt.xlabel('Número de vizinhos mais próximos')
    plt.legend(loc="upper left")
    plt.title('KNN')
    plt.show()

main() 