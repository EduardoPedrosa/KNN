import math
import random
from functools import cmp_to_key

class KNN:
    def __init__(self, k, testRate, classes):
        self.k = k
        self.testRate = testRate    #quantidade de dados para teste
        self.classes = classes      #as classes presentes no dataset
        self.df = []    #dataframe sem a ultima coluna que é a classe
        self.res = []   #ultima coluna do dataframe que é a classificação, se é ou não spam
        self.x_train = []   #dados para treino
        self.y_train = []   #classificação dos dados de treino
        self.x_test = []    #dados para teste
        self.y_test = []    #classificação dos dados de teste

    def loadFile(self):
        f = open("spambase.data", "r")
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
                    res.append(float(lString[i]))
            df.append(l)
        self.df = df
        self.res = res

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

    def distance(self, a, b):
        distance = 0
        for i in range(len(a)):
            distance += (a[i]-b[i])**2
        return math.sqrt(distance)

    def stackSort(self, a, b):  #compara qual das distancias é menor
        if(a[1] < b[1]):
            return -1
        elif(a[1] > b[1]):
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
            if(distance < stack[len(stack)-1][1]):  #se o elemento é maior que o ultimo da pilha então coloca ele no lugar deste elemento
                stack[len(stack)-1] = (i, distance)
                stack.sort(key=cmp_to_key(self.stackSort))  #ordena novamente a pilha
        
        return stack


    def execute(self):
        self.loadFile()
        self.separateData()
        correct = 0
        for i in range(len(self.x_test)):
            kClosest = self.kNeighbors(self.x_test[i])
            classes = []
            for j in range(len(self.classes)):
                classes.append(0)   # Iniciando a quantidade de vezes que a classe apareceu com 0
            for x in kClosest:
                xClass = self.y_train[x[0]]     # Qual a classe do elemento a ser testado 
                index = self.classes.index(xClass)  # Qual o indice dessa classe na lista de classes
                classes[index] = classes[index] + 1     # Acrescentando 1 ao numero de vezes que a classe apareceu
            classification = classes.index(max(classes))
            if(classification == self.y_test[i]):   # Caso a classificação esteja correta
                correct += 1 

        print('k:', self.k, 'accuracy:', correct / len(self.x_test))

def main():
    testRate = 0.25
    classes = [0,1]
    for k in range(7,8):
        knn = KNN(k, testRate, classes)
        knn.execute()
main()