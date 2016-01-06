import sys
import os
import multiprocessing

LIBSVM = '/libsvm-3.20'
WEKA = '/weka-3-6-11/weka.jar'

fileNames = ['1mer', '2mer', '3mer', '4mer', '5mer', 'order']
c = [2048, 8192, 32768, 32768, 32768, 0.5]
gamma = [8, 0.5, 0.03125, 0.001953125, 0.001953125, 2]

def clean(f, t):
    '''
    Reads the input file and extracts cleaned data from it.
    '''
    datafile = open(f, 'r')
    cleaned = []
    line = datafile.readline()
    while line != "":
        if line[0] != '>':
            line = datafile.readline()
        else:
            data = [line, '']
            length = 0
            valid = True
            line = datafile.readline()
            while line != '' and line[0] != '>' and valid:
                valid = not 'N' in line
                data[1] += line
                line = line.replace('\n', '')
                length += len(line)
                line = datafile.readline()
            if length >= t and valid:
                cleaned.append(data)
    datafile.close()
    return cleaned

def extract(p, t, numThreads):
    '''
    Extracts features from the input file.
    '''
    cleanData = []
    sequences = []
    data = clean(p, t)
    samples = []
    for i in range(1, 6):
        samples.append(open(str(i) + "mer.svmtest", "w"))
    samples.append(open("order.svmtest", "w"))
    sequences = open('sequences', 'w')
    sem = multiprocessing.Semaphore(numThreads)
    lock = multiprocessing.Lock()
    threadList = []
    for seq in range(len(data)):
        sem.acquire()
        lock.acquire()
        threadList.append(multiprocessing.Process(target=calculateFeatures, args=(data[seq], samples, sequences, lock, sem)))
        threadList[-1].start()
        lock.release()
        i = 0
        while i < len(threadList):
            if not threadList[i].is_alive():
                threadList[i].join()
                threadList.pop(i)
            else:
                i += 1
    for thread in threadList:
        thread.join()
    for i in range(6):
        samples[i].close()
    sequences.close()

def calculateFeatures(data, samples, sequences, lock, sem):
    '''
    Calculates feature values for the input sequences.
    '''
    # Initialize variables
    currentSequence = data[1].replace('\n', '')
    currentLength = len(currentSequence)
    nucleotides = ['A', 'C', 'G', 'T']
    kmer = []
    for k in range(1, 6):
        kmer.append([0] * 4**k)
    delta = [0] * 199

    # Counts for features
    for pos in range(len(currentSequence)):
        for k in range(1, 6):
            if pos + k < len(currentSequence):
                window = currentSequence[pos:pos+k]
                index = 0
                for i in range(len(window)):
                    index += 4**i * nucleotides.index(window[i])
                kmer[k - 1][index] += 1
        i = 1
        while i < 200 and pos + i < len(currentSequence):
            delta[i - 1] += int(currentSequence[pos] != currentSequence[pos + i])
            i += 1

    # Write the svm-training files
    toWrite = ["0 ", "0 ", "0 ", "0 ", "0 ", "0 "]
    index = [1, 1, 1, 1, 1, 1]
    for k in range(1, 6):
        w = 1 / (4**(5 - k))
        s = float(currentLength - k + 1)
        mod = w/s
        for i in range(len(kmer[k - 1])):
            value = kmer[k - 1][i] * mod
            if value:
                toWrite[k - 1] += str(index[k - 1]) + ":" + str(value) + " "
            index[k - 1] += 1
    for i in range(1, 199):
        value = delta[i - 1] / float(currentLength - i)
        if value:
            toWrite[-1] += str(index[-1]) + ":" + str(value) + " "
        index[-1] += 1
    value = delta[198] / float(currentLength - 199)
    if value:
        toWrite[-1] += str(index[-1]) + ":" + str(value)
    for i in range(6):
        toWrite[i] += "\n"
    lock.acquire()
    for i in range(6):
        samples[i].write(toWrite[i])
        samples[i].flush()
        sequences.write(data[0]+data[1])
        sequences.flush()
    lock.release()
    sem.release()

def run(command, sem):
    '''
    This function is for multiprocessing.
    '''
    os.system(command)
    sem.release()

def scale(numThreads):
    '''
    Scales the svm-test files.
    '''
    threadList = []
    sem = multiprocessing.Semaphore(numThreads)
    for k in range(6):
        fileName = fileNames[k]
        command = LIBSVM + "svm-scale -r data/" + fileName + ".range " + fileName + ".svmtest > " + fileName + ".svmtest.scale"
        threadList.append(multiprocessing.Process(target=run, args=(command, sem)))
        sem.acquire()
        threadList[-1].start()
    for thread in threadList:
        thread.join()
    for k in range(6):
        fileName = fileNames[k]
        os.system('mv ' + fileName + '.svmtest.scale ' + fileName + '.svmtest')

def runTests(numThreads):
    '''
    Runs svm-predict and generates features for weka.
    '''
    threadList = []
    sem = multiprocessing.Semaphore(numThreads)
    for k in range(6):
        fileName = fileNames[k]
        command = LIBSVM + "svm-predict -q " + fileName + ".svmtest data/" + fileName + ".model " + fileName + ".wekatest"
        threadList.append(multiprocessing.Process(target=run, args=(command, sem)))
        sem.acquire()
        threadList[-1].start()
    for thread in threadList:
        thread.join()

def combine():
    '''
    Combines the results from the different svm-models into an arff for weka.
    '''
    for k in range(5):
        arff = open('wekatest.arff', 'w')
        arff.write('@RELATION LNCvsMRNA\n\n@ATTRIBUTE 1mer {-1,1}\n@ATTRIBUTE 2mer {-1,1}\n@ATTRIBUTE 3mer {-1,1}\n@ATTRIBUTE 4mer {-1,1}\n@ATTRIBUTE 5mer {-1,1}\n@ATTRIBUTE order {-1,1}\n@ATTRIBUTE LNC {-1,1}\n\n@DATA\n')
        towrite = []
        for fileName in fileNames:
            cur = 0
            p = open(fileName+'.wekatest', 'r').readlines()
            for j in range(len(p)):
                if fileName == '1mer':
                    towrite.append([])
                    towrite[cur].append('?')
                towrite[cur].append(p[j].split()[0])
                cur += 1
        for i in towrite:
            arff.write(i[1]+','+i[2]+','+i[3]+','+i[4]+','+i[5]+','+i[6]+','+i[0]+'\n')
        arff.close()

def runWeka():
    '''
    Runs a naive bayes classifier and reports the predicted results in prediction.
    '''
    os.system("export CLASSPATH="+WEKA+";java weka.classifiers.bayes.NaiveBayes -t data/wekatrain.arff -T wekatest.arff -p 0 > temp")
    temp = open('temp', 'r')
    predictions = temp.readlines()
    temp.close()
    os.system('rm temp')
    results = open('predictions', 'w')
    classified = ['coding', 'long-non-coding']
    for i in range(5, len(predictions)):
        l = predictions[i].split()
        if len(l):
            results.write(classified[int(l[2][0])-1]+'\n')
    results.close()

def main():
    '''
    Takes a fasta file as input and outputs a sequence file also in fasta format and a prediction file to go along with it which classifies the sequences as either coding or long-non-coding.
    '''

    # Read command line parameters
    p = None
    t = 200
    numThreads = 1
    for i in range(1,len(sys.argv),2):
        if sys.argv[i] == '-p':
            p = sys.argv[i+1]
        if sys.argv[i] == '-t':
            t = int(sys.argv[i+1])
            if t < 200:
                print('t must be at least 200, defaulted to 200')
                t = 200
        if sys.argv[i] == '-n':
            numThreads = int(sys.argv[i+1])
    if p == None:
        print('invalid input')
        print('\n-p input file for prediction\n-t sequence length threshhold; default 200\n-n number of threads; default 1 ')
        return 0

    # Run pipeline
    print('extracting data')
    extract(p, t, numThreads)
    print('scaling data')
    scale(numThreads)
    print('svm-classification')
    runTests(numThreads)
    print('combining data')
    combine()
    print('naivebayes-classification')
    runWeka()
    os.system('rm *wekatest* *svmtest')
    print('done')
    
main()
