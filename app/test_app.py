import os
import sys
import threading
from pprint import pprint

from dotenv import load_dotenv

from app.classifier import ImageClassifier


load_dotenv('./setup/.env[MODEL]')


def buildModel():

    classifier = ImageClassifier()

    MODELLING_IMAGES = int(os.environ.get('MODELLING_IMAGES'))
    classifier.IMAGE_SIZE = [int(value) for value in str(os.environ.get('IMAGE_SIZE')).split(', ')]
    classifier.RANDOM_STATE = int(os.environ.get('RANDOM_STATE'))
    classifier.TRAIN_SIZE = float(os.environ.get('TRAIN_SIZE'))
    classifier.VALIDATION_SIZE = float(os.environ.get('VALIDATION_SIZE'))
    classifier.TEST_SIZE = float(os.environ.get('TEST_SIZE'))
    classifier.BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
    classifier.LEARNING_RATE = float(os.environ.get('LEARNING_RATE'))
    classifier.EPOCHS = int(os.environ.get('EPOCHS'))
    classifier.PLOTS_DPI = int(os.environ.get('PLOTS_DPI'))

    classifier.setSeed(seed=int(os.environ.get('RANDOM_STATE')))
    classifier.setListImages(limit=MODELLING_IMAGES)
    classifier.setImagesSize()
    classifier.setLabels()
    classifier.setLabelsCatOrDog()

    classifier.loadImages()
    classifier.createModel()
    classifier.compileModel()
    classifier.showModel()
    
    train_dataset, test_dataset, validation_dataset = \
        classifier.generateImageData(
            *classifier.splitDataset()
        )
    history = classifier.runTrainValidate(
        train_dataset, 
        validation_dataset
    )
    loss, accuracy = classifier.evaluateModel(
        test_dataset=test_dataset
    )
    pprint(loss, accuracy)
    
    classifier.saveModel()

def predictImage(filename: str):

    classifier = ImageClassifier()
    classifier.loadModel()

    return classifier.predictImage(
        filename=filename
    )




if __name__ == '__main__':

    buildModel()
    
    filename = 'beagle_100.jpg'
    predictImage(filename)

