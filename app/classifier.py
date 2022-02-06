import os
import glob
import datetime
import random
import json

import numpy
import matplotlib
import seaborn
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plot

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard
)
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator
)
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense,
    LeakyReLU,
    BatchNormalization
)
from dotenv import load_dotenv


seaborn.set_theme()


class ImageClassifier():


    # [FEATURES]

    ROOT_PATH = os.getcwd()
    BASE_PATH = os.path.join(
        os.getcwd(),
        "dataset",
        "images"
    )
    RELATIVE_PATH = os.path.join(
        "dataset",
        "images"
    )
    IMAGE_SIZE = (256, 256)
    RANDOM_STATE = int(10000*random.random())
    TRAIN_SIZE = 0.70
    VALIDATION_SIZE = 0.15
    TEST_SIZE = 0.15
    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.0001
    PLOTS_DPI = 200
    LOGS = "logs/fit/{}".format(
        datetime.datetime.now().isoformat()
    )


    # [CONSTRUCTOR]

    def __init__(self,
            path: str=os.path.join(
                "dataset",
                "images"
            ),
            args: list=[],
            kwargs: dict={},
            ) -> None:
        self.relative_path = path
        self.absolute_path = os.path.join(
            os.getcwd(),
            self.relative_path
        )
        self.dataset_files = list()
        self.dataset_size = None
        self.dataset = list()
        self.extension = "jpg"
        self.model = None
        self.setSeed()
    
    def __str__(self) -> str:
        return str(self.__dict__)


    # [GENERAL-METHODS]

    def setSeed(self, seed: int=None):
        if not seed:
            self.seed = self.RANDOM_STATE
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        #tensorflow.set_random_seed(self.seed)
    
    def serialize(self) -> bytes:
        return json.dumps(
            self.__dict__
        ).encode('utf-8')
    
    def deserialize(self, payload):
        return json.loads(
            payload.decode('utf-8')
        )
    
    def listImages(self,
            path: str=None,
            extension: str="jpg",
            start: int=0,
            limit: int=None
            ) -> list:
        if not path:
            path = self.absolute_path
        files = [
            os.path.basename(file) \
                for file in glob.glob(
                    path + f'/*.{extension}'
                    #f'./**/*.{extension}',
                    #recursive=True)
                )
        ]
        if limit:
            start = start \
                if start<len(files) \
                else (len(files)-1)
            end = (start+limit) \
                if (start+limit)<len(files) \
                else len(files)
            return files[start:end]
        return files
    def setListImages(self,
            path: str=None,
            extension: str="jpg",
            start: int=0,
            limit: int=None
            ) -> None:
        self.dataset_files = self.listImages(
            path=path,
            extension=extension,
            start=start,
            limit=limit
        )

    def countImages(self,
            path: str=None,
            extension: str="jpg",
            start: int=0,
            limit: int=None
            ) -> int:
        if not self.dataset_files:
            self.setListImages(
                path=path,
                extension=extension,
                start=start,
                limit=limit
            )
        return len(self.dataset_files)
    def setImagesSize(self,
            path: str=None,
            extension: str="jpg",
            start: int=0,
            limit: int=None
            ) -> None:
        self.dataset_size = self.countImages(
                path=path,
                extension=extension,
                start=start,
                limit=limit
            )

    def getLabels(self, separator: str='_') -> list:
        return [
            ' '.join(files.split(f'{separator}')[:-1]) \
                for files in self.dataset_files
        ]
    def enumerateLabels(self, separator: str='_') -> dict:
        if not self.dataset_labels:
            self.setLabels(separator=separator)
        return {
            index: label for index, label in enumerate(
                numpy.unique(self.dataset_labels)
            )
        }
    def setLabels(self, separator: str='_') -> None:
        self.dataset_labels = self.getLabels(
            separator=separator
        )
    
    def isDog(self, label: str) -> bool:
        return True if label[0].islower() else False
    def isCat(self, label: str) -> bool:
        return True if label[0].isupper() else False
    
    def getLabelsCatOrDog(self, separator: str='_') -> list:
        labels = self.getLabels(separator=separator)
        return ['dog' if label[0].islower() else 'cat' \
            for label in labels]
    def setLabelsCatOrDog(self, separator: str='_') -> None:
        self.dataset_labels = self.getLabelsCatOrDog(
            separator=separator
        )
    def classifyLabelsCatOrDog(self, separator: str='_') -> dict:
        cat_or_dog = {'cat': [], 'dog': []}
        labels = self.getLabels(separator=separator)
        for label in set(labels):
            key = 'cat' if label[0].isupper() else 'dog'
            cat_or_dog[key].append(label)
        return cat_or_dog
    
    def countLabels(self):
        counter = {}
        for label in self.dataset_labels:
            try:
                counter[label] += 1
            except:
                counter[label] = 1
    
    def loadImages(self) -> None:
        for filename in tqdm(
                self.dataset_files,
                desc='Loading images',
                unit=' images'
                ):
            try:
                image = load_img(
                    os.path.join(self.relative_path, filename)
                )
                image = tensorflow.image.resize_with_pad(
                    img_to_array(image, dtype='uint8'),
                    *self.IMAGE_SIZE
                ).numpy().astype('uint8')
                self.dataset.append(image)
            except FileNotFoundError:
                self.dataset_files.remove(filename)
                self.dataset_size -= 1
        self.setLabels()
    
    def loadImage(self, filename) -> None:
        image = load_img(
            os.path.join(self.relative_path, filename)
        )
        image = tensorflow.image.resize_with_pad(
            img_to_array(image, dtype='uint8'),
            *self.IMAGE_SIZE
        ).numpy().astype('uint8')
        return image
    
    def plotImages(self,
            rows: int=3,
            columns: int=3,
            figure_size: tuple=(15, 15)
            ):
        plot.subplots(
            nrows=rows,
            ncols=columns,
            figsize=figure_size
        )
        for index, image_id in enumerate(
                numpy.random.randint(
                    0,
                    self.dataset_size,
                    size=(rows*columns)
                    )
                ):
            try:
                plot.subplot(3, 3, index + 1)
                plot.axis(False)
                plot.grid(False)
                plot.title(f'{self.dataset_files[image_id]}')
                plot.imshow(self.dataset[image_id])
            except: #noqa
                print(f'error on plot {self.dataset_files[image_id]}')
        plot.show()
    
    
    # [MODEL]

    def createModel(self):
        if not self.dataset_labels:
            self.setLabels()
        self.setLabelsCatOrDog()
        labels = {
            label: index for index, label in enumerate(
                numpy.unique(self.dataset_labels)
            )
        }
        self.model = Sequential([
            Conv2D(
                32, 5,
                padding='same',
                input_shape=(*self.IMAGE_SIZE, 3)
            ),
            Conv2D(
                32, 5,
                padding='same',
                activation=LeakyReLU(alpha=0.5)
            ),
            MaxPooling2D(),
            Conv2D(32, 4, padding='same'),
            Conv2D(
                32, 4,
                padding='same',
                activation=LeakyReLU(alpha=0.5)
            ),
            MaxPooling2D(),
            Conv2D(64, 4, padding='same'),
            Conv2D(
                64, 4,
                padding='same',
                activation=LeakyReLU(alpha=0.5)
            ),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(64, 3, padding='same'),
            Conv2D(
                64, 3,
                padding='same',
                activation=LeakyReLU(alpha=0.5)
            ),
            MaxPooling2D(),
            Conv2D(128, 3, padding='same'),
            Conv2D(
                128, 3,
                padding='same',
                activation=LeakyReLU(alpha=0.5)
            ),
            MaxPooling2D(),
            Conv2D(128, 2, padding='same'),
            Conv2D(
                128, 2,
                padding='same',
                activation=LeakyReLU(alpha=0.5)
            ),
            MaxPooling2D(),
            Flatten(),
            BatchNormalization(),
            Dropout(0.2),
            Dense(512, activation='sigmoid'),
            Dropout(0.2),
            Dense(256, activation='sigmoid'),
            Dropout(0.1),
            Dense(len(labels), activation='softmax')
        ])
    
    def compileModel(self):
        self.model.compile(
            optimizer=Adam(self.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
    
    def buildModel(self):
        self.createModel()
        self.compileModel()
    
    def showModel(self):
        if self.model:
            self.model.summary()
        return False
    
    def evaluateModel(self, test_dataset):
        if self.model:
            loss, accuracy = self.model.evaluate(
                test_dataset,
                verbose=0
            )
            return {
                'loss': loss,
                'accuracy': accuracy
            }
        return False
    
    def predictImage(self, filename: str):
        image = load_img(
            os.path.join(self.relative_path, filename),
            target_size=tuple(self.IMAGE_SIZE)
        )
        image_tensor = img_to_array(image)
        image_tensor = numpy.expand_dims(
            image_tensor,
            axis=0
        )
        return self.model.predict(image_tensor)
    
    def saveModel(self, path: str="model"):
        if self.model:
            self.model.save(path)
        return False

    def loadModel(self, path: str='model'):
        if self.model:
            self.model = tensorflow.keras.models.load_model(
                path
            )
        return False

    # [TRAIN-TEST-VALIDATE]

    def splitDataset(self) -> list:
        self.dataset = numpy.array(self.dataset)
        self.dataset.shape
        if not self.dataset_labels:
            self.setLabels()
        self.setLabelsCatOrDog()
        labels = {
            label: index for index, label in enumerate(
                numpy.unique(self.dataset_labels)
            )
        }
        labels_map = list(
            map(
                lambda label: labels.get(label),
                self.dataset_labels
            )
        )
        self.dataset.max()
        features, features_test, \
        labels, labels_test = train_test_split(
            self.dataset,
            labels_map,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=labels_map
        )
        features_train, features_validation, \
        labels_train, labels_validation = train_test_split(
            features,
            labels,
            test_size=self.VALIDATION_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=labels
        )
        return [
            features_train, labels_train,
            features_validation, labels_validation,
            features_test, labels_test,
        ]
    
    def generateImageData(self,
            features_train,
            labels_train,
            features_validation,
            labels_validation,
            features_test,
            labels_test,
            ):
        train_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_dataset = train_generator.flow(
            x=features_train,
            y=labels_train,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        validation_generator  = ImageDataGenerator(
            rescale=1./255,
        )
        validation_dataset = validation_generator.flow(
            x=features_validation,
            y=labels_validation,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        test_generator = ImageDataGenerator(
            rescale=1./255
        )
        test_dataset = test_generator.flow(
            x=features_test,
            y=labels_test,
            batch_size=self.BATCH_SIZE
        )
        return [
            train_dataset,
            validation_dataset,
            test_dataset,
        ]
    
    def runTrainValidate(self,
            train_dataset,
            validation_dataset,
            ):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        tensor_board = TensorBoard(
            log_dir=self.LOGS,
            histogram_freq=20
        )
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.EPOCHS,
            verbose=1,
            callbacks=[early_stopping, tensor_board]
        )
        return history
    
    def getPerformance(self, history):
        return {
            'epochs_range': history.epoch,
            'train_loss': history.history['loss'],
            'train_accuracy': history.history['sparse_categorical_accuracy'],
            'validation_loss': history.history['val_loss'],
            'validation_accuracy': history.history['val_sparse_categorical_accuracy'],
        }

    
    # [APPLICATION]

    def buildClassifier(self):
        self.setListImages()
        self.setImagesSize()
        self.setLabels()
        self.loadImages()
        self.createModel()
        self.compileModel()
        train_dataset, test_dataset, \
        validation_dataset = \
            self.generateImageData(
                *self.splitDataset()
            )
        history = self.runTrainValidate(
            train_dataset, 
            validation_dataset
        )
        performance = self.getPerformance(history)
        evaluation = self.evaluateModel(
            test_dataset=test_dataset
        )
        self.saveModel()
        return [
            history,
            performance,
            evaluation
        ]

    def predictImage(self, filename: str):
        if not self.model:
            self.loadModel()
        return self.predictImage(
            filename=filename
        )

