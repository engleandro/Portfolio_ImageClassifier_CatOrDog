# PORTFOLIO PROJECT: PET IMAGE CLASSIFIER, CAT OR DOG?

My portfolio project on artificial intelligence based on linux, docker, docker-compose, python and its classical external libraries (numpy, sklearn, tensorflow) on data science and artificial intelligence.

It comprises a Convolutional Deep Neural Networks (CDNN) based on Tensorflow and Scikit-Learn. This pet image classifier predicts if a pet image has a dog or a cat.

# SOURCES

- [Ebook] Chollet, François; 2021. Deep Learning with Python. Github: https://www.manning.com/books/deep-learning-with-python
- [Ebook] Gad, Ahmed Fawzy; 2019. Practical Computer Vision Applications Using Deep Learning with CNNs With Detailed Examples in Python Using TensorFlow and Kivy-Apress (2019).
- [Article] https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
- [Article] https://towardsdatascience.com/cat-or-dog-image-classification-with-convolutional-neural-network-d421a9363c7a

# DATASET

This image classifier employes the Oxford IIIT Pet Image dataset to train, validate and test the model.

This dataset is composed by 7390 images of pets in 37 classes each with about 200 images. The images are not standarized, they change in size, aspect ratio, color parameters, and so on.

Website:

- https://www.robots.ox.ac.uk/~vgg/data/pets/

Downloads:

- https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
- https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# INSTALLATION AND SETUP

Clone the source codes on my GitHub repository:

    $ git clone git@github.com:engleandro/Portfolio_ImageClassifier_CatOrDog.git

Create a repository and download the dataset:

    $ cd Portifolio-ImageClassifier
    $ mkdir dataset
    $ cd dataset
    $ wget -P . https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    $ tar -xf images.tar.gz -C .
    $ rm -f images.tar.gz
    $ cd ..

Create a docker image and run the Image Classifier process:

    $ docker-compose up

Another way:

    $ docker-compose up -d
    $ docker-compose logs -f

# DEVELOPER GUIDE

This application is based on linux, docker, docker-compose and python.

It employes some external libraries:

- Numpy,
- Scikit-learn,
- Tensorflow (Keras),
- Tqdm.

# TENSOR BOARD

The TensorBoard GUI can be accessed at localhost:6006 to see model metrics.

The docker-compose allows TensorBoard GUI by running the command line below:

    $ tensorboard --bind_all --logdir logs/fit

# DATA PROCESSING

The dataset should be divided into train, validation and test datasets by:

- 70% to train the model,
- 15% to validate the model,
- 15% to test the model.

For each image file, its label class was extracted from the filename. Dog's image starts with lower case and cat's image starts with upper case.

Since the images are not standarized, the images were loaded with fixed (256, 256) resolution with padding. The padding helps prevent distortion due to stretching or shrinking of images when changing its aspect ratio. The ImageDataGenerator from Keras API was used for image augmentation during training.

# DEEP CNN MODEL

The model comprises a deep Convolutional Neural Network based on Keras (tensorflow API).

It comprises the best practice on image processing with Deep CNN with many layers:

- Input
- Convolution 2D
- Max Pooling 2D
- Batch Normalization
- Dropout
- Dense
- Output

The keras model is defined as:

    model = Sequential([
        Conv2D(
            32, 5,
            padding='same',
            input_shape=(2556, 256, 3)
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

The model employs callbacks to prevent overfitting and monitor training respectively. The dropout layers avoid overfitting, increase performance and adaptability to the model.

# BUILDING THE MODEL

This project was developed based on Objet-Oriented Programming (OOP) paradigm.

A class <b>ImageClassifier</b> was created to improve development and maintenance. It is a specific object to the current problem, not a general class to classify pet images.

In the app.py, we can load dataset, set up, create, compile, train, validate, test and save the model:

    def buildModel():

        classifier = ImageClassifier()

        MODELLING_IMAGES = int(
            os.environ.get('MODELLING_IMAGES')
        )

        classifier.setSeed(
            seed=int(os.environ.get('RANDOM_STATE'))
        )
        classifier.setListImages(
            limit=MODELLING_IMAGES
        )
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

        classifier.saveModel()

# USING THE MODEL

In the app.py, to load the model and predicts if a pet image is a dog or cat:

    filename = 'beagle_100.jpg'

    def predictImage(filename: str):

        classifier = ImageClassifier()
        classifier.loadModel()
        if classsifier.model:
            return classifier.predictImage(
                filename=filename
            )
        return False

The model metrics can be accesed via Tensorflow Board on localhost:6006.

# RESULTS

The metrics of the model as accuracy and loss can be accessed at Tensorflwo Board at localhost:6006.

# REST API

A rest API based on Django was started, but not finished unfortunately. Please, just ignore.
