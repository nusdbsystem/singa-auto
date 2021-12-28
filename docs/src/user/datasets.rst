Dataset Types
=============

   **note**

   Refer to
   `./examples/datasets/ <https://github.com/nusdbsystem/singa-auto/tree/master/examples/datasets/>`__
   for examples on pre-processing common dataset formats to conform to
   the SINGA-Auto's own dataset formats.

CORPUS
------

The dataset file must be of the ``.zip`` archive format with a
``corpus.tsv`` at the root of the directory.

The ``corpus.tsv`` should be of a
`.TSV <https://en.wikipedia.org/wiki/Tab-separated_values>`__ format
with columns of ``token`` and ``N`` other variable column names (*tag
columns*).

For each row,

   ``token`` should be a string, a token (e.g. word) in the corpus.
   These tokens should appear in the order as it is in the text of the
   corpus. To delimit sentences, ``token`` can be take the value of
   ``\n``.

   The other ``N`` columns describe the corresponding token as part of
   the text of the corpus, *depending on the task*.

SEGMENTATION_IMAGES
-------------------

-  Inside the uploaded ``.zip`` file, the training and validation sets
   should be wrapped separately, and be named strictly as ``train`` and
   ``val``.

-  For ``train`` folder (the same for ``val`` folder), the images and
   annotated masks should also be wrapped separately, and be named
   strictly as ``image`` and ``mask``.

-  ``mask`` folder should contain only ``.png`` files and file name
   should be the same as each mask's corresponding image. (eg. for an
   image named ``0001.jpg``, its corresponding mask should be named as
   ``0001.png``)

-  An JSON file named ``params.json`` must also be included in the
   ``.zip`` file, in order to indicates the essential training
   parameters such as ``num_classes``, for example:

   .. code:: json

      {
          "num_classes": 21
      }

An example of the upload ``.zip`` file structure:

::

   + dataset.zip
       + train
           + image
               + 0001.jpg
               + 0002.jpg
               + ...
           + mask
               + 0001.png
               + 0002.png
               + ..  
       + val
           + image
               + 0003.jpg
               + ...
           + mask
               + 0003.png
               + ...
       + params.json

IMAGE_FILES
-----------

The dataset file must be of the ``.zip`` archive format with a
``images.csv`` at the root of the directory.

The ``images.csv`` should be of a
`.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`__ format
with columns of ``path`` and ``N`` other variable column names (*tag
columns*).

For each row,

   ``path`` should be a file path to a ``.png``, ``.jpg`` or ``.jpeg``
   image file within the archive, relative to the root of the directory.

   The other ``N`` columns describe the corresponding image, *depending
   on the task*.

DETECTION_DATASET
-----------------

It is recommended to follow the YOLO dataset format.

-  For folder hierarchy, two folders ‘images’ and ‘labels’ should be
   prepared. In ‘images’ folder, there are PIL loadable images, and the
   corresponding ``txt`` label files should be placed in ‘labels’
   folder, with the same basename with the images.

-  The label file format is as follows, where ``object-id`` is the index
   of object, the following four numbers should be normalized to range
   between 0 and 1 by dividing by the width and height of the image.
   ``center_x center_y`` are the central coordinates of bounding box,
   and ``width heigh`` is the sides lengths of it. It is allowable to
   use empty label file (negative samples), which means there are no
   objects to detect in the image.

::

   object-id center_x center_y width height
   ...

-  In addition, ``train.txt``, ``valid.txt`` can be provided to note
   images used for training/validataion, only including the path of
   image files. A ``class.names`` contains the category names and thier
   line numbers are ``object-id``.

GENERAL_FILES
-------------

-  For general task, as its name states, any domain’s task (or model)
   can be included within this category, such as image processing, nlp,
   speech, or video.
-  There is no requirements for the form of dataset, as long as it can
   be read into memory in the form of a file. However, the model
   developer has to know in advance how to handle the read-in file.

QUESTION_ANSWERING_COVID19
--------------------------

The dataset file must be of the ``.zip`` archive format, containing
`JSON <https://en.wikipedia.org/wiki/JSON>`__ files. JSON files under
different levels of folders will be automaticly read all together.

Each JSON file is extracted from one paper. `JSON
structure <https://en.wikipedia.org/wiki/JSON#Example>`__ contains field
body_text, which is a list of {"text": <str>} blocks. Each text block is
namely each paragraph of corresponding paper.

Meanwhile, a metadata.csv file, at the root of the archive directory, is
optional. It is to provide the model with publish_time column, each
entry is in Date format, e.g. 2001-12-17. In this condition, each
metadata entry is required to have sha value column in General format,
and each JSON file required to have "sha":<str> field, while both sha
values linked. When neither metadata.csv or publish_time Date value is
provided, the model would not check the timeliness of corresponding JSON
body_text field.

QUESTION_ANSWERING_MEDQUAD
--------------------------

The dataset file must be of the ``.zip`` archive format, containing
`xml <https://en.wikipedia.org/wiki/XML#/media/File:XMLSample.png>`__
files. Xml files under different levels of folders will be automaticly
read all together.

Model would only take <Document> <QAPairs> ... </QAPairs>
</Document>field, and this filed contains multiple <QAPair> ...
</QAPair>. Each QAPair has one <Question> ... </Question> and its
<Answer> ... </Answer> combination.

TABULAR
-------

The dataset file must be a tabular dataset of the ``.csv`` format with
``N`` columns.

AUDIO_FILES
-----------

The dataset file must be of the ``.zip`` archive format with a
``audios.csv`` at the root of the directory.

The ``audios.csv`` should be of a
`.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`__ format
with 3 columns of ``wav_filename``, ``wav_filesize`` and ``transcript``.

For each row,

   ``wav_filename`` should be a file path to a ``.wav`` audio file
   within the archive, relative to the root of the directory. Each audio
   file's sample rate must equal to 16kHz.

   ``wav_filesize`` should be an integer representing the size of the
   ``.wav`` audio file, in number of bytes.

   ``transcript`` should be a string of the true transcript for the
   audio file. Transcripts should only contain the following alphabets:

      ::

         a
         b
         c
         d
         e
         f
         g
         h
         i
         j
         k
         l
         m
         n
         o
         p
         q
         r
         s
         t
         u
         v
         w
         x
         y
         z


         '

   An example of ``audios.csv`` follows:

.. code:: text

   wav_filename,wav_filesize,transcript
   6930-81414-0000.wav,412684,audio transcript one
   6930-81414-0001.wav,559564,audio transcript two
   ...
   672-122797-0005.wav,104364,audio transcript one thousand
   ...
   1995-1837-0001.wav,279404,audio transcript three thousand

Query Format
~~~~~~~~~~~~

A `Base64-encoded <https://en.wikipedia.org/wiki/Base64>`__ string of
the bytes of the audio as a 16kHz .wav file

Prediction Format
~~~~~~~~~~~~~~~~~

A string, representing the predicted transcript for the audio.
