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

    ``token`` should be a string, a token (e.g. word) in the corpus.
    These tokens should appear in the order as it is in the text of the
    corpus. To delimit sentences, ``token`` can be take the value of
    ``\n``.

    The other ``N`` columns describe the corresponding token as part of
    the text of the corpus, *depending on the task*.

SEGMENTATION\_IMAGES
--------------------

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

IMAGE\_FILES
------------

The dataset file must be of the ``.zip`` archive format with a
``images.csv`` at the root of the directory.

The ``images.csv`` should be of a
`.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`__ format
with columns of ``path`` and ``N`` other variable column names (*tag
columns*).

For each row,

    ``path`` should be a file path to a ``.png``, ``.jpg`` or ``.jpeg``
    image file within the archive, relative to the root of the
    directory.

    The other ``N`` columns describe the corresponding image, *depending
    on the task*.

QUESTION\_ANSWERING\_COVID19
----------------------------

The dataset file must be of the ``.zip`` archive format, containing
`JSON <https://en.wikipedia.org/wiki/JSON>`__ files. JSON files under
different levels of folders will be automaticly read all together.

Each JSON file is extracted from one paper. `JSON
structure <https://en.wikipedia.org/wiki/JSON#Example>`__ contains field
body\_text, which is a list of {"text": <str>} blocks. Each text block
is namely each paragraph of corresponding paper.

Meanwhile, a metadata.csv file, at the root of the archive directory, is
optional. It is to provide the model with publish\_time column, each
entry is in Date format, e.g. 2001-12-17. In this condition, each
metadata entry is required to have sha value column in General format,
and each JSON file required to have "sha":<str> field, while both sha
values linked. When neither metadata.csv or publish\_time Date value is
provided, the model would not check the timeliness of corresponding JSON
body\_text field.

QUESTION\_ANSWERING\_MEDQUAD
----------------------------

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

AUDIO\_FILES
------------

The dataset file must be of the ``.zip`` archive format with a
``audios.csv`` at the root of the directory.
