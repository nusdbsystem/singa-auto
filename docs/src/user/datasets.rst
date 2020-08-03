Dataset Types
====================================================================

.. note::

    Refer to `./examples/datasets/ <https://github.com/nusdbsystem/singa-auto/tree/master/examples/datasets/>`_ for examples on pre-processing 
    common dataset formats to conform to the SINGA-Auto's own dataset formats.


.. _`dataset-type:CORPUS`:

CORPUS
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``corpus.tsv`` at the root of the directory.

The ``corpus.tsv`` should be of a `.TSV <https://en.wikipedia.org/wiki/Tab-separated_values>`_ 
format with columns of ``token`` and ``N`` other variable column names (*tag columns*).

For each row,

    ``token`` should be a string, a token (e.g. word) in the corpus. 
    These tokens should appear in the order as it is in the text of the corpus.
    To delimit sentences, ``token`` can be take the value of ``\n``.

    The other ``N`` columns describe the corresponding token as part of the text of the corpus, *depending on the task*.


.. _`dataset-type:IMAGE_FILES`:

IMAGE_FILES
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``images.csv`` at the root of the directory.

The ``images.csv`` should be of a `.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_
format with columns of ``path`` and ``N`` other variable column names (*tag columns*).

For each row,

    ``path`` should be a file path to a ``.png``, ``.jpg`` or ``.jpeg`` image file within the archive, 
    relative to the root of the directory.

    The other ``N`` columns describe the corresponding image, *depending on the task*.

.. _`dataset-type:QUESTION_ANSWERING_COVID19`:

QUESTION_ANSWERING_COVID19
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format, containing `JSON <https://en.wikipedia.org/wiki/JSON>`_ files. JSON files under different levels of folders will be automaticly read all together.

Each JSON file is extracted from one paper. `JSON structure <https://en.wikipedia.org/wiki/JSON#Example>`_ contains field `body_text`, which is a list of `{"text": <str>}` blocks. Each `text` block is namely each paragraph of corresponding paper.

Meanwhile, a `metadata.csv` file, at the root of the archive directory, is optional. It is to provide the model with `publish_time` column, each entry is in Date format, e.g. 2001-12-17. In this condition, each metadata entry is required to have `sha` value column in General format, and each JSON file required to have `"sha":<str>` field, while both sha values linked. When neither metadata.csv or `publish_time` Date value is provided, the model would not check the timeliness of  corresponding JSON `body_text` field.


.. _`dataset-type:QUESTION_ANSWERING_MEDQUAD`:

QUESTION_ANSWERING_MEDQUAD
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format, containing `xml <https://en.wikipedia.org/wiki/XML#/media/File:XMLSample.png>`_ files. Xml files under different levels of folders will be automaticly read all together.

Model would only take <Document> <QAPairs> ... </QAPairs> </Document>field, and this filed contains multiple <QAPair> ... </QAPair>. Each QAPair has one <Question> ... </Question> and its <Answer> ... </Answer> combination. 


.. _`dataset-type:TABULAR`:

TABULAR
--------------------------------------------------------------------

The dataset file must be a tabular dataset of the ``.csv`` format with ``N`` columns.

.. _`dataset-type:AUDIO_FILES`:

AUDIO_FILES
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``audios.csv`` at the root of the directory.

