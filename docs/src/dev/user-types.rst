.. _`user-types`:

Users of Singa-Auto
====================================================================

.. figure:: ../images/system-context-diagram.jpg
    :align: center
    :width: 500px

    Users of Singa-Auto

There are 4 types of users on Singa-Auto:

    *Application Developers* create, manage, monitor and stop model training and serving jobs on Singa-Auto.
    They are the primary users of Singa-Auto - they upload their datasets onto Singa-Auto and create model training jobs that train on these datasets.
    After model training, they trigger the deployment of these trained ML models as a web service that Application Users interact with.
    While their model training and serving jobs are running, they administer these jobs and monitor their progress.

    *Application Users* send queries to trained models exposed as a web service on Singa-Auto, receiving predictions back.
    Not to be confused with Application Developers, these users may be developers that are looking to conveniently integrate ML predictions into their mobile, web or desktop applications.
    These application users have consumer-provider relationships with the aforementioned ML application developers, having delegated the work of training and deploying ML models to them.

    *Model Developers* create, update and delete model templates to form Singa-Auto’s dynamic repository of ML model templates.
    These users are key external contributors to Singa-Auto, and represent the main source of up-to-date ML expertise on Singa-Auto,
    playing a crucial role in consistently expanding and diversifying Singa-Auto’s underlying set of ML model templates for a variety of ML tasks.
    Coupled with Singa-Auto’s modern ML model tuning framework on Singa-Auto, these contributions heavily dictate the ML performance that Singa-Auto provides to Application Developers.

    *Singa-Auto Admins* create, update and remove users on Singa-Auto. They regulate access of the other types of users to a running instance of Singa-Auto.
