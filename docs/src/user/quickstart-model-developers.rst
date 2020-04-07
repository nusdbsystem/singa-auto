.. _`quickstart-model-developers`:

Quick Start (Model Developers)
====================================================================

As a *Model Developer*, you can manage models, datasets, train jobs & inference jobs on Singa-Auto. This guide only highlights the key methods available to manage models.

To learn about how to manage datasets, train jobs & inference jobs, go to :ref:`quickstart-app-developers`.

This guide assumes that you have access to a running instance of *Singa-Auto Admin* at ``<singa_auto_host>:<admin_port>``
and *Singa-Auto Web Admin* at ``<singa_auto_host>:<web_admin_port>``.

To learn more about what else you can do on Singa-Auto, explore the methods of :class:`singa_auto.client.Client`

Installing the client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the client
--------------------------------------------------------------------

Example:

    .. code-block:: python

        from singa_auto.client import Client
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='model_developer@singa_auto', password='singa_auto')

.. seealso:: :meth:`singa_auto.client.Client.login`

Creating models
--------------------------------------------------------------------

.. include:: ./client-create-models.include.rst


Listing available models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst


Deleting a model
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.delete_model('fb5671f1-c673-40e7-b53a-9208eb1ccc50')

.. seealso:: :meth:`singa_auto.client.Client.delete_model`
