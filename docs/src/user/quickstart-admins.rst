Quick Start (Admins)
====================================================================

As an *Admin*, you can manage users, datasets, models, train jobs & inference jobs on SINGA-Auto. This guide only highlights the key methods available to manage users.

To learn about how to manage models, go to :ref:`quickstart-model-developers`.

To learn about how to manage train & inference jobs, go to :ref:`quickstart-app-developers`.

This guide assumes that you have access to a running instance of *SINGA-Auto Admin* at ``<singa_auto_host>:<admin_port>``, e.g., ``127.0.0.1:3000``, 
and *SINGA-Auto Web Admin* at ``<singa_auto_host>:<web_admin_port>``, e.g., ``127.0.0.1:3001``.

Installation
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the client
--------------------------------------------------------------------

Example:

    .. code-block:: python

        from singa_auto.client import Client
        client = Client(admin_host='localhost', admin_port=3000) # 'localhost' can be replaced by '127.0.0.1' or other server address
        client.login(email='superadmin@singaauto', password='singa_auto')

.. seealso:: :meth:`singa_auto.client.Client.login`
        
Creating users
--------------------------------------------------------------------

Examples:

    .. code-block:: python

        client.create_user(
            email='admin@singaauto',
            password='singa_auto',
            user_type='ADMIN'
        )
        
        client.create_user(
            email='model_developer@singaauto',
            password='singa_auto',
            user_type='MODEL_DEVELOPER'
        )

        client.create_user(
            email='app_developer@singaauto',
            password='singa_auto',
            user_type='APP_DEVELOPER'
        )


.. seealso:: :meth:`singa_auto.client.Client.create_user`


Listing all users
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.get_users()
    

    .. code-block:: python

        [{'email': 'superadmin@singaauto',
        'id': 'c815fa08-ce06-467d-941b-afc27684d092',
        'user_type': 'SUPERADMIN'},
        {'email': 'admin@singaauto',
        'id': 'cb2c0d61-acd3-4b65-a5a7-d78aa5648283',
        'user_type': 'ADMIN'},
        {'email': 'model_developer@singaauto',
        'id': 'bfe58183-9c69-4fbd-a7b3-3fdc267b3290',
        'user_type': 'MODEL_DEVELOPER'},
        {'email': 'app_developer@singaauto',
        'id': '958a7d65-aa1d-437f-858e-8837bb3ecf32',
        'user_type': 'APP_DEVELOPER'}]
        

.. seealso:: :meth:`singa_auto.client.Client.get_users`


Banning a user
--------------------------------------------------------------------

Example:

    .. code-block:: python

        client.ban_user('app_developer@singaauto')
    

.. seealso:: :meth:`singa_auto.client.Client.ban_user`
