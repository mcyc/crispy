.. _{{ fullname }}:

{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

.. automethod:: {{ fullname }}
   :noindex:

{% if example -%}
Examples
--------

.. code-block:: python

   {{ example }}
{% endif -%}
