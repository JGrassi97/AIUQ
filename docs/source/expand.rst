Expand the workflow
===================

Add a new model
---------------

To integrate a new AI model backend:

1. Add a model card in ``conf/cards/models/``.
2. Add platform-specific job templates in ``conf/jobs/<PLATFORM>/models/<MODEL>/``.
3. Implement or adapt run scripts in ``runscripts/``.
4. Add shell wrapper templates under ``templates/`` if needed.
5. Validate outputs against AIUQ-st variables and metadata conventions.

Add a new initial condition source
----------------------------------

1. Add an IC card in ``conf/cards/ics/``.
2. Add retrieval jobs in ``conf/jobs/<PLATFORM>/ics/<IC>/``.
3. Implement data retrieval/normalization scripts in ``runscripts/``.
4. Ensure fallback logic is respected for missing variables/levels.

Add a new platform
------------------

1. Create ``conf/jobs/<NEW_PLATFORM>/`` mirroring existing structure.
2. Update or add job scripts for IC retrieval, model execution, and postprocessing.
3. Check filesystem paths, module loads, and Singularity execution policy.

Checklist before merge
----------------------

- Jobs pass in at least one short end-to-end test.
- New cards and templates are documented.
- ``README.md`` and this documentation are updated.
- Backward compatibility with existing platforms is preserved.