# Codeguidelines


- Comments, documentation, filepaths and outputs (like logging messages) are to be written in english!

- Always use the custom logger in `logger_setup.py` and the custom `cli_ui.py`, that are part of the installable python module `data_fusion_project`!

- Keep `logger` and `raise` statements strictly one line in code (other statements don’t have to adher to this rule).

- Always comment functions (or larger sections or significant files) according to the following principle:
    ```
    """
    description.
    :param: <parameter_name> (<parameter_data_type>): dascription.
    :return: <parameter_name> (<parameter_data_type>): dascription.
    :raises: <error_name>: description.
    """
    ```
  
- If your code reads data, always specify the path to this data in the project directory as follows:
    ```
    Input:
    drectory/
    ├── directory/
    │   └── filename.filetype
    ...
    ```

- If your code writes data, always specify the path to this data in the project directory as follows:
    ```
    Output:
    drectory/
    ├── directory/
    │   └── filename.filetype
    ...
    ```

- If your code creates tabular files such as .pkl, .csv, .xlsx, ects. always include a table in the documentation with a description of the file:
    ```
    **Columns (filename.filetype):**
    | Column name | Description |
    |-------------|-------------|
    | ...         | ...         |
    ```

- Structure Code you write with the following markers:
    ```python
    # ======================================================================================================================
    # Section Title
    # ======================================================================================================================
    ```

- Keep three free lines between functions.