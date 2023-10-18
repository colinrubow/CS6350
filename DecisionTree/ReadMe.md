A class for running the ID3 decision tree machine learning algorithm. To use:
1) create the instance with my_tree = ID3()
2) train the model with my_tree.train_model(*args)
note that attributes is a dictionary where the keys are the different attributes and the values are the different values that attribute may take on.
If the attribute's values are numeric, simply make that attribute's value the string 'numeric' and set the function's argument contains_numeric to True.
If the dataset has values that are unknown there are two methods for handling this. The first is to include the string 'unknown' as a value in the attribute list.
The second way is to not include the string 'unknown' as a value in the attribute list. Any value that appears in the dataset, that is not included in the
attribute's list will be handled by assuming the most common value. When testing the model, the contains_unknown argument is True when the string
'unknown' is not included in the attribute's values when the model was trained.
